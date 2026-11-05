// Complete original linear corpus. Each action carries its own input state;
// this permits defined CUDA operations after a Metal-only duplicate-row update
// without claiming that CUDA executed the undefined intermediate update.
#pragma once
#include <memory>
#include <optional>
#include "ranking_rounding_io.h"

namespace linear_contracts {
using ranking_rounding::Bits;
using ranking_rounding::Read;
using ranking_rounding::Require;
using ranking_rounding::Values;
using ranking_rounding::Write;
constexpr unsigned kCases=56, kSteps=893, kInput=0, kMetal=1, kCuda=2;
inline constexpr char kMagic[8]={'X','G','B','L','I','N','1','\0'};
enum Family : unsigned { kMain, kNegative, kOrder, kOverflow, kZero,
                         kInvalidFeatureLow, kInvalidFeatureHigh, kInvalidRow };
struct Spec { std::uint32_t id,family,rows,groups,mode,features; };
inline std::array<std::uint32_t,6> Identity(Spec s) {
  return {s.id,s.family,s.rows,s.groups,s.mode,s.features};
}
inline std::vector<Spec> Specs() {
  std::vector<Spec> out;
  auto add=[&](unsigned family,unsigned rows,unsigned groups,unsigned mode,unsigned features){
    out.push_back({unsigned(out.size()),family,rows,groups,mode,features});
  };
  add(kNegative,1,1,0,1);add(kOrder,1,1,0,1);add(kOverflow,2,1,0,0);add(kZero,1,1,0,1);
  for(unsigned rows:{0u,1u,31u,257u,4097u,65539u})
    for(unsigned groups:{1u,3u})for(unsigned mode=0;mode<4;++mode)add(kMain,rows,groups,mode,5);
  add(kMain,1000003,3,0,5);
  for(unsigned family:{kInvalidFeatureLow,kInvalidFeatureHigh,kInvalidRow})add(family,1,1,0,1);
  Require(out.size()==kCases,"Incomplete linear corpus");return out;
}
struct Action { std::int32_t feature;bool gradient,update,copy;float delta;unsigned domain; };
// domain0: common operations; domain1: CUDA gradient defined, duplicate residual
// writes excluded; domain2: Metal rejection, unsafe CUDA access not executed.
inline std::vector<Action> Actions(Spec s) {
  std::vector<Action> out;
  if(s.family==kMain) {
    for(float delta:{0.f,.125f,-.5f})for(int feature=-1;feature<5;++feature)
      out.push_back({feature,true,true,true,delta,unsigned(feature==2 && s.rows>0)});
  } else if(s.family==kNegative) {
    out={{-1,true,false,false,0,0},{0,true,false,false,0,0},
         {-1,false,true,true,.5f,0},{0,false,true,true,.5f,0}};
  } else if(s.family==kOrder) {
    out={{0,false,true,true,Bits<float>(std::uint32_t{0x3fd4c3f0}),0}};
  } else if(s.family==kOverflow) {
    out={{-1,true,false,false,0,0}};
  } else if(s.family==kZero) {
    out={{-1,false,true,true,0,0},{0,false,true,true,0,0}};
  } else {
    out={{s.family==kInvalidFeatureLow?-2:s.family==kInvalidFeatureHigh?1:0,
          s.family!=kInvalidRow,false,false,0,2}};
  }
  return out;
}
inline std::array<std::uint32_t,7> Identity(unsigned index,Action a) {
  return {index,std::uint32_t(a.feature),unsigned(a.gradient),unsigned(a.update),
          unsigned(a.copy),Bits<std::uint32_t>(a.delta),a.domain};
}
inline std::size_t Entries(Spec s) {
  if(s.family!=kMain)return s.features;
  return 2ull*s.rows+(s.rows+1)/2+(s.rows+6)/7+(s.rows+96)/97;
}
inline std::size_t StateWords(Spec s) {return 2ull*s.rows*s.groups;}
inline unsigned Status(unsigned kind,Action a) {
  return kind==kInput?0:a.domain==2?(kind==kMetal?1:2):(kind==kCuda && a.domain==1?3:0);
}
struct Input {
  Spec spec;
  std::vector<std::uint64_t> offsets;
  std::vector<std::uint32_t> rows;
  std::vector<float> entries,original;
};
struct Observation {unsigned status{0};std::vector<float> gradients,residuals;};
inline void Validate(Input const& in,Spec s) {
  Require(Identity(in.spec)==Identity(s) && in.offsets.size()==s.features+1 &&
          in.rows.size()==Entries(s) && in.entries.size()==Entries(s) &&
          in.original.size()==StateWords(s)+(s.family==kMain?4:0),"Wrong linear input identity/shape");
}
class Writer {
  std::ofstream out_;
  unsigned kind_,cases_{0},step_{0},total_{0};bool pending_{false};
  std::optional<Spec> current_;std::vector<Action> actions_;std::vector<Spec> specs_{Specs()};
 public:
  Writer(std::string const& path,unsigned kind):kind_{kind} {
    Require(kind<=kCuda && !std::filesystem::exists(path),"Unknown linear backend or existing output");
    out_.exceptions(std::ios::badbit|std::ios::failbit);out_.open(path,std::ios::binary);
    out_.write(kMagic,8);Write(out_,std::uint32_t{kind});Write(out_,std::uint32_t{kCases});
  }
  void Start(Input const& in) {
    Require(cases_<kCases && !pending_ && step_==actions_.size(),"Unfinished or excess linear case");
    Validate(in,specs_[cases_]);current_=in.spec;actions_=Actions(in.spec);step_=0;++cases_;
    for(auto v:Identity(in.spec))Write(out_,v);
    for(auto v:in.offsets)Write(out_,v);
    for(auto v:in.rows)Write(out_,v);
    Values(out_,in.entries);Values(out_,in.original);
  }
  void StartStep(std::vector<float> const& state) {
    Require(current_.has_value() && step_<actions_.size() && !pending_ &&
            state.size()==StateWords(*current_),"Wrong linear action/input state");
    for(auto v:Identity(step_,actions_[step_]))Write(out_,v);
    Values(out_,state);pending_=true;
  }
  void EndStep(Observation const& o={}) {
    Require(pending_,"Linear action has no input state");auto s=*current_;auto a=actions_[step_];
    auto gradients=kind_ && a.gradient && a.domain!=2?2*s.groups:0;
    auto residuals=kind_ && a.copy && !(kind_==kCuda && a.domain)?StateWords(s)+(s.family==kMain?4:0):0;
    Require(o.status==Status(kind_,a) && o.gradients.size()==gradients && o.residuals.size()==residuals,
            "Wrong linear observation/status/domain");
    if(kind_){Write(out_,std::uint32_t{o.status});Values(out_,o.gradients);Values(out_,o.residuals);}
    pending_=false;++step_;++total_;
    if(step_==actions_.size())out_.flush();
  }
  void Finish() {
    Require(cases_==kCases && total_==kSteps && !pending_ && step_==actions_.size(),"Incomplete linear corpus/actions");
    out_.write("LINE",4);Write(out_,cases_);Write(out_,total_);out_.flush();
  }
};
class InputReader {
  std::ifstream in_;unsigned cases_{0},step_{0},total_{0};
  std::optional<Spec> current_;std::vector<Action> actions_;std::vector<Spec> specs_{Specs()};
 public:
  explicit InputReader(std::string const& path) {
    in_.exceptions(std::ios::badbit|std::ios::failbit);in_.open(path,std::ios::binary);char magic[8];in_.read(magic,8);
    Require(!std::memcmp(magic,kMagic,8) && Read<std::uint32_t>(in_)==kInput && Read<std::uint32_t>(in_)==kCases,
            "CUDA requires input-only complete linear records");
  }
  Input NextCase() {
    Require(cases_<kCases && step_==actions_.size(),"Unconsumed linear actions or excess case");
    auto s=specs_[cases_++];for(auto v:Identity(s))Require(Read<std::uint32_t>(in_)==v,"Wrong linear identity/order");
    current_=s;actions_=Actions(s);step_=0;Input out{s,{},{},{},{}};
    for(unsigned i=0;i<=s.features;++i)out.offsets.push_back(Read<std::uint64_t>(in_));
    for(std::size_t i=0;i<Entries(s);++i)out.rows.push_back(Read<std::uint32_t>(in_));
    out.entries=Values<float>(in_,Entries(s));out.original=Values<float>(in_,StateWords(s)+(s.family==kMain?4:0));
    return out;
  }
  std::vector<float> NextStep() {
    Require(current_.has_value() && step_<actions_.size(),"Excess linear action");
    for(auto v:Identity(step_,actions_[step_]))Require(Read<std::uint32_t>(in_)==v,"Wrong linear action/parameters/domain");
    ++step_;++total_;return Values<float>(in_,StateWords(*current_));
  }
  void Finish() {
    Require(cases_==kCases && total_==kSteps && step_==actions_.size(),"Unconsumed linear corpus/actions");
    char footer[4];in_.read(footer,4);
    Require(!std::memcmp(footer,"LINE",4) && Read<std::uint32_t>(in_)==kCases &&
            Read<std::uint32_t>(in_)==kSteps && in_.peek()==std::char_traits<char>::eof(),
            "Missing linear completion marker or trailing bytes");
  }
};
class Exporter {
  Writer inputs_,metal_;unsigned cases_{0};
 public:
  explicit Exporter(std::string const& prefix):inputs_{prefix+".inputs.bin",kInput},metal_{prefix+".metal.bin",kMetal}{}
  unsigned CaseIndex()const{return cases_;}
  void Start(Input const& in){inputs_.Start(in);metal_.Start(in);++cases_;}
  void StartStep(std::vector<float> const& state){inputs_.StartStep(state);metal_.StartStep(state);}
  void EndStep(Observation const& out){inputs_.EndStep();metal_.EndStep(out);}
  void Finish(){inputs_.Finish();metal_.Finish();}
};
inline std::unique_ptr<Exporter> ReportOption(int argc,char** argv) {
  // Preserve the existing argc>1 quick-differences mode, except for the new flag.
  if(argc==1 || std::string(argv[1])!="--report-prefix")return {};
  Require(argc==3,"Usage: metal_linear_api --report-prefix new-prefix");std::string p=argv[2];
  Require(!std::filesystem::exists(p+".inputs.bin") && !std::filesystem::exists(p+".metal.bin"),"Use a new linear report prefix");
  return std::make_unique<Exporter>(p);
}
}  // namespace linear_contracts
