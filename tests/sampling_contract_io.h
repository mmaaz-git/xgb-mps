// Complete current CUDA-derived sampling corpus, including lifecycle and guards.
#pragma once
#include <algorithm>
#include <memory>
#include <random>
#include <sstream>
#include "ranking_rounding_io.h"

namespace sampling_contracts {
using ranking_rounding::Bits;
using ranking_rounding::Read;
using ranking_rounding::Require;
using ranking_rounding::Values;
using ranking_rounding::Write;
using Words=std::vector<std::uint64_t>;
constexpr unsigned kCases=199, kInput=0, kMetal=1, kCuda=2;
inline constexpr char kMagic[8]={'X','G','B','S','M','P','1','\0'};
struct Spec {
  std::uint32_t id,family,rows,targets,layout,mode,fraction,seed,threads,sequence,step,domain;
};
inline std::array<std::uint32_t,12> Identity(Spec s) {
  return {s.id,s.family,s.rows,s.targets,s.layout,s.mode,s.fraction,s.seed,s.threads,s.sequence,s.step,s.domain};
}
inline bool OutsideRescale(unsigned id) {
  for (unsigned excluded:{114u,117u,126u,129u,132u,135u,138u}) if (id==excluded) return true;
  return false;
}
inline std::vector<Spec> Specs() {
  std::vector<Spec> result; unsigned accepted=0;
  auto add=[&](unsigned rows,unsigned targets,unsigned layout,unsigned mode,float fraction,unsigned seed) {
    unsigned id=result.size(),domain=OutsideRescale(id);
    result.push_back({id,0,rows,targets,layout,mode,Bits<std::uint32_t>(fraction),seed,0,0,0,domain});
    accepted+=!domain;
  };
  for (unsigned rows:{0u,1u,2u,255u,256u,257u,1027u}) for (unsigned targets:{1u,3u})
    for (unsigned mode=0;mode<8;++mode) {
      float p[]{0,.001f,.37f,.999f,1}; add(rows,targets,accepted%4,mode,p[accepted%5],accepted*7919);
    }
  for (unsigned targets:{16u,17u,31u,32u,33u,511u,512u,513u,1027u})
    for (unsigned mode:{0u,3u,4u}) add(33,targets,mode%3,mode,.6f,793);
  for (unsigned rows:{65537u,1000003u}) for (unsigned targets:{1u,3u})
    add(rows,targets,targets==1?0:1,0,.4f,9939);
  Require(result.size()==143 && accepted==136,"Incomplete sampling primitive matrix");
  unsigned sequence=0;
  for (unsigned rows:{0u,1u,257u}) for (float fraction:{0.f,.6f,1.f}) for (unsigned threads:{1u,3u}) {
    for (unsigned step=0;step<3;++step)
      result.push_back({unsigned(result.size()),1,rows,1,0,0,Bits<std::uint32_t>(fraction),701,threads,sequence,step,0});
    ++sequence;
  }
  for (float fraction:{0.f,.5f})
    result.push_back({unsigned(result.size()),2,4,1,0,0,Bits<std::uint32_t>(fraction),1,0,0,0,unsigned(fraction>0)});
  Require(result.size()==kCases,"Incomplete sampling corpus"); return result;
}
struct Shape { unsigned stride0,stride1,offset,extent,storage,replay_targets,replay_stride,replay_storage; };
inline Shape Layout(Spec s) {
  if (s.family==2) return {1,1,0,4,4,0,0,0};
  if (s.family==1) return {1,1,0,s.rows,s.rows,3,3,3*s.rows};
  unsigned a=s.layout==1?1:s.layout==2?2*s.targets+1:s.targets+2;
  if (s.layout==3) a=0;
  unsigned b=s.layout==1?s.rows+3:s.layout==2?2:1;
  unsigned extent=s.rows?(s.rows-1)*a+(s.targets-1)*b+1:0;
  unsigned rt=s.targets==1?3:1;
  return {a,b,4,extent,extent+8,rt,rt+2,s.domain?0:s.rows*(rt+2)+8};
}
enum Policy:unsigned { kFloatExact,kFloatNaN,kDoubleNumeric,kIntegerExact,kRngExact };
inline std::vector<unsigned> Policies(Spec s) {
  std::vector<unsigned> result; auto shape=Layout(s);
  if (s.domain || s.family==2) return result;
  auto append=[&](unsigned n,unsigned p){result.insert(result.end(),n,p);};
  auto roots=[&]{append(2*s.targets,kDoubleNumeric);append(2*s.targets,kIntegerExact);};
  if (s.family==0 && Bits<float>(s.fraction)<1) append(s.rows+1,kFloatNaN);
  roots();
  if (s.family==1) append(2*shape.storage,kFloatNaN);
  append(2*shape.replay_storage,s.family==0?kFloatExact:kFloatNaN);
  if (s.family==0) roots(); else append(1,kRngExact);
  return result;
}
struct Input { Spec spec; std::vector<float> storage,replay; };
struct Observation { unsigned status{0}; Words actual,reference; };
inline void Validate(Input const& in,Spec expected) {
  auto shape=Layout(expected);
  Require(Identity(in.spec)==Identity(expected) && in.storage.size()==2*shape.storage &&
          in.replay.size()==2*shape.replay_storage,"Wrong sampling identity/order/input shape");
}
class Writer {
  std::ofstream out_; unsigned kind_,index_{0}; std::vector<Spec> specs_{Specs()};
 public:
  Writer(std::string const& path,unsigned kind):kind_{kind} {
    Require(kind<=kCuda && !std::filesystem::exists(path),"Unknown sampling backend or existing output");
    out_.exceptions(std::ios::badbit|std::ios::failbit); out_.open(path,std::ios::binary);
    out_.write(kMagic,8); Write(out_,std::uint32_t{kind});Write(out_,std::uint32_t{kCases});
  }
  void Add(Input const& in,Observation const& out={}) {
    Require(index_<kCases,"Too many sampling cases"); Validate(in,specs_[index_]);
    auto policies=kind_?Policies(in.spec):std::vector<unsigned>{};
    Require(out.status==(kind_ && in.spec.domain ? (kind_==kMetal?1u:2u) : 0u),"Wrong sampling domain status");
    Require(out.actual.size()==(kind_?policies.size():0) && out.reference.size()==(kind_==kMetal?policies.size():0),
            "Wrong sampling observation shape/backend");
    for (auto const* words:{&out.actual,&out.reference})
      for (std::size_t i=0;i<words->size();++i)
        if (policies[i]==kFloatExact || policies[i]==kFloatNaN || policies[i]==kRngExact)
          Require((*words)[i]<=UINT32_MAX,"Nonzero upper sampling FP32/RNG bits");
    for (auto word:Identity(in.spec)) Write(out_,word);
    Values(out_,in.storage); Values(out_,in.replay);
    if (kind_) Write(out_,std::uint32_t{out.status});
    for (auto word:out.actual) Write(out_,word);
    for (auto word:out.reference) Write(out_,word);
    ++index_;out_.flush();
  }
  void Finish() {
    Require(index_==kCases,"Incomplete sampling corpus");out_.write("SMPE",4);Write(out_,index_);out_.flush();
  }
};
class InputReader {
  std::ifstream in_;unsigned index_{0};std::vector<Spec> specs_{Specs()};
 public:
  explicit InputReader(std::string const& path) {
    in_.exceptions(std::ios::badbit|std::ios::failbit);in_.open(path,std::ios::binary);
    char magic[8];in_.read(magic,8);
    Require(!std::memcmp(magic,kMagic,8) && Read<std::uint32_t>(in_)==kInput && Read<std::uint32_t>(in_)==kCases,
            "CUDA requires complete input-only sampling records");
  }
  Input Next() {
    Require(index_<kCases,"Too many sampling inputs");Input in{specs_[index_],{}, {}};
    for (auto word:Identity(in.spec)) Require(Read<std::uint32_t>(in_)==word,"Wrong sampling identity/order/parameters");
    auto shape=Layout(in.spec);in.storage=Values<float>(in_,2*shape.storage);in.replay=Values<float>(in_,2*shape.replay_storage);
    ++index_;return in;
  }
  void Finish() {
    Require(index_==kCases,"Unconsumed sampling inputs");char footer[4];in_.read(footer,4);
    Require(!std::memcmp(footer,"SMPE",4) && Read<std::uint32_t>(in_)==kCases &&
            in_.peek()==std::char_traits<char>::eof(),"Missing sampling completion marker or trailing bytes");
  }
};
class Exporter {
  Writer inputs_,metal_;
 public:
  explicit Exporter(std::string const& prefix):inputs_{prefix+".inputs.bin",kInput},metal_{prefix+".metal.bin",kMetal} {}
  void Add(Input const& in,Observation const& out) {inputs_.Add(in);metal_.Add(in,out);}
  void Finish(){inputs_.Finish();metal_.Finish();}
};
inline std::unique_ptr<Exporter> ReportOption(int argc,char** argv) {
  Require(argc==1 || (argc==3 && std::string(argv[1])=="--report-prefix"),"Usage: metal_cuda_gradient_api [--report-prefix new-prefix]");
  if(argc==1)return {};
  std::string p=argv[2];Require(!std::filesystem::exists(p+".inputs.bin") && !std::filesystem::exists(p+".metal.bin"),"Use a new sampling report prefix");
  return std::make_unique<Exporter>(p);
}
// Test-supplied MT19937 state, NOT seed(row_seed). Both supported standard
// library serializations describe a state before the next twist. The first
// twisted word is x[397] when x[0]=x[1]=0, then output tempering produces seed.
inline std::mt19937 RowSeedEngine(std::uint32_t seed) {
  auto right=[](std::uint32_t x,int shift){auto y=x;for(int i=0;i<32;i+=shift)y=x^(y>>shift);return y;};
  auto left=[](std::uint32_t x,int shift,std::uint32_t mask){auto y=x;for(int i=0;i<32;i+=shift)y=x^((y<<shift)&mask);return y;};
  auto word=right(seed,18);word=left(word,15,0xefc60000u);word=left(word,7,0x9d2c5680u);word=right(word,11);
  std::ostringstream encoded;encoded<<std::mt19937{701};std::istringstream input(encoded.str());
  std::uint64_t value;unsigned count=0;while(input>>value)++count;
  Require(count==624 || count==625,"Unsupported MT19937 serialization; cannot align row seed");
  std::ostringstream state;
  for(unsigned i=0;i<624;++i)state<<(i==397?word:i==623?1u:0u)<<' ';
  if(count==625)state<<624;  // libstdc++ also serializes the position; libc++ rotates its state.
  std::mt19937 engine;std::istringstream parse(state.str());parse>>engine;
  auto verify=engine;Require(!parse.fail() && verify()==seed,"MT19937 first draw does not match requested row seed");
  return engine;
}
}  // namespace sampling_contracts
