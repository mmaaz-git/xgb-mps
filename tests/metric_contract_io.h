// Complete metal_metric_api corpus, including math, label guards and empty calls.
// Inputs never contain references. Reduction diagnostics in observations are
// computed from each backend's own observed row terms, not another GPU's terms.
#pragma once
#include <cmath>
#include <memory>
#include "ranking_rounding_io.h"
#include "cuda_metric_reference.h"

namespace metric_contracts {
using ranking_rounding::Bits;
using ranking_rounding::Read;
using ranking_rounding::Require;
using ranking_rounding::Values;
using ranking_rounding::Write;
constexpr unsigned kCases=375, kInput=0, kMetal=1, kCuda=2;
inline constexpr char kMagic[8]={'X','G','B','M','E','T','1','\0'};
struct Spec {
  std::uint32_t id,family,kind,rows,targets,levels,mode,accuracy,predictions,labels,upper,weights,
                alphas,fields,parameter;
};
inline std::array<std::uint32_t,15> Identity(Spec s) {
  return {s.id,s.family,s.kind,s.rows,s.targets,s.levels,s.mode,s.accuracy,s.predictions,s.labels,
          s.upper,s.weights,s.alphas,s.fields,s.parameter};
}
inline float Parameter(unsigned kind) { return kind==9 ? 1.3712345f : kind==10 ? 1.3f : .371234567f; }
inline std::vector<Spec> Specs() {
  std::vector<Spec> r;
  auto add=[&](unsigned family,unsigned kind,unsigned rows,unsigned targets,unsigned levels,
               unsigned mode,unsigned accuracy,unsigned p,unsigned y,unsigned u,unsigned w,
               unsigned a,unsigned fields,float parameter) {
    r.push_back({unsigned(r.size()),family,kind,rows,targets,levels,mode,accuracy,p,y,u,w,a,fields,
                 Bits<std::uint32_t>(parameter)});
  };
  add(0,0,100057,1,1,0,0,100057,0,0,0,0,500285,0.f);
  auto ordinary=[&](unsigned kind,unsigned rows,unsigned mode) {
    unsigned targets=kind>=11 ? 7 : 3, labels=rows*(kind>=11 ? 1 : targets);
    add(1,kind,rows,targets,1,mode,0,rows*targets,labels,0,mode ? rows : 0,0,2*labels,Parameter(kind));
  };
  for (unsigned kind=0; kind<13; ++kind) {
    for (unsigned rows : {1u,257u,4097u}) for (unsigned mode=0; mode<7; ++mode) ordinary(kind,rows,mode);
    ordinary(kind,65537,0);
  }
  ordinary(2,262145,3);
  for (unsigned expectile : {0u,1u}) {
    for (unsigned rows : {1u,257u,4097u}) for (unsigned mode=0; mode<5; ++mode)
      add(2,expectile,rows,3,7,mode,0,rows*21,rows*3,0,rows,7,rows*42,0.f);
    add(2,expectile,65537,3,7,0,0,65537*21,65537*3,0,65537,7,65537*42,0.f);
  }
  for (unsigned distribution=0; distribution<3; ++distribution) for (unsigned mode=0; mode<4; ++mode)
    for (unsigned accuracy : {0u,1u})
      add(3,distribution,65537,1,1,mode,accuracy,65537,65537,65537,65537,0,0,1.f);
  for (unsigned distribution=0; distribution<3; ++distribution) for (unsigned mode=0; mode<2; ++mode)
    add(4,distribution,5,1,1,mode,0,5,5,5,5,0,0,1.f);
  for (unsigned kind : {11u,12u}) for (unsigned mode=0; mode<6; ++mode)
    add(5,kind,1,3,1,mode,0,3,1,0,0,0,0,.5f);
  for (unsigned kind=0; kind<13; ++kind) add(6,kind,0,1,1,0,0,0,0,0,0,0,0,.5f);
  Require(r.size()==kCases,"Incomplete metric control matrix"); return r;
}
struct Input { Spec spec; std::vector<float> predictions,labels,upper,weights,alphas; };
struct Reduction { double sum,magnitude; };
inline std::array<Reduction,2> ReductionReferences(std::vector<float> const& pairs) {
  Require(pairs.size()%2==0,"Odd metric pair size"); std::array<cuda_metric::Sum,2> sums{};
  for (std::size_t i=0; i<pairs.size(); ++i) sums[i%2].Add(pairs[i]);
  return {{{sums[0].Result(),sums[0].magnitude},{sums[1].Result(),sums[1].magnitude}}};
}
struct Observation {
  // 1: actual caught invalid-label error. 2: audited CUDA-domain exclusion,
  // not executed. Neither status writes fabricated output sums.
  unsigned status{0};
  std::vector<float> actual,reference;
  std::array<double,2> sums{},analytic{};
  std::array<Reduction,2> reductions{};
};
inline void Validate(Input const& in,Spec s) {
  Require(Identity(in.spec)==Identity(s) && in.predictions.size()==s.predictions &&
          in.labels.size()==s.labels && in.upper.size()==s.upper && in.weights.size()==s.weights &&
          in.alphas.size()==s.alphas,"Wrong metric identity/order/input shape");
}
class Writer {
  std::ofstream out_;
  unsigned kind_,index_{0};
  std::vector<Spec> specs_{Specs()};
 public:
  Writer(std::string const& path,unsigned kind) : kind_{kind} {
    Require(kind<=kCuda && !std::filesystem::exists(path),"Unknown backend or existing metric report");
    out_.exceptions(std::ios::badbit|std::ios::failbit); out_.open(path,std::ios::binary);
    out_.write(kMagic,8); Write(out_,kind); Write(out_,kCases);
  }
  void Add(Input const& in,Observation const& o={}) {
    Require(index_<kCases,"Too many metric records"); Validate(in,specs_[index_]); auto s=in.spec;
    bool excluded=kind_==kCuda && s.family==5 && s.mode>=2;
    Require(excluded ? o.status==2 : o.status<=(s.family==5 && kind_ ? 1u : 0u),
            "Unexpected metric error/domain status");
    Require(o.actual.size()==(kind_ ? s.fields : 0),"Wrong metric observed field count");
    Require(o.reference.size()==(kind_==kMetal ? s.fields : 0),"Wrong metric reference count");
    if (!kind_) Require(o.status==0,"Input report contains metric observations");
    for (auto word : Identity(s)) Write(out_,word);
    Values(out_,in.predictions); Values(out_,in.labels); Values(out_,in.upper);
    Values(out_,in.weights); Values(out_,in.alphas);
    if (kind_) {
      Write(out_,o.status); Values(out_,o.actual);
      if (s.family!=0 && !o.status) for (double v : o.sums) Write(out_,Bits<std::uint64_t>(v));
      if (s.family==1 || s.family==2) for (auto r : o.reductions) {
        Write(out_,Bits<std::uint64_t>(r.sum)); Write(out_,Bits<std::uint64_t>(r.magnitude));
      }
    }
    if (kind_==kMetal) {
      Values(out_,o.reference);
      if (s.family==3 || s.family==4) for (double v : o.analytic) Write(out_,Bits<std::uint64_t>(v));
    }
    ++index_; out_.flush();
  }
  void Finish() {
    Require(index_==kCases,"Incomplete metric corpus"); out_.write("METE",4); Write(out_,index_); out_.flush();
  }
};
class InputReader {
  std::ifstream in_;
  unsigned index_{0};
  std::vector<Spec> specs_{Specs()};
 public:
  explicit InputReader(std::string const& path) {
    in_.exceptions(std::ios::badbit|std::ios::failbit); in_.open(path,std::ios::binary);
    char magic[8]; in_.read(magic,8);
    Require(!std::memcmp(magic,kMagic,8),"Unknown metric interchange version");
    Require(Read<unsigned>(in_)==kInput && Read<unsigned>(in_)==kCases,
            "CUDA requires complete input-only metric records");
  }
  Input Next() {
    Require(index_<kCases,"Too many metric inputs"); Input in{specs_[index_],{},{},{},{},{}};
    for (auto word : Identity(in.spec)) Require(Read<std::uint32_t>(in_)==word,"Wrong metric identity/order/shape");
    in.predictions=Values<float>(in_,in.spec.predictions); in.labels=Values<float>(in_,in.spec.labels);
    in.upper=Values<float>(in_,in.spec.upper); in.weights=Values<float>(in_,in.spec.weights);
    in.alphas=Values<float>(in_,in.spec.alphas); ++index_; return in;
  }
  void Finish() {
    Require(index_==kCases,"Unconsumed metric inputs"); char end[4]; in_.read(end,4);
    Require(!std::memcmp(end,"METE",4) && Read<unsigned>(in_)==kCases,"Missing metric completion marker");
    Require(in_.peek()==std::char_traits<char>::eof(),"Trailing metric bytes");
  }
};
class Exporter {
  unsigned index_{0};
  Writer inputs_,metal_;
 public:
  explicit Exporter(std::string const& prefix)
      : inputs_{prefix+".inputs.bin",kInput},metal_{prefix+".metal.bin",kMetal} {}
  void Add(unsigned family,unsigned kind,unsigned rows,unsigned targets,unsigned levels,unsigned mode,
           unsigned accuracy,float parameter,std::vector<float> const& p,std::vector<float> const& y,
           std::vector<float> const& u,std::vector<float> const& w,std::vector<float> const& a,
           Observation const& o) {
    Input in{{index_,family,kind,rows,targets,levels,mode,accuracy,unsigned(p.size()),unsigned(y.size()),
              unsigned(u.size()),unsigned(w.size()),unsigned(a.size()),unsigned(o.actual.size()),Bits<std::uint32_t>(parameter)},p,y,u,w,a};
    inputs_.Add(in); metal_.Add(in,o); ++index_;
  }
  void Finish() { inputs_.Finish(); metal_.Finish(); }
};
inline std::unique_ptr<Exporter> ReportOption(int argc,char** argv) {
  Require(argc==2 || (argc==4 && std::string(argv[2])=="--report-prefix"),
          "Usage: native-test production-shader [--report-prefix new-prefix]");
  if (argc==2) return {};
  std::string prefix=argv[3];
  Require(!std::filesystem::exists(prefix+".inputs.bin") && !std::filesystem::exists(prefix+".metal.bin"),
          "Use a new metric report prefix");
  return std::make_unique<Exporter>(prefix);
}
}  // namespace metric_contracts
