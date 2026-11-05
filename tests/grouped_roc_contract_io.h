// Full original grouped ROC corpus. Common CUDA calls remove only zero-pair
// groups (length 0/1); two-row groups retain the source's invalid-query branch.
#pragma once
#include <algorithm>
#include <memory>
#include "ranking_rounding_io.h"
namespace grouped_roc {
using ranking_rounding::Bits;
using ranking_rounding::Read;
using ranking_rounding::Require;
using ranking_rounding::Values;
using ranking_rounding::Write;
constexpr unsigned kCases=278,kInput=0,kMetal=1,kCuda=2;
inline constexpr char kMagic[8]={'X','G','B','G','R','O','1','\0'};
enum Family : unsigned { kEmpty,kOverflow,kEdges,kMain,kLarge };
struct Spec { std::uint32_t id,family,layout,mode,rows,groups,weights; };
inline auto Identity(Spec s) { return std::array<std::uint32_t,7>{s.id,s.family,s.layout,s.mode,s.rows,s.groups,s.weights}; }
struct Shape { Spec spec; std::vector<std::uint32_t> ptr; };
inline std::vector<Shape> Shapes() {
  std::vector<Shape> out;
  auto add=[&](unsigned family,unsigned layout,unsigned mode,std::vector<unsigned> lengths,bool weighted) {
    std::vector<std::uint32_t> ptr{0};
    for (auto n:lengths) ptr.push_back(ptr.back()+n);
    out.push_back({{unsigned(out.size()),family,layout,mode,ptr.back(),unsigned(lengths.size()),
                   weighted ? unsigned(lengths.size()) : 0},ptr});
  };
  add(kEmpty,0,0,{},false);add(kOverflow,0,0,{3},true);
  std::vector<unsigned> edge_lengths;
  for (unsigned n:{0u,1u,2u,3u,31u,257u})
    for (unsigned w=0;w<13;++w) for (unsigned m=0;m<5;++m) edge_lengths.push_back(n);
  add(kEdges,0,0,edge_lengths,true);
  auto main=[&](unsigned n,unsigned layout,unsigned mode) {
    add(kMain,layout,mode,{n,0,n/2,1,2,3,31,0},mode!=0);
  };
  for (unsigned n:{0u,1u,2u,3u,31u,256u,257u,1025u})
    for (unsigned layout=0;layout<3;++layout) for (unsigned mode=0;mode<11;++mode) main(n,layout,mode);
  for (unsigned n:{4097u,8193u}) for (unsigned mode:{0u,1u,6u,9u}) main(n,2,mode);
  for (unsigned mode=0;mode<3;++mode) add(kLarge,0,mode,{1000003},false);
  Require(out.size()==kCases,"Incomplete grouped ROC matrix");return out;
}
inline unsigned Stride(Spec s) { return s.family==kMain ? (s.layout==0 ? 1 : s.layout==1 ? 3 : 7) : 1; }
inline unsigned Offset(Spec s) { return s.family==kMain ? 4 : 0; }
inline std::size_t Storage(Spec s) { return std::size_t{s.rows}*Stride(s)+(s.family==kMain ? 8 : 0); }
struct Input { Spec spec; std::vector<std::uint32_t> ptr; std::vector<float> predictions,labels,weights; };
inline void Validate(Input const& in,Shape const& shape) {
  Require(Identity(in.spec)==Identity(shape.spec) && in.ptr==shape.ptr &&
          in.predictions.size()==Storage(in.spec) && in.labels.size()==Storage(in.spec) &&
          in.weights.size()==in.spec.weights,"Wrong grouped ROC identity/order/input shape");
}
inline std::vector<unsigned> CommonGroups(Input const& in) {
  std::vector<unsigned> ids;
  for (unsigned g=0;g<in.spec.groups;++g) if (in.ptr[g+1]-in.ptr[g]>=2) ids.push_back(g);
  return ids;
}
inline unsigned ValidGroups(Input const& in) {
  unsigned n=0;for (unsigned g=0;g<in.spec.groups;++g) n+=in.ptr[g+1]-in.ptr[g]>=3;return n;
}
inline Input Compact(Input const& in) {
  Input c{in.spec,{0},{},{},{}};
  for (auto g:CommonGroups(in)) {
    for (unsigned row=in.ptr[g];row<in.ptr[g+1];++row) {
      c.predictions.push_back(in.predictions[Offset(in.spec)+row*Stride(in.spec)]);
      c.labels.push_back(in.labels[Offset(in.spec)+row*Stride(in.spec)]);
    }
    c.ptr.push_back(c.predictions.size());
    if (!in.weights.empty()) c.weights.push_back(in.weights[g]);
  }
  return c;
}
struct Observation {
  unsigned guarded{0},cpu_overflow_rejected{0};
  std::vector<double> original,per_query_guards,common,isolated,cpu;
};
inline void CheckObservation(Input const& in,Observation const& o,unsigned kind) {
  auto s=in.spec;bool metal=kind==kMetal,normal=kind!=kInput && s.family!=kLarge;
  auto guards=s.family==kMain ? 2*(s.groups+4) : s.family==kEdges ? 2*(s.groups+2) : 0;
  Require(o.guarded==unsigned(kind==kCuda && s.family==kLarge) &&
          o.cpu_overflow_rejected==unsigned(metal && s.family==kOverflow),"Wrong grouped ROC guard status");
  Require(o.original.size()==(metal ? 2u : 0u) &&
          o.per_query_guards.size()==(metal ? guards : 0) &&
          o.common.size()==(normal ? 2u : 0u) &&
          o.isolated.size()==(normal ? 2*s.groups : 0u) &&
          o.cpu.size()==(metal && s.family==kMain ? ValidGroups(in) : 0),
          "Wrong grouped ROC observation shape");
}
class Writer {
  std::ofstream stream_;unsigned kind_,index_{0};std::vector<Shape> shapes_{Shapes()};
 public:
  Writer(std::string const& path,unsigned kind):kind_{kind} {
    Require(kind<=kCuda && !std::filesystem::exists(path),"Unknown backend or existing grouped ROC report");
    stream_.exceptions(std::ios::badbit|std::ios::failbit);stream_.open(path,std::ios::binary);
    stream_.write(kMagic,8);Write(stream_,kind);Write(stream_,kCases);
  }
  void Add(Input const& in,Observation const& o={}) {
    Require(index_<kCases,"Too many grouped ROC cases");Validate(in,shapes_[index_]);CheckObservation(in,o,kind_);
    for (auto word:Identity(in.spec)) Write(stream_,word);
    Values(stream_,in.ptr);Values(stream_,in.predictions);Values(stream_,in.labels);Values(stream_,in.weights);
    if (kind_) {
      Write(stream_,o.guarded);Write(stream_,o.cpu_overflow_rejected);
      Values(stream_,o.original);Values(stream_,o.per_query_guards);Values(stream_,o.common);
      Values(stream_,o.isolated);Values(stream_,o.cpu);
    }
    ++index_;stream_.flush();
  }
  void Finish() {
    Require(index_==kCases,"Incomplete grouped ROC corpus");stream_.write("GROE",4);Write(stream_,index_);stream_.flush();
  }
};
class InputReader {
  std::ifstream stream_;unsigned index_{0};std::vector<Shape> shapes_{Shapes()};
 public:
  explicit InputReader(std::string const& path) {
    stream_.exceptions(std::ios::badbit|std::ios::failbit);stream_.open(path,std::ios::binary);
    char magic[8];stream_.read(magic,8);
    Require(!std::memcmp(magic,kMagic,8) && Read<unsigned>(stream_)==kInput &&
            Read<unsigned>(stream_)==kCases,"Requires complete input-only grouped ROC corpus");
  }
  Input Next() {
    Require(index_<kCases,"Too many grouped ROC inputs");auto shape=shapes_[index_];Input in{shape.spec,{},{},{},{}};
    for (auto word:Identity(in.spec)) Require(Read<unsigned>(stream_)==word,"Wrong grouped ROC identity/order");
    in.ptr=Values<std::uint32_t>(stream_,in.spec.groups+1);
    in.predictions=Values<float>(stream_,Storage(in.spec));in.labels=Values<float>(stream_,Storage(in.spec));
    in.weights=Values<float>(stream_,in.spec.weights);Validate(in,shape);++index_;return in;
  }
  void Finish() {
    Require(index_==kCases,"Unconsumed grouped ROC inputs");char end[4];stream_.read(end,4);
    Require(!std::memcmp(end,"GROE",4) && Read<unsigned>(stream_)==kCases &&
            stream_.peek()==std::char_traits<char>::eof(),"Incomplete/trailing grouped ROC report");
  }
};
class Exporter {
  unsigned index_{0};std::vector<Shape> shapes_{Shapes()};Writer inputs_,metal_;
 public:
  explicit Exporter(std::string const& prefix):inputs_{prefix+".inputs.bin",kInput},metal_{prefix+".metal.bin",kMetal} {}
  Spec NextSpec() const { Require(index_<kCases,"Too many grouped ROC exports");return shapes_[index_].spec; }
  void Add(Input const& in,Observation const& out) { inputs_.Add(in);metal_.Add(in,out);++index_; }
  void Finish() { inputs_.Finish();metal_.Finish(); }
};
inline std::unique_ptr<Exporter> ReportOption(int argc,char** argv) {
  Require(argc==1 || (argc==3 && std::string(argv[1])=="--report-prefix"),
          "Usage: metal_ranking_roc_api_test [--report-prefix NEW_PREFIX]");
  return argc==1 ? nullptr : std::make_unique<Exporter>(argv[2]);
}
}  // namespace grouped_roc
