// Exact interchange for the complete metal_ranking_pr_api corpus.
// Original Metal calls retain empty queries and strided storage. Common replay
// removes empty queries before calling CUDA; isolated-query results are separate
// observations, not fabricated per-query outputs from the aggregate CUDA call.
#pragma once
#include <algorithm>
#include <memory>
#include "ranking_rounding_io.h"

namespace grouped_pr {
using ranking_rounding::Bits;
using ranking_rounding::Read;
using ranking_rounding::Require;
using ranking_rounding::Values;
using ranking_rounding::Write;
constexpr unsigned kCases=319, kInput=0, kMetal=1, kCuda=2;
inline constexpr char kMagic[8]={'X','G','B','G','P','R','1','\0'};
struct Spec { std::uint32_t id,family,layout,mode,rows,groups,weights; };
inline auto Identity(Spec s) {
  return std::array<std::uint32_t,7>{s.id,s.family,s.layout,s.mode,s.rows,s.groups,s.weights};
}
struct Shape { Spec spec; std::vector<std::uint32_t> ptr; };
inline std::vector<Shape> Shapes() {
  std::vector<Shape> out;
  auto add=[&](unsigned family,unsigned layout,unsigned mode,std::vector<unsigned> lengths,bool weighted) {
    std::vector<std::uint32_t> ptr{0};
    for (auto n:lengths) ptr.push_back(ptr.back()+n);
    out.push_back({{unsigned(out.size()),family,layout,mode,ptr.back(),unsigned(lengths.size()),
                   weighted ? unsigned(lengths.size()) : 0},ptr});
  };
  add(0,0,0,{1},false);
  for (unsigned mode=1;mode<=15;++mode) add(0,0,mode,{2},true);
  for (unsigned n:{0u,1u,31u,255u,256u,257u,4097u,65537u})
    for (unsigned layout=0;layout<3;++layout) for (unsigned mode=0;mode<12;++mode)
      add(1,layout,mode,{0,1,2,3,31,32,33,255,256,257,513,0,n,n+1,n/2,0},mode!=0);
  for (unsigned mode:{0u,2u,3u,7u,8u,10u,11u})
    add(1,2,mode,{1,31,1000003,0,65537,257,1},mode!=0);
  add(1,0,0,{},false); add(1,0,0,{0,0,0},false);
  for (unsigned mode=0;mode<6;++mode) add(2,0,mode,{1,0,2},false);
  Require(out.size()==kCases,"Incomplete grouped PR shape matrix"); return out;
}
inline unsigned Stride(Spec s) { return s.family==1 ? (s.layout==0 ? 1 : s.layout==1 ? 3 : 7) : 1; }
inline unsigned Offset(Spec s) { return s.family==1 ? 4 : 0; }
inline std::size_t Storage(Spec s) { return std::size_t{s.rows}*Stride(s)+(s.family==1 ? 8 : 0); }
struct Input { Spec spec; std::vector<std::uint32_t> ptr; std::vector<float> predictions,labels,weights; };
inline void Validate(Input const& in,Shape const& shape) {
  Require(Identity(in.spec)==Identity(shape.spec) && in.ptr==shape.ptr &&
          in.predictions.size()==Storage(in.spec) && in.labels.size()==Storage(in.spec) &&
          in.weights.size()==in.spec.weights,"Wrong grouped PR identity/order/input shape");
}
inline std::vector<unsigned> Nonempty(Input const& in) {
  std::vector<unsigned> ids;
  for (unsigned g=0;g<in.spec.groups;++g) if (in.ptr[g+1]>in.ptr[g]) ids.push_back(g);
  return ids;
}
// Deliberate API adaptation: CUDA takes contiguous prediction spans. Zero-row
// query removal avoids LastOf(empty_group) and packed reduce_by_key totals.
inline Input Compact(Input const& in) {
  Input c{in.spec,{0},{},{},{}};
  for (unsigned row=0;row<in.spec.rows;++row) {
    c.predictions.push_back(in.predictions[Offset(in.spec)+row*Stride(in.spec)]);
    c.labels.push_back(in.labels[Offset(in.spec)+row*Stride(in.spec)]);
  }
  for (auto g:Nonempty(in)) {
    c.ptr.push_back(c.ptr.back()+in.ptr[g+1]-in.ptr[g]);
    if (!in.weights.empty()) c.weights.push_back(in.weights[g]);
  }
  return c;
}
struct Observation {
  unsigned rejected{0};
  std::vector<double> original,guards,common,isolated;
};
inline void CheckObservation(Input const& in,Observation const& o,unsigned kind) {
  auto s=in.spec; bool rejected=s.family==2 && kind!=kInput;
  Require(o.rejected==unsigned(rejected),"Wrong grouped PR rejection status");
  auto normal=kind!=kInput && !rejected;
  Require(o.original.size()==(normal && kind==kMetal ? 2u : 0u) &&
          o.guards.size()==(normal && kind==kMetal && s.family==1 ? 2*(s.groups+4) : 0u) &&
          o.common.size()==(normal ? 2u : 0u) &&
          o.isolated.size()==(normal ? 2*Nonempty(in).size() : 0u),
          "Wrong grouped PR observation shape");
}
class Writer {
  std::ofstream stream_;
  unsigned kind_,index_{0};
  std::vector<Shape> shapes_{Shapes()};
 public:
  Writer(std::string const& path,unsigned kind):kind_{kind} {
    Require(kind<=kCuda && !std::filesystem::exists(path),"Unknown backend or existing grouped PR report");
    stream_.exceptions(std::ios::badbit|std::ios::failbit);stream_.open(path,std::ios::binary);
    stream_.write(kMagic,8);Write(stream_,kind);Write(stream_,kCases);
  }
  void Add(Input const& in,Observation const& o={}) {
    Require(index_<kCases,"Too many grouped PR cases");Validate(in,shapes_[index_]);CheckObservation(in,o,kind_);
    for (auto word:Identity(in.spec)) Write(stream_,word);
    Values(stream_,in.ptr);Values(stream_,in.predictions);Values(stream_,in.labels);Values(stream_,in.weights);
    if (kind_) {
      Write(stream_,o.rejected);Values(stream_,o.original);Values(stream_,o.guards);
      Values(stream_,o.common);Values(stream_,o.isolated);
    }
    ++index_;stream_.flush();
  }
  void Finish() {
    Require(index_==kCases,"Incomplete grouped PR corpus");stream_.write("GPRE",4);Write(stream_,index_);stream_.flush();
  }
};
class InputReader {
  std::ifstream stream_;
  unsigned index_{0};
  std::vector<Shape> shapes_{Shapes()};
 public:
  explicit InputReader(std::string const& path) {
    stream_.exceptions(std::ios::badbit|std::ios::failbit);stream_.open(path,std::ios::binary);
    char magic[8];stream_.read(magic,8);
    Require(!std::memcmp(magic,kMagic,8) && Read<unsigned>(stream_)==kInput &&
            Read<unsigned>(stream_)==kCases,"Requires complete input-only grouped PR corpus");
  }
  Input Next() {
    Require(index_<kCases,"Too many grouped PR inputs");auto shape=shapes_[index_];Input in{shape.spec,{},{},{},{}};
    for (auto word:Identity(in.spec)) Require(Read<unsigned>(stream_)==word,"Wrong grouped PR identity/order");
    in.ptr=Values<std::uint32_t>(stream_,in.spec.groups+1);
    in.predictions=Values<float>(stream_,Storage(in.spec));in.labels=Values<float>(stream_,Storage(in.spec));
    in.weights=Values<float>(stream_,in.spec.weights);Validate(in,shape);++index_;return in;
  }
  void Finish() {
    Require(index_==kCases,"Unconsumed grouped PR inputs");char end[4];stream_.read(end,4);
    Require(!std::memcmp(end,"GPRE",4) && Read<unsigned>(stream_)==kCases &&
            stream_.peek()==std::char_traits<char>::eof(),"Incomplete/trailing grouped PR report");
  }
};
class Exporter {
  unsigned index_{0};
  std::vector<Shape> shapes_{Shapes()};
  Writer inputs_,metal_;
 public:
  explicit Exporter(std::string const& prefix):inputs_{prefix+".inputs.bin",kInput},metal_{prefix+".metal.bin",kMetal} {}
  Spec NextSpec() const { Require(index_<kCases,"Too many grouped PR exports");return shapes_[index_].spec; }
  void Add(Input const& in,Observation const& out) { inputs_.Add(in);metal_.Add(in,out);++index_; }
  void Finish() { inputs_.Finish();metal_.Finish(); }
};
inline std::unique_ptr<Exporter> ReportOption(int argc,char** argv) {
  Require(argc==1 || (argc==3 && std::string(argv[1])=="--report-prefix"),"Usage: metal_ranking_pr_api_test [--report-prefix NEW_PREFIX]");
  return argc==1 ? nullptr : std::make_unique<Exporter>(argv[2]);
}
}  // namespace grouped_pr
