// Complete metal_ranking_api corpus; no output/reference values in CUDA inputs.
#pragma once
#include <algorithm>
#include <memory>
#include "ranking_rounding_io.h"
namespace ranking_metric {
using ranking_rounding::Bits;
using ranking_rounding::Read;
using ranking_rounding::Require;
using ranking_rounding::Values;
using ranking_rounding::Write;
constexpr unsigned kCases=1031,kInput=0,kMetal=1,kCuda=2,kEdge=0,kSort=1,kScore=2;
inline constexpr char kMagic[8]={'X','G','B','R','M','T','1','\0'};
// kind matches CheckScores: precision, exponential NDCG, linear NDCG, MAP; 4=sort.
struct Spec { unsigned id,family,layout,kind,stride,topk,mode,minus,rows,groups,weights,discounts,inverse; };
inline auto Identity(Spec s) { return std::array<unsigned,13>{s.id,s.family,s.layout,s.kind,s.stride,s.topk,s.mode,s.minus,s.rows,s.groups,s.weights,s.discounts,s.inverse}; }
inline bool NDCG(Spec s) { return s.kind==1 || s.kind==2; }
inline bool NoGroups(Spec s) { return NDCG(s) && !s.groups; }
inline unsigned Guards(Spec s) { return (s.family!=kSort && s.topk==0 ? 1u : 0u) | (NoGroups(s) ? 2u : 0u); }
struct Shape { Spec spec;std::vector<unsigned> ptr; };
inline std::vector<Shape> Shapes() {
  std::vector<Shape> shapes;
  auto add=[&](unsigned family,unsigned layout,unsigned kind,unsigned stride,unsigned topk,unsigned mode,
               bool minus,std::vector<unsigned> lengths,bool weighted,unsigned discounts,unsigned inverse) {
    std::vector<unsigned> ptr{0};for (auto n:lengths) ptr.push_back(ptr.back()+n);
    shapes.push_back({{unsigned(shapes.size()),family,layout,kind,stride,topk,mode,unsigned(minus),ptr.back(),
                       unsigned(lengths.size()),weighted ? unsigned(lengths.size()) : 0u,discounts,inverse},ptr});
  };
  add(kEdge,0,0,1,10,0,false,{0,0},true,0,0);
  add(kEdge,0,0,1,0,0,false,{1},true,0,0);
  add(kEdge,0,0,1,1,0,false,{1},true,0,0);
  add(kEdge,0,0,1,3,0,false,{3},true,0,0);
  for (bool minus:{false,true}) for (unsigned mode:{0u,1u})
    add(kEdge,0,3,1,10,mode,minus,{0,2,0},true,0,0);
  add(kEdge,0,3,1,4,0,false,{4},true,0,0);
  add(kEdge,0,3,1,0,0,false,{1},true,0,0);
  add(kEdge,0,3,1,0,1,false,{1},true,0,0);
  add(kEdge,0,2,1,2,0,false,{2},false,2,1);
  for (unsigned mode=0;mode<3;++mode) add(kEdge,0,2,1,1,mode,false,{1},true,1,1);
  std::vector<std::vector<unsigned>> sorts{{},{0,0,0},{1},{31,0,32,33},{255,256,0,1},
      {257,0,1,256},{0,65537,0},{65537,1,4097,0,257},
      {0,511,512,513,1023,1024,1025,2047,2048,2049,4095,4096,0},{4097,1}};
  std::vector<unsigned> many(4097),skew(65537,1);
  for (unsigned i=0;i<many.size();++i) many[i]=i%257;
  skew[32768]=4097;sorts.push_back(many);sorts.push_back(skew);
  for (unsigned l=0;l<sorts.size();++l) for (unsigned stride:{1u,3u}) for (unsigned mode=0;mode<5;++mode)
    add(kSort,l,4,stride,0,mode,false,sorts[l],false,0,0);
  std::vector<std::vector<unsigned>> scores{{},{0,0},{1},{0,31,0,257,1},{4096,32,0},{0,4097,1},{65537,1,256}};
  for (unsigned l=0;l<scores.size();++l) {
    auto const& sizes=scores[l];unsigned maximum=sizes.empty() ? 0 : *std::max_element(sizes.begin(),sizes.end());
    for (unsigned kind=0;kind<4;++kind) for (unsigned stride:{1u,3u})
      for (unsigned k:{0u,1u,10u,~0u}) for (unsigned mode=0;mode<4;++mode)
        add(kScore,l,kind,stride,k,mode,mode&1,sizes,mode!=0,maximum,sizes.size());
  }
  Require(shapes.size()==kCases,"Incomplete ranking metric matrix");return shapes;
}
struct Input { Spec spec;std::vector<unsigned> ptr;std::vector<float> predictions,labels,weights;std::vector<double> discounts,inverse; };
inline std::size_t Storage(Spec s) { return std::size_t{s.rows}*s.stride; }
inline void Validate(Input const& in,Shape const& shape) {
  auto s=in.spec;
  Require(Identity(s)==Identity(shape.spec) && in.ptr==shape.ptr && in.predictions.size()==Storage(s) &&
          in.labels.size()==(s.family==kSort ? 0 : Storage(s)) && in.weights.size()==s.weights &&
          in.discounts.size()==s.discounts && in.inverse.size()==s.inverse,"Wrong ranking metric identity/input shape");
}
struct Observation {
  unsigned guards{0};
  // original: unchanged Metal call; common: same call except custom-discount edge 11.
  // CUDA order/reducer/prefix are read from the actual producer's owned buffers.
  std::vector<double> original,common,reducer,prefix;
  std::vector<std::uint64_t> order;
};
inline void CheckObservation(Spec s,Observation const& o,unsigned kind) {
  bool metal=kind==kMetal,cuda=kind==kCuda,score=s.family!=kSort,live=cuda && !NoGroups(s);
  Require(o.guards==(cuda ? Guards(s) : 0) &&
          o.original.size()==(metal && score ? 2u : 0u) &&
          o.common.size()==(score && (metal || live) ? 2u : 0u) &&
          o.order.size()==((metal && !score) || live ? s.rows : 0u) &&
          o.reducer.size()==(live && score ? s.groups : 0u) &&
          o.prefix.size()==(live && s.kind==3 ? s.rows : 0u),"Wrong ranking metric observation shape/guards");
}
class Writer {
  std::ofstream stream_;unsigned kind_,index_{0};std::vector<Shape> shapes_{Shapes()};
 public:
  Writer(std::string const& path,unsigned kind):kind_{kind} {
    Require(kind<=kCuda && !std::filesystem::exists(path),"Unknown backend or existing ranking metric report");
    stream_.exceptions(std::ios::badbit|std::ios::failbit);stream_.open(path,std::ios::binary);
    stream_.write(kMagic,8);Write(stream_,kind);Write(stream_,kCases);
  }
  void Add(Input const& in,Observation const& o={}) {
    Require(index_<kCases,"Too many ranking metric cases");Validate(in,shapes_[index_]);CheckObservation(in.spec,o,kind_);
    for (auto word:Identity(in.spec)) Write(stream_,word);
    Values(stream_,in.ptr);Values(stream_,in.predictions);Values(stream_,in.labels);Values(stream_,in.weights);
    Values(stream_,in.discounts);Values(stream_,in.inverse);
    if (kind_) { Write(stream_,o.guards);Values(stream_,o.original);Values(stream_,o.common);
      Values(stream_,o.reducer);Values(stream_,o.prefix);Values(stream_,o.order); }
    ++index_;stream_.flush();
  }
  void Finish() { Require(index_==kCases,"Incomplete ranking metric corpus");stream_.write("RMTE",4);Write(stream_,index_);stream_.flush(); }
};
class InputReader {
  std::ifstream stream_;unsigned index_{0};std::vector<Shape> shapes_{Shapes()};
 public:
  explicit InputReader(std::string const& path) {
    stream_.exceptions(std::ios::badbit|std::ios::failbit);stream_.open(path,std::ios::binary);
    char magic[8];stream_.read(magic,8);
    Require(!std::memcmp(magic,kMagic,8) && Read<unsigned>(stream_)==kInput && Read<unsigned>(stream_)==kCases,
            "Requires complete input-only ranking metric corpus");
  }
  Input Next() {
    Require(index_<kCases,"Too many ranking metric inputs");auto shape=shapes_[index_];Input in;in.spec=shape.spec;
    for (auto word:Identity(in.spec)) Require(Read<unsigned>(stream_)==word,"Wrong ranking metric identity/order");
    auto s=in.spec;in.ptr=Values<unsigned>(stream_,s.groups+1);in.predictions=Values<float>(stream_,Storage(s));
    in.labels=Values<float>(stream_,s.family==kSort ? 0 : Storage(s));in.weights=Values<float>(stream_,s.weights);
    in.discounts=Values<double>(stream_,s.discounts);in.inverse=Values<double>(stream_,s.inverse);
    Validate(in,shape);++index_;return in;
  }
  void Finish() {
    Require(index_==kCases,"Unconsumed ranking metric inputs");char end[4];stream_.read(end,4);
    Require(!std::memcmp(end,"RMTE",4) && Read<unsigned>(stream_)==kCases && stream_.peek()==std::char_traits<char>::eof(),
            "Incomplete/trailing ranking metric report");
  }
};
class Exporter {
  unsigned index_{0};std::vector<Shape> shapes_{Shapes()};Writer inputs_,metal_;
 public:
  explicit Exporter(std::string const& prefix):inputs_{prefix+".inputs.bin",kInput},metal_{prefix+".metal.bin",kMetal} {}
  Spec NextSpec() const { Require(index_<kCases,"Too many ranking metric exports");return shapes_[index_].spec; }
  void Add(Input const& in,Observation const& out) { inputs_.Add(in);metal_.Add(in,out);++index_; }
  void Finish() { inputs_.Finish();metal_.Finish(); }
};
inline std::unique_ptr<Exporter> ReportOption(int argc,char** argv) {
  Require(argc==1 || (argc==3 && std::string(argv[1])=="--report-prefix"),"Usage: metal_ranking_api_test [--report-prefix NEW_PREFIX]");
  return argc==1 ? nullptr : std::make_unique<Exporter>(argv[2]);
}
}  // namespace ranking_metric
