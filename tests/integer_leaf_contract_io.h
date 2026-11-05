// Complete integer leaf corpus; actual converted sums and integer diagnostics
// are distinct. Metal does not expose its private integer buffer through this API.
#pragma once
#include <memory>
#include "ranking_rounding_io.h"

namespace integer_leaf_contracts {
using ranking_rounding::Bits;
using ranking_rounding::Read;
using ranking_rounding::Require;
using ranking_rounding::Values;
using ranking_rounding::Write;
constexpr unsigned kCases=256,kInput=0,kMetal=1,kCuda=2;
inline constexpr char kMagic[8]={'X','G','B','I','L','F','1','\0'};
struct Spec {std::uint32_t id,family,rows,targets,layout,mode,extra,domain;};
inline std::array<std::uint32_t,8> Identity(Spec s) {
  return {s.id,s.family,s.rows,s.targets,s.layout,s.mode,s.extra,s.domain};
}
inline std::vector<Spec> Specs() {
  std::vector<Spec> result;
  auto add=[&](unsigned rows,unsigned targets,unsigned layout,unsigned mode) {
    result.push_back({unsigned(result.size()),0,rows,targets,layout,mode,0,0});
  };
  for(unsigned n:{0u,1u,31u,257u,4097u,65539u})for(unsigned t:{1u,3u})
    for(unsigned layout=0;layout<5;++layout)for(unsigned mode=0;mode<4;++mode)add(n,t,layout,mode);
  add(1000003,3,0,0);add(1000003,3,1,0);
  for(int exponent:{-80,0,54,100})for(unsigned paged:{0u,1u})
    result.push_back({unsigned(result.size()),1,4,1,0,paged,std::uint32_t(exponent),0});
  result.push_back({250,2,4,1,0,0,0,0});
  for(std::uint32_t value:{0xc0800000u,0x407fffffu,0x40800000u,0x7f800000u,0x7fc00000u})
    result.push_back({unsigned(result.size()),3,1,1,0,0,value,unsigned(result.size()>=253)});
  Require(result.size()==kCases,"Incomplete integer leaf case matrix");return result;
}
struct Shape {unsigned rows,stride0,stride1,offset,extent,storage,leaves,snapshots,weight_words;};
inline Shape Layout(Spec s) {
  if(s.family) return {s.rows,1,1,0,s.rows,s.rows,s.family==1?2u:1u,
                       s.domain?0u:s.family==1 && s.mode?2u:1u,0};
  auto n=s.rows+11;
  unsigned a=s.layout==1?1:s.layout==2?s.targets*2+3:s.targets;
  unsigned b=s.layout==1?n+3:s.layout==2?2:1;
  if(s.layout==3)a=0;
  if(s.layout==4){a=1;b=0;}
  auto extent=(n-1)*a+(s.targets-1)*b+1;
  return {n,a,b,4,extent,extent+8,7,3,5*(7*s.targets+2)};
}
inline unsigned SumWords(Spec s) {auto l=Layout(s);return 2*l.leaves*s.targets*l.snapshots;}
struct Input {Spec spec;std::vector<float> original,sampled;};
struct Observation {
  unsigned status{0};
  std::vector<std::uint64_t> sums,integers,reference_sums;
  std::vector<std::uint32_t> weights,reference_weights;
};
inline void Validate(Input const& in,Spec s) {
  auto shape=Layout(s);
  Require(Identity(in.spec)==Identity(s) && in.original.size()==2*shape.storage &&
          in.sampled.size()==2*shape.storage,"Wrong integer leaf identity/order/backing shape");
}
class Writer {
  std::ofstream out_;unsigned kind_,index_{0};std::vector<Spec> specs_{Specs()};
 public:
  Writer(std::string const& path,unsigned kind):kind_{kind} {
    Require(kind<=kCuda && !std::filesystem::exists(path),"Unknown integer leaf backend or existing output");
    out_.exceptions(std::ios::badbit|std::ios::failbit);out_.open(path,std::ios::binary);
    out_.write(kMagic,8);Write(out_,std::uint32_t{kind});Write(out_,std::uint32_t{kCases});
  }
  void Add(Input const& in,Observation const& observed={}) {
    Require(index_<kCases,"Too many integer leaf records");Validate(in,specs_[index_]);
    auto n=kind_?SumWords(in.spec):0,w=kind_?Layout(in.spec).weight_words:0;
    Require(observed.status==(kind_ && in.spec.domain?(kind_==kMetal?1u:2u):0u),"Wrong integer leaf domain status");
    Require(observed.sums.size()==n && observed.integers.size()==n && observed.weights.size()==w &&
            observed.reference_sums.size()==(kind_==kMetal?n:0) && observed.reference_weights.size()==(kind_==kMetal?w:0),
            "Wrong integer leaf observation shape/backend");
    for(auto word:Identity(in.spec))Write(out_,word);
    Values(out_,in.original);Values(out_,in.sampled);
    if(kind_)Write(out_,std::uint32_t{observed.status});
    for(auto word:observed.sums)Write(out_,word);
    for(auto word:observed.weights)Write(out_,word);
    // Metal: independent integer references. CUDA: actual integer observations.
    for(auto word:observed.integers)Write(out_,word);
    for(auto word:observed.reference_sums)Write(out_,word);
    for(auto word:observed.reference_weights)Write(out_,word);
    ++index_;out_.flush();
  }
  void Finish(){Require(index_==kCases,"Incomplete integer leaf corpus");out_.write("ILFE",4);Write(out_,index_);out_.flush();}
};
class InputReader {
  std::ifstream in_;unsigned index_{0};std::vector<Spec> specs_{Specs()};
 public:
  explicit InputReader(std::string const& path) {
    in_.exceptions(std::ios::badbit|std::ios::failbit);in_.open(path,std::ios::binary);
    char magic[8];in_.read(magic,8);
    Require(!std::memcmp(magic,kMagic,8) && Read<std::uint32_t>(in_)==kInput && Read<std::uint32_t>(in_)==kCases,
            "CUDA requires complete input-only integer leaf records");
  }
  Input Next() {
    Require(index_<kCases,"Too many integer leaf inputs");Input in{specs_[index_],{}, {}};
    for(auto word:Identity(in.spec))Require(Read<std::uint32_t>(in_)==word,"Wrong integer leaf case/order/parameters");
    auto n=2*Layout(in.spec).storage;in.original=Values<float>(in_,n);in.sampled=Values<float>(in_,n);++index_;return in;
  }
  void Finish() {
    Require(index_==kCases,"Unconsumed integer leaf inputs");char footer[4];in_.read(footer,4);
    Require(!std::memcmp(footer,"ILFE",4) && Read<std::uint32_t>(in_)==kCases &&
            in_.peek()==std::char_traits<char>::eof(),"Missing integer leaf completion marker or trailing bytes");
  }
};
class Exporter {
  Writer inputs_,metal_;
 public:
  explicit Exporter(std::string const& prefix):inputs_{prefix+".inputs.bin",kInput},metal_{prefix+".metal.bin",kMetal} {}
  void Add(Input const& in,Observation const& out){inputs_.Add(in);metal_.Add(in,out);}
  void Finish(){inputs_.Finish();metal_.Finish();}
};
inline std::unique_ptr<Exporter> ReportOption(int argc,char** argv) {
  Require(argc==1 || (argc==3 && std::string(argv[1])=="--report-prefix"),"Usage: metal_integer_leaf_api [--report-prefix new-prefix]");
  if(argc==1)return {};
  std::string prefix=argv[2];
  Require(!std::filesystem::exists(prefix+".inputs.bin") && !std::filesystem::exists(prefix+".metal.bin"),"Use a new integer leaf report prefix");
  return std::make_unique<Exporter>(prefix);
}
}  // namespace integer_leaf_contracts
