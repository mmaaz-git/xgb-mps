// Full cache lifecycle corpus. Materialized Metal guards and integer CUDA guards
// have different representations and are validated separately.
#pragma once
#include <memory>
#include "ranking_rounding_io.h"
namespace integer_cache_contracts {
using ranking_rounding::Bits;using ranking_rounding::Read;using ranking_rounding::Write;
using ranking_rounding::Values;using ranking_rounding::Require;
constexpr unsigned kCases=48,kInput=0,kMetal=1,kCuda=2;
inline constexpr char kMagic[8]={'X','G','B','H','I','C','1','\0'};
struct Spec {std::uint32_t id,dense,max_bin,exponent,bins;};
inline std::array<std::uint32_t,5> Identity(Spec s){return {s.id,s.dense,s.max_bin,s.exponent,s.bins};}
inline std::vector<Spec> Specs() {
  std::vector<Spec> out;
  for(unsigned dense:{0u,1u})for(unsigned max_bin:{16u,512u,65537u})
    for(int exponent:{-80,0,54,100})for(unsigned bins:{1u,2u})
      out.push_back({unsigned(out.size()),dense,max_bin,std::uint32_t(exponent),bins});
  Require(out.size()==kCases,"Incomplete integer cache matrix");return out;
}
inline unsigned CommonWords(Spec s){return 18*s.bins+6;}
struct Input {Spec spec;std::vector<float> data,cuts,first,second;};
struct Observation {
  std::vector<std::uint32_t> bins;
  std::vector<std::uint64_t> values,prefix_before,prefix_after,guards,integers;
};
inline void Validate(Input const& in,Spec s) {
  Require(Identity(in.spec)==Identity(s) && in.data.size()==4 && in.cuts.size()==s.bins &&
          in.first.size()==8 && in.second.size()==8,"Wrong integer cache identity/input shape");
}
class Writer {
  std::ofstream out_;unsigned kind_,index_{0};std::vector<Spec> specs_{Specs()};
 public:
  Writer(std::string const& path,unsigned kind):kind_{kind} {
    Require(kind<=kCuda && !std::filesystem::exists(path),"Unknown cache backend or existing output");
    out_.exceptions(std::ios::badbit|std::ios::failbit);out_.open(path,std::ios::binary);
    out_.write(kMagic,8);Write(out_,std::uint32_t{kind});Write(out_,std::uint32_t{kCases});
  }
  void Add(Input const& in,Observation const& o={}) {
    Require(index_<kCases,"Too many cache records");Validate(in,specs_[index_]);
    auto width=kind_==kCuda?in.spec.bins:1;
    Require(o.bins.size()==(kind_?4u:0u) && o.values.size()==(kind_?CommonWords(in.spec):0u) &&
            o.prefix_before.size()==(kind_?32*width:0u) && o.prefix_after.size()==(kind_?32*width:0u) &&
            o.guards.size()==(kind_?20*width:0u) && o.integers.size()==(kind_==kCuda?CommonWords(in.spec):0u),
            "Wrong integer cache observation shape/backend");
    for(auto v:Identity(in.spec))Write(out_,v);
    Values(out_,in.data);Values(out_,in.cuts);Values(out_,in.first);Values(out_,in.second);
    for(auto v:o.bins)Write(out_,v);
    for(auto const* words:{&o.values,&o.prefix_before,&o.prefix_after,&o.guards,&o.integers})
      for(auto v:*words)Write(out_,v);
    ++index_;out_.flush();
  }
  void Finish(){Require(index_==kCases,"Incomplete cache corpus");out_.write("HICE",4);Write(out_,index_);out_.flush();}
};
class InputReader {
  std::ifstream in_;unsigned index_{0};std::vector<Spec> specs_{Specs()};
 public:
  explicit InputReader(std::string const& path) {
    in_.exceptions(std::ios::badbit|std::ios::failbit);in_.open(path,std::ios::binary);char magic[8];in_.read(magic,8);
    Require(!std::memcmp(magic,kMagic,8) && Read<std::uint32_t>(in_)==kInput && Read<std::uint32_t>(in_)==kCases,
            "CUDA requires input-only complete integer cache records");
  }
  Input Next() {
    Require(index_<kCases,"Too many cache inputs");Input in{specs_[index_],{},{},{},{}};
    for(auto v:Identity(in.spec))Require(Read<std::uint32_t>(in_)==v,"Wrong cache identity/order/parameters");
    in.data=Values<float>(in_,4);in.cuts=Values<float>(in_,in.spec.bins);
    in.first=Values<float>(in_,8);in.second=Values<float>(in_,8);++index_;return in;
  }
  void Finish() {
    Require(index_==kCases,"Unconsumed cache inputs");char footer[4];in_.read(footer,4);
    Require(!std::memcmp(footer,"HICE",4) && Read<std::uint32_t>(in_)==kCases && in_.peek()==std::char_traits<char>::eof(),
            "Missing cache completion marker or trailing bytes");
  }
};
class Exporter {
  Writer inputs_,metal_;
 public:
  explicit Exporter(std::string const& prefix):inputs_{prefix+".inputs.bin",kInput},metal_{prefix+".metal.bin",kMetal}{}
  void Add(Input const& in,Observation const& out){inputs_.Add(in);metal_.Add(in,out);}
  void Finish(){inputs_.Finish();metal_.Finish();}
};
inline std::unique_ptr<Exporter> ReportOption(int argc,char** argv) {
  Require(argc==1 || (argc==3 && std::string(argv[1])=="--report-prefix"),"Usage: metal_integer_cache_api [--report-prefix new-prefix]");
  if(argc==1)return {};
  std::string p=argv[2];Require(!std::filesystem::exists(p+".inputs.bin") && !std::filesystem::exists(p+".metal.bin"),"Use a new cache report prefix");
  return std::make_unique<Exporter>(p);
}
}  // namespace integer_cache_contracts
