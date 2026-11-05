// Complete existing root/stump corpus. CUDA inputs contain no expected outputs.
#pragma once
#include <memory>
#include "ranking_rounding_io.h"

namespace stump_contracts {
using ranking_rounding::Bits;
using ranking_rounding::Read;
using ranking_rounding::Require;
using ranking_rounding::Values;
using ranking_rounding::Write;
constexpr unsigned kCases=405, kInput=0, kMetal=1, kCuda=2;
inline constexpr char kMagic[8]={'X','G','B','S','T','P','1','\0'};
struct Spec { std::uint32_t id, rows, targets, layout, mode, stride0, stride1, pairs; };
inline std::array<std::uint32_t,8> Identity(Spec s) {
  return {s.id,s.rows,s.targets,s.layout,s.mode,s.stride0,s.stride1,s.pairs};
}
inline std::vector<Spec> Specs() {
  std::vector<Spec> specs;
  for (unsigned rows : {1u,31u,32u,255u,256u,257u,2048u,4097u,65537u})
    for (unsigned targets : {1u,3u,7u}) for (unsigned layout=0; layout<3; ++layout)
      for (unsigned mode=0; mode<5; ++mode) {
        unsigned a=layout==1 ? 1 : targets*(layout==2 ? 2 : 1);
        unsigned b=layout==1 ? rows+3 : (layout==2 ? 2 : 1);
        specs.push_back({unsigned(specs.size()),rows,targets,layout,mode,a,b,
                         (rows-1)*a+(targets-1)*b+4});
      }
  Require(specs.size()==kCases,"Incomplete stump matrix"); return specs;
}
struct Input { Spec spec; std::vector<float> storage; };
struct Observation {
  std::vector<double> sums;  // Interleaved gradient/Hessian, from a separate source probe on CUDA.
  std::vector<float> weights, api_weights;
  std::vector<double> references;  // Metal only: sum/magnitude for each gradient/Hessian channel.
};
inline void Validate(Input const& in, Spec expected) {
  Require(Identity(in.spec)==Identity(expected) && in.storage.size()==2*expected.pairs,
          "Wrong stump identity/order/shape");
}
class Writer {
  std::ofstream out_;
  unsigned kind_, index_{0};
  std::vector<Spec> specs_{Specs()};
 public:
  Writer(std::string const& path,unsigned kind):kind_{kind} {
    Require(kind<=kCuda && !std::filesystem::exists(path),"Unknown stump backend or existing output");
    out_.exceptions(std::ios::badbit|std::ios::failbit); out_.open(path,std::ios::binary);
    out_.write(kMagic,8); Write(out_,std::uint32_t{kind}); Write(out_,std::uint32_t{kCases});
  }
  void Add(Input const& in,Observation const& observed={}) {
    Require(index_<kCases,"Too many stump cases"); Validate(in,specs_[index_]);
    auto n=in.spec.targets;
    Require(observed.sums.size()==(kind_ ? 2*n : 0) && observed.weights.size()==(kind_ ? n : 0) &&
            observed.api_weights.size()==(kind_==kCuda ? n : 0) &&
            observed.references.size()==(kind_==kMetal ? 4*n : 0),"Wrong stump observation shape/backend");
    for (auto word:Identity(in.spec)) Write(out_,word);
    Values(out_,in.storage); Values(out_,observed.sums); Values(out_,observed.weights);
    Values(out_,observed.api_weights); Values(out_,observed.references);
    ++index_; out_.flush();
  }
  void Finish() {
    Require(index_==kCases,"Incomplete stump corpus"); out_.write("STPE",4); Write(out_,index_); out_.flush();
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
    Require(!std::memcmp(magic,kMagic,8) && Read<std::uint32_t>(in_)==kInput &&
            Read<std::uint32_t>(in_)==kCases,"CUDA requires complete input-only stump records");
  }
  Input Next() {
    Require(index_<kCases,"Too many stump inputs"); Input in{specs_[index_],{}};
    for (auto word:Identity(in.spec)) Require(Read<std::uint32_t>(in_)==word,"Wrong stump identity/order/shape");
    in.storage=Values<float>(in_,2*in.spec.pairs); ++index_; return in;
  }
  void Finish() {
    Require(index_==kCases,"Unconsumed stump inputs"); char footer[4]; in_.read(footer,4);
    Require(!std::memcmp(footer,"STPE",4) && Read<std::uint32_t>(in_)==kCases &&
            in_.peek()==std::char_traits<char>::eof(),"Missing stump completion marker or trailing bytes");
  }
};
class Exporter {
  unsigned index_{0}; std::vector<Spec> specs_{Specs()}; Writer inputs_,metal_;
 public:
  explicit Exporter(std::string const& prefix):inputs_{prefix+".inputs.bin",kInput},metal_{prefix+".metal.bin",kMetal} {}
  void Add(unsigned rows,unsigned targets,unsigned layout,unsigned mode,std::vector<float> storage,
           Observation const& observed) {
    Require(index_<kCases,"Too many native stump exports"); auto s=specs_[index_];
    Require(s.rows==rows && s.targets==targets && s.layout==layout && s.mode==mode,"Native stump case differs");
    Input in{s,std::move(storage)}; inputs_.Add(in); metal_.Add(in,observed); ++index_;
  }
  void Finish() { inputs_.Finish(); metal_.Finish(); }
};
inline std::unique_ptr<Exporter> ReportOption(int argc,char** argv) {
  Require(argc==2 || (argc==4 && std::string(argv[2])=="--report-prefix"),
          "Usage: metal_reduction shader [--report-prefix new-prefix]");
  if (argc==2) return {};
  std::string p=argv[3];
  Require(!std::filesystem::exists(p+".inputs.bin") && !std::filesystem::exists(p+".metal.bin"),"Use a new stump report prefix");
  return std::make_unique<Exporter>(p);
}
}  // namespace stump_contracts
