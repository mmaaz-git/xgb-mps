// Complete metal_linalg_api corpus, including two uncounted empty operations.
// Reference values belong only to Metal observations, never CUDA inputs.
#pragma once
#include <cmath>
#include <memory>
#include "ranking_rounding_io.h"

namespace linalg_contracts {
using ranking_rounding::Bits;
using ranking_rounding::Read;
using ranking_rounding::Require;
using ranking_rounding::Values;
using ranking_rounding::Write;
constexpr unsigned kCases = 260, kInput = 0, kMetal = 1, kCuda = 2;
inline constexpr char kMagic[8] = {'X','G','B','L','I','N','1','\0'};
struct Spec {
  std::uint32_t id, kind, rows, columns, mode, op, stride0, stride1, out_stride,
                values, weights, initial, outputs;
  double parameter;
};
inline std::array<std::uint32_t, 13> Identity(Spec s) {
  return {s.id,s.kind,s.rows,s.columns,s.mode,s.op,s.stride0,s.stride1,s.out_stride,
          s.values,s.weights,s.initial,s.outputs};
}
inline std::vector<Spec> Specs() {
  std::vector<Spec> r;
  auto hist = [&](unsigned n, unsigned bins, unsigned mode) {
    unsigned stride = 1+mode%3;
    r.push_back({unsigned(r.size()),0,n,bins,mode,0,stride,1,stride,n*stride+4,n,
                 bins*stride+4,bins*stride+4,1.});
  };
  for (unsigned n : {0u,1u,257u,4097u,65537u})
    for (unsigned bins : {1u,3u,33u,257u}) for (unsigned mode=0; mode<7; ++mode) hist(n,bins,mode);
  hist(262145,1025,3);
  for (unsigned stride : {1u,3u}) for (unsigned op=0; op<4; ++op)
    for (double p : {0.,-0.,1.,-1.,1./3,1e-40,1e-300,1e300,double(INFINITY),double(NAN)})
      r.push_back({unsigned(r.size()),1,65548,1,0,op,stride,1,stride,65548*stride+3,0,0,65548*stride+3,p});
  for (unsigned n : {1u,32u,257u,4097u,65537u,262145u}) for (unsigned mode=0; mode<5; ++mode)
    r.push_back({unsigned(r.size()),2,n,1,mode,0,3,1,1,n*3+4,0,0,1,0.});
  for (unsigned targets : {3u,7u})
    r.push_back({unsigned(r.size()),2,257,targets,5,0,2*targets+1,2,3,
                 257*(2*targets+1)+4,0,targets*3+4,targets*3+4,0.});
  r.push_back({unsigned(r.size()),2,256,1,6,0,1,1,1,256,0,0,1,0.});
  for (unsigned n : {4u,2u,1u})
    r.push_back({unsigned(r.size()),3,n,3,0,0,1,1,1,n,0,3,3,1.});
  r.push_back({unsigned(r.size()),4,5,4,0,0,1,1,1,5,0,4,4,.5});
  r.push_back({unsigned(r.size()),5,0,1,0,0,1,1,1,0,0,0,0,2.});
  r.push_back({unsigned(r.size()),6,0,0,0,0,1,1,1,0,0,0,0,1.});
  Require(r.size()==kCases,"Incomplete linalg matrix"); return r;
}
struct Input { Spec spec; std::vector<float> values, weights, initial; };
struct Reduction { double sum, magnitude; std::uint64_t count; float rounded; };
inline Reduction Reference(long double sum, long double magnitude, std::size_t count) {
  return {double(sum),double(magnitude),count,float(sum)};
}
struct Observation {
  // 0: executed, 1: Metal rejection, 2: outside CUDA domain, NOT executed.
  unsigned status{0};
  std::vector<float> actual, reference;
  std::vector<Reduction> reductions;
};
inline unsigned Reductions(Spec s) { return s.kind==0 || s.kind==2 || s.kind==4 ? s.columns : 0; }
inline void Validate(Input const& in, Spec expected) {
  Require(Identity(in.spec)==Identity(expected) &&
          Bits<std::uint64_t>(in.spec.parameter)==Bits<std::uint64_t>(expected.parameter) &&
          in.values.size()==expected.values && in.weights.size()==expected.weights &&
          in.initial.size()==expected.initial,"Wrong linalg identity/order/input shape");
}
class Writer {
  std::ofstream out_;
  unsigned kind_, index_{0};
  std::vector<Spec> specs_{Specs()};
 public:
  Writer(std::string const& path, unsigned kind) : kind_{kind} {
    Require(kind<=kCuda && !std::filesystem::exists(path),"Unknown backend or existing linalg report");
    out_.exceptions(std::ios::badbit|std::ios::failbit); out_.open(path,std::ios::binary);
    out_.write(kMagic,8); Write(out_,kind); Write(out_,kCases);
  }
  void Add(Input const& in, Observation const& o={}) {
    Require(index_<kCases,"Too many linalg cases"); Validate(in,specs_[index_]); auto s=in.spec;
    unsigned status=s.kind==3 ? (kind_==kCuda ? 2 : 1) : 0;
    Require(kind_ ? o.status==status && o.actual.size()==(status==2 ? 0 : s.outputs)
                  : o.status==0 && o.actual.empty(),"Wrong linalg observation/status");
    Require(o.reference.size()==(kind_==kMetal && s.kind==1 ? s.outputs : 0) &&
            o.reductions.size()==(kind_==kMetal ? Reductions(s) : 0),"Wrong linalg references");
    for (auto word : Identity(s)) Write(out_,word);
    Write(out_,Bits<std::uint64_t>(s.parameter));
    Values(out_,in.values); Values(out_,in.weights); Values(out_,in.initial);
    if (kind_) { Write(out_,o.status); Values(out_,o.actual); }
    if (kind_==kMetal) {
      Values(out_,o.reference);
      for (auto r : o.reductions) {
        Write(out_,Bits<std::uint64_t>(r.sum)); Write(out_,Bits<std::uint64_t>(r.magnitude));
        Write(out_,r.count); Write(out_,Bits<std::uint32_t>(r.rounded));
      }
    }
    ++index_; out_.flush();
  }
  void Finish() {
    Require(index_==kCases,"Incomplete linalg corpus");
    out_.write("LINE",4); Write(out_,index_); out_.flush();
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
    Require(!std::memcmp(magic,kMagic,8),"Unknown linalg interchange version");
    Require(Read<unsigned>(in_)==kInput && Read<unsigned>(in_)==kCases,
            "CUDA requires complete input-only linalg records");
  }
  Input Next() {
    Require(index_<kCases,"Too many linalg inputs"); Input in{specs_[index_],{},{},{}};
    for (auto word : Identity(in.spec)) Require(Read<std::uint32_t>(in_)==word,"Wrong linalg identity/order/shape");
    Require(Read<std::uint64_t>(in_)==Bits<std::uint64_t>(in.spec.parameter),"Wrong linalg parameter bits");
    in.values=Values<float>(in_,in.spec.values); in.weights=Values<float>(in_,in.spec.weights);
    in.initial=Values<float>(in_,in.spec.initial); ++index_; return in;
  }
  void Finish() {
    Require(index_==kCases,"Unconsumed linalg inputs"); char footer[4]; in_.read(footer,4);
    Require(!std::memcmp(footer,"LINE",4) && Read<unsigned>(in_)==kCases,"Missing linalg completion marker");
    Require(in_.peek()==std::char_traits<char>::eof(),"Trailing linalg bytes");
  }
};
class Exporter {
  unsigned index_{0};
  Writer inputs_, metal_;
 public:
  explicit Exporter(std::string const& prefix)
      : inputs_{prefix+".inputs.bin",kInput}, metal_{prefix+".metal.bin",kMetal} {}
  void Add(unsigned kind,unsigned rows,unsigned columns,unsigned mode,unsigned op,
           unsigned stride0,unsigned stride1,unsigned out_stride,double parameter,
           std::vector<float> const& values,std::vector<float> const& weights,
           std::vector<float> const& initial,Observation const& observed) {
    Input in{{index_,kind,rows,columns,mode,op,stride0,stride1,out_stride,unsigned(values.size()),
              unsigned(weights.size()),unsigned(initial.size()),unsigned(observed.actual.size()),parameter},
             values,weights,initial};
    inputs_.Add(in); metal_.Add(in,observed); ++index_;
  }
  void Finish() { inputs_.Finish(); metal_.Finish(); }
};
inline std::unique_ptr<Exporter> ReportOption(int argc,char** argv) {
  Require(argc==1 || (argc==3 && std::string(argv[1])=="--report-prefix"),
          "Usage: native-test [--report-prefix new-prefix]");
  if (argc==1) return {};
  std::string prefix=argv[2];
  Require(!std::filesystem::exists(prefix+".inputs.bin") && !std::filesystem::exists(prefix+".metal.bin"),
          "Use a new linalg report prefix");
  return std::make_unique<Exporter>(prefix);
}
}  // namespace linalg_contracts
