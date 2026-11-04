// Independent Google Snappy CPU codec validates both directions. No CPU codec
// is linked into the production library.
#include <snappy.h>

#include <algorithm>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "../xgboost/src/common/metal_histogram.h"
#include "../xgboost/src/common/metal_snappy.h"

namespace metal = xgboost::common::metal;
using Bytes = std::vector<std::uint8_t>;
std::size_t cases = 0, fields = 0, invalid = 0, checked_bytes = 0;
void Check(bool value, char const *message) {
  ++fields;
  if (!value)
    throw std::runtime_error(message);
}
void Equal(Bytes const &actual, Bytes const &expected) {
  Check(actual == expected, "Decoded bytes differ");
  checked_bytes += expected.size();
}
void Varint(Bytes *bytes, std::uint32_t value) {
  do {
    bytes->push_back((value & 127) | (value >= 128 ? 128 : 0));
    value >>= 7;
  } while (value);
}
metal::SnappyBlocks One(Bytes bytes, std::uint32_t expected) {
  auto size = bytes.size();
  return {std::move(bytes), {0, size}, {expected}};
}
void Reject(metal::SnappyBlocks const &block) {
  bool failed = false;
  try {
    metal::DecompressSnappy(block);
  } catch (std::exception const &e) {
    if (std::string{e.what()}.find("guard overwritten") != std::string::npos)
      throw;
    failed = true;
  }
  Check(failed, "Malformed Snappy input accepted");
  ++invalid;
}
void Decode(Bytes const &encoded, Bytes const &expected) {
  std::string decoded;
  auto src = reinterpret_cast<char const *>(encoded.data());
  Check(snappy::Uncompress(src, encoded.size(), &decoded),
        "CPU rejected handcrafted stream");
  Equal(Bytes(decoded.begin(), decoded.end()), expected);
  Equal(metal::DecompressSnappy(One(encoded, expected.size())), expected);
  ++cases;
}
void RoundTrip(Bytes const &bytes, std::size_t chunk = 1 << 21) {
  auto before = metal::GetStatistics();
  auto compressed = metal::CompressSnappy(bytes, chunk);
  auto after = metal::GetStatistics();
  auto chunks = bytes.size() / chunk + (bytes.size() % chunk != 0);
  Check(after.snappy_compress_dispatches - before.snappy_compress_dispatches ==
            chunks,
        "Actual GPU compression per chunk");
  Check(after.snappy_input_bytes - before.snappy_input_bytes == bytes.size(),
        "Input accounting");
  Check(after.snappy_compressed_bytes - before.snappy_compressed_bytes ==
            compressed.data.size(),
        "Compressed byte accounting");
  Check(compressed.offsets.size() == chunks + 1, "Chunk endpoints");
  Bytes decoded;
  for (std::size_t c = 0; c < chunks; ++c) {
    std::string part;
    auto begin = compressed.offsets[c], end = compressed.offsets[c + 1];
    Check(snappy::Uncompress(
              reinterpret_cast<char const *>(compressed.data.data() + begin),
              end - begin, &part),
          "CPU rejected GPU encoding");
    Check(part.size() == compressed.decoded_sizes[c], "Per-chunk length");
    decoded.insert(decoded.end(), part.begin(), part.end());
  }
  Equal(decoded, bytes);
  Equal(metal::DecompressSnappy(compressed), bytes);
  std::string cpu;
  snappy::Compress(reinterpret_cast<char const *>(bytes.data()), bytes.size(),
                   &cpu);
  Equal(
      metal::DecompressSnappy(One(Bytes(cpu.begin(), cpu.end()), bytes.size())),
      bytes);
  if (bytes.size() >= 32768 &&
      std::all_of(bytes.begin(), bytes.end(),
                  [&](auto x) { return x == bytes[0]; }))
    Check(compressed.data.size() < bytes.size() / 10,
          "Real backreferences must compress repeats");
  ++cases;
}
void Handmade() {
  // All extended-literal forms, including valid non-minimal length encodings.
  for (unsigned extra = 0; extra <= 4; ++extra) {
    std::size_t n = extra == 0   ? 60
                    : extra == 1 ? 256
                    : extra == 2 ? 65536
                                 : 70001;
    Bytes expected(n), encoded;
    for (std::size_t i = 0; i < n; ++i)
      expected[i] = i * 19;
    Varint(&encoded, n);
    encoded.push_back((extra ? 59 + extra : n - 1) << 2);
    for (unsigned i = 0; i < extra; ++i)
      encoded.push_back((n - 1) >> (8 * i));
    encoded.insert(encoded.end(), expected.begin(), expected.end());
    Decode(encoded, expected);
  }
  // COPY1 high offset bits, COPY2/COPY4 length boundaries and overlap.
  for (unsigned kind = 1; kind <= 3; ++kind) {
    for (unsigned offset : {1u, 3u, 255u, 2047u, 65535u, 70000u}) {
      if ((kind == 1 && offset > 2047) || (kind == 2 && offset > 65535))
        continue;
      for (unsigned length :
           (kind == 1 ? std::vector<unsigned>{4, 11}
                      : std::vector<unsigned>{1, 3, 4, 63, 64})) {
        Bytes expected(offset), encoded;
        for (unsigned i = 0; i < offset; ++i)
          expected[i] = (i * 31 + 17) % 251;
        Varint(&encoded, offset + length);
        encoded.push_back(
            62u << 2); // 3-byte literal length, deliberately non-minimal.
        for (unsigned i = 0; i < 3; ++i)
          encoded.push_back((offset - 1) >> (8 * i));
        encoded.insert(encoded.end(), expected.begin(), expected.end());
        encoded.push_back(kind == 1
                              ? 1 | ((length - 4) << 2) | ((offset >> 8) << 5)
                              : kind | ((length - 1) << 2));
        for (unsigned i = 0; i < (kind == 3 ? 4 : kind); ++i)
          encoded.push_back(offset >> (8 * i));
        for (unsigned i = 0; i < length; ++i)
          expected.push_back(expected[i % offset]);
        Decode(encoded, expected);
      }
    }
  }
  Decode({0}, {});
  Decode({0x80, 0x80, 0x80, 0x80, 0}, {}); // Valid non-minimal uint32 header.
}
void Invalid() {
  for (auto bytes : std::vector<Bytes>{{},
                                       {128},
                                       {128, 128, 128, 128, 128},
                                       {255, 255, 255, 255, 16},
                                       {0, 0},
                                       {1}})
    Reject(One(bytes, 0));
  for (auto bytes : std::vector<Bytes>{{1, 0},
                                       {1, 4, 42, 43},
                                       {1, 240},
                                       {1, 252, 255, 255, 255, 255},
                                       {1, 2, 0, 0},
                                       {1, 2, 1, 0},
                                       {1, 2, 1},
                                       {4, 1, 0},
                                       {4, 1},
                                       {1, 3, 1, 0, 0},
                                       {2, 0, 42, 254, 1, 0},
                                       {2, 0, 42, 2, 0, 0}})
    Reject(One(bytes, bytes[0]));
  Reject({{}, {}, {}});
  Reject({{}, {1}, {}});
  Reject({{0}, {0, 2}, {0}});
  Reject({{0}, {0, 2, 1}, {0, 0}});
  // Deterministic mutations: acceptance must agree with Google's actual decoder
  // and the independently supplied destination size, not just our own encoder.
  Bytes original(4096);
  for (std::size_t i = 0; i < original.size(); ++i)
    original[i] = i % 79;
  std::string raw;
  snappy::Compress(reinterpret_cast<char const *>(original.data()),
                   original.size(), &raw);
  std::mt19937 rng{9317};
  for (unsigned i = 0; i < 400; ++i) {
    Bytes mutated(raw.begin(), raw.end());
    if (i % 3 == 0)
      mutated.resize(rng() % mutated.size());
    else if (i % 3 == 1)
      mutated[rng() % mutated.size()] ^= 1u << (rng() % 8);
    else
      mutated.insert(mutated.begin() + rng() % mutated.size(), rng() & 255);
    std::string cpu;
    bool valid =
        snappy::Uncompress(reinterpret_cast<char const *>(mutated.data()),
                           mutated.size(), &cpu) &&
        cpu.size() == original.size();
    if (valid) {
      Equal(metal::DecompressSnappy(One(mutated, original.size())),
            Bytes(cpu.begin(), cpu.end()));
      ++cases;
    } else {
      Reject(One(mutated, original.size()));
    }
  }
  for (std::size_t size : {0u, (1u << 21) + 1}) {
    bool failed = false;
    try {
      metal::CompressSnappy({}, size);
    } catch (std::exception const &) {
      failed = true;
    }
    Check(failed, "Invalid compressor chunk size accepted");
    ++invalid;
  }
}
int main() {
  try {
    std::mt19937 rng{135791};
    for (unsigned pattern = 0; pattern < 4; ++pattern) {
      for (std::size_t n : {0u,
                            1u,
                            2u,
                            3u,
                            4u,
                            5u,
                            7u,
                            60u,
                            61u,
                            127u,
                            128u,
                            255u,
                            256u,
                            257u,
                            32767u,
                            32768u,
                            32769u,
                            65535u,
                            65536u,
                            65537u,
                            (1u << 21) - 1,
                            1u << 21,
                            (1u << 21) + 1,
                            (1u << 22) + 57}) {
        Bytes input(n);
        for (std::size_t i = 0; i < n; ++i)
          input[i] = pattern == 0   ? 0
                     : pattern == 1 ? i % 7
                     : pattern == 2 ? rng() & 255
                                    : (i % 331 < 100 ? rng() & 255 : 42);
        RoundTrip(input);
      }
      std::printf(
          "Snappy pattern %u/4 complete: %zu cases, %zu decoded bytes\n",
          pattern + 1, cases, checked_bytes);
      std::fflush(stdout);
    }
    for (std::size_t chunk : {1u, 31u, 255u, 32769u}) {
      Bytes input(chunk * 2 + 1);
      for (auto &x : input)
        x = rng() & 255;
      RoundTrip(input, chunk);
    }
    Handmade();
    std::puts("All tag/length/offset forms passed; checking malformed streams");
    std::fflush(stdout);
    Invalid();
    auto stats = metal::GetStatistics();
    Check(stats.snappy_compress_dispatches > 100 &&
              stats.snappy_decompress_dispatches > 500,
          "GPU codec was exercised");
    std::printf("PASS Snappy: %zu valid cases, %zu rejected inputs, %zu "
                "fields, %zu exact bytes; "
                "%llu compress / %llu decompress GPU dispatches\n",
                cases, invalid, fields, checked_bytes,
                (unsigned long long)stats.snappy_compress_dispatches,
                (unsigned long long)stats.snappy_decompress_dispatches);
  } catch (std::exception const &e) {
    std::fprintf(stderr, "FAIL: %s\n", e.what());
    return 1;
  }
}
