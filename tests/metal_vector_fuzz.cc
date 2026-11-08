// Stateful randomized vector operations checked against an independent host model.
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#include "xgboost/host_device_vector.h"

using xgboost::DeviceOrd;
using xgboost::HostDeviceVector;

template <typename T>
void Run(std::uint32_t seed) {
  std::mt19937 random{seed};
  std::array<HostDeviceVector<T>, 4> actual;
  std::array<std::vector<T>, 4> expected;
  std::size_t step = 0, operation = 0, target = 0, source = 0;
  auto verify = [&](std::size_t index) {
    if (actual[index].ConstHostVector() != expected[index]) {
      throw std::runtime_error("Payload differs from host model");
    }
  };
  try {
    for (; step < 512; ++step) {
      target = random() % actual.size();
      source = random() % actual.size();
      operation = random() % 12;
      auto& vector = actual[target];
      auto& model = expected[target];
      auto value = static_cast<T>(random() % 97);
      auto size = random() % 129;
      switch (operation) {
        case 0:
          vector.Resize(size, value);
          model.resize(size, value);
          break;
        case 1:
          vector.Fill(value);
          std::fill(model.begin(), model.end(), value);
          break;
        case 2:
          vector.SetDevice(random() % 2 ? DeviceOrd::CPU() : DeviceOrd::MPS());
          break;
        case 3:
          verify(target);
          break;
        case 4: {
          auto span = vector.ConstDeviceSpan();
          if (vector.Device().IsCPU()) {
            if (!span.empty()) throw std::runtime_error("CPU device span is not empty");
          } else if (span.size() != model.size() ||
                     !std::equal(span.begin(), span.end(), model.begin())) {
            throw std::runtime_error("Device payload differs from host model");
          }
          break;
        }
        case 5:
          vector.HostVector().push_back(value);
          model.push_back(value);
          break;
        case 6:
          vector.SetDevice(DeviceOrd::MPS());
          if (!model.empty()) {
            auto index = random() % model.size();
            vector.DevicePointer()[index] = value;
            model[index] = value;
          }
          break;
        case 7:
          vector.Resize(expected[source].size(), T{});
          model.resize(expected[source].size(), T{});
          vector.Copy(actual[source]);
          model = expected[source];
          break;
        case 8:
          if (model.size() + expected[source].size() <= 256) {
            auto snapshot = expected[source];  // Includes self-extension.
            vector.Extend(actual[source]);
            model.insert(model.end(), snapshot.begin(), snapshot.end());
          }
          break;
        case 9:
          vector = std::move(actual[source]);
          if (source != target) {
            model = std::move(expected[source]);
            expected[source].clear();
          }
          break;
        case 10: {
          auto replacement = model;
          for (auto& element : replacement) element = static_cast<T>(random() % 97);
          vector.Copy(replacement);
          model = std::move(replacement);
          break;
        }
        case 11:
          // The no-fill overload intentionally leaves newly grown device data
          // uninitialized. Shrinking has an exact, backend-independent contract.
          size = std::min<std::size_t>(size, model.size());
          vector.Resize(size);
          model.resize(size);
          break;
      }
      for (std::size_t i = 0; i < actual.size(); ++i) {
        if (actual[i].Size() != expected[i].size()) {
          throw std::runtime_error("Size differs from host model");
        }
        // Do not read back after every operation: preserve random authority
        // transitions and stale mirrors between these checkpoints.
        if (step % 31 == 0) verify(i);
      }
    }
    for (std::size_t i = 0; i < actual.size(); ++i) verify(i);
  } catch (std::exception const& error) {
    std::fprintf(stderr, "FAIL width=%zu seed=%u step=%zu op=%zu target=%zu source=%zu: %s\n",
                 sizeof(T), seed, step, operation, target, source, error.what());
    throw;
  }
}

int main() {
  try {
    for (std::uint32_t seed = 0; seed < 16; ++seed) {
      Run<float>(seed);
      Run<double>(seed);
      Run<std::uint8_t>(seed);
      std::printf("PASS seed=%u (1536 operations)\n", seed);
      std::fflush(stdout);
    }
    std::puts("PASS 24576 stateful vector operations");
  } catch (...) {
    return 1;
  }
}
