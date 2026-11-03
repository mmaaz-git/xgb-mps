#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdio>
int main() {
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
      std::fprintf(stderr, "No Metal device is visible to this process.\n");
      return 1;
    }
    std::printf("Metal device: %s; unified memory: %s\n", device.name.UTF8String,
                device.hasUnifiedMemory ? "yes" : "no");
    return 0;
  }
}
