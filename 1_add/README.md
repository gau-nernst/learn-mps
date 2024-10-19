# Add kernel

Resources:
- https://www.youtube.com/watch?v=VQK28rRK6OU and https://github.com/twohyjr/Metal-Game-Engine-Tutorial/tree/master/MetalCode/AddCompute/AddCompute
- https://developer.apple.com/documentation/metal/performing_calculations_on_a_gpu?language=objc

Objective-C resources:
- https://www.hackingwithswift.com/articles/114/objective-c-to-swift-conversion-cheat-sheet
- https://en.wikipedia.org/wiki/Objective-C
- https://developer.apple.com/library/archive/documentation/Cocoa/Conceptual/ProgrammingWithObjectiveC/Introduction/Introduction.html

```bash
xcrun metal *.metal
clang++ -std=c++17 main.mm -o main -framework Metal -framework Foundation
./main
```

Learnings:
- Set up Objective-C language server: https://github.com/MaskRay/ccls (can be installed with homebrew).
- Compile Metal shaders with `xcrun metal *.metal`. This will produce `default.metallib`. In host code, load this default library with `id<MTLLibrary> lib = [device newDefaultLibrary]`.
- Metal shader concepts: Compute pipeline, command queue, command buffer, command encoder, grid, thread group.
