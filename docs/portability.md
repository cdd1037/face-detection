# Portability and engine integration

The library targets `net8.0` and uses only .NET libraries. It needs no bundled
native binary, model download, or image decoder. Runtime feature checks select
supported SIMD/FMA paths; portable vector and scalar fallbacks remain available.
This design enables portability but does not establish platform support by itself.

Linux x64 JIT and standalone Native AOT execution have been exercised. ARM64,
Windows, macOS, Android, iOS, Web, and Godot editor/export targets have not been
validated. Disabling hardware intrinsics on x64 does not test ARM64. A configured
CI job is not a passing test, and standalone AOT is not a Godot export test.
Native AOT publication may require platform SDK/compiler prerequisites even though
the resulting detector has no third-party native runtime dependency.

For deliberate scalar selection, use `-p:FaceDetectionScalar=true` on build,
run, or publish. This controls detector algorithms, not every instruction emitted
by the runtime. Inspect `FaceDetector.IsSimdEnabled` and `SimdFloatLanes` rather
than assuming a path from the CPU name. Rebuild normally to restore SIMD selection.

## Godot C# integration checklist

Use a Godot C# project with a compatible .NET target and add a local project/package
reference. Godot's own C# platform/export limitations still apply. This checklist
is guidance, not a runtime-tested Godot sample:

1. Capture a safe CPU image snapshot through the engine's supported API. Decompress
   and convert it to RGB8 or RGBA8 before passing pixels to the detector.
2. Supply top-down interleaved bytes and the actual row stride. RGBA alpha is
   ignored; undo premultiplication yourself if needed.
3. Reuse a detector per worker. Keep frame queues bounded, usually retaining only
   the newest frame. Do not schedule an unbounded task per rendered frame.
4. Run only inference on workers; access scene/UI objects using the engine's
   supported thread and scheduling rules.
5. Map output coordinates through any resize scale, crop, letterbox, and display
   transform. Output coordinates are not clipped to the image.
6. Finish outstanding work before disposal. Disposal waits for the instance lock;
   there is no mid-inference cancellation API.
7. Test the actual editor, release export, hardware path, memory budget, and
   lifecycle. Image capture/readback and decoding costs are separate from inference.
