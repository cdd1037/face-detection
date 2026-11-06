# Optional managed benchmark

```sh
dotnet run -c Release --project benchmarks/FaceDetection.Benchmarks -- 15
```

The optional positional argument is the number of measured iterations (1–1000; default 15). The console emits JSON lines with runtime/SIMD diagnostics and per-resolution first-call latency, warmed median/min/max latency, current-thread allocations and retained workspace bytes. Input resizing and output serialization are excluded from timed detection calls. Each resolution has five warmup calls.

This is a smoke benchmark, not a statistical comparison harness. The first 256×256 call includes lazy model initialization; later resolutions reuse the shared model and must not be interpreted as fresh-process cold-start measurements. Tiered JIT, scheduling, CPU scaling and the runtime can affect results. Do not use its timings as a regression threshold. There are no native bindings, external package dependencies, baseline assembly switches or platform-specific `/proc` reads.
