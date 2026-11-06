# Testing

Run all commands from the repository root with the .NET 8 SDK.

```sh
dotnet build FaceDetection.sln -c Release
dotnet run --project tests/FaceDetection.Tests -c Release
dotnet run --project tests/FaceDetection.Tests -c Release -p:FaceDetectionScalar=true
# Restore normal hardware selection after the scalar build.
dotnet run --project tests/FaceDetection.Tests -c Release
dotnet pack src/FaceDetection -c Release -o artifacts/packages
```

The single test project is a dependency-free executable. Its exit code signals
success or failure. `dotnet test` is not the assertion runner. Tests exercise the
public API, kernels, and postprocessing, including input layouts, color formats,
workspace lifetime, concurrency, scalar selection, and reference/canary checks.
The public-domain portrait's source is recorded in `tests/fixtures/README.md`.

Postprocessing allocation assertions compare exact total bytes with result-array
allocation in a bounded 16 MiB no-GC region. The test fails explicitly if that
region cannot be reserved or maintained. This isolates measurement windows; the
library does not change application GC settings.

A portrait and transformed/synthetic images provide regression coverage, not a
representative accuracy dataset or a demographic-quality evaluation. Do not infer
accuracy parity across arbitrary scenes from those fixtures. Future model or
arithmetic changes should include comparisons against the original implementation,
including detections near score, NMS, and integer-coordinate boundaries.

The repository also provides `scripts/verify.sh` and `scripts/verify.ps1` to run
the build, default and fallback tests, scalar rebuild, normal-build restoration,
and package creation together.

## Verification scope

The refactored tree was verified on Linux x64 with .NET SDK 8.0.427 / runtime
8.0.33. All 39 named checks passed under default SIMD, reduced four-lane SIMD,
runtime-disabled intrinsics, compile-time scalar, and restored default builds.
Standalone Native AOT passed the same suite in hardware-selected and scalar builds;
the raw-pixel sample also ran as an AOT executable.

Additional local checks covered package contents (no runtime package/native assets),
a fresh package-only consumer, model conversion round-trip and rejection without
overwriting the destination, and 36 transformed/synthetic-image comparisons against
the original native implementation. Those comparisons used a separate temporary
harness; no native library is required by the maintained tests or package.

These are Linux x64 results, not a claim that every deployment target is verified.
CI configuration expresses intended coverage, not evidence that hosted jobs have
already passed. Local Windows x64 verification with .NET SDK 10.0.200 also passed
all 39 checks in each of the five modes above, including the PowerShell verification
script and package creation. ARM64 hardware, macOS, mobile, Web, and actual Godot
editor/export integration remain unverified. Native AOT needs its
platform SDK/toolchain at application publish time; normal managed consumers do not
compile C++ or build the original detector.

## Performance measurements

The optional dependency-free benchmark uses the same raw test fixture:

```sh
dotnet run --project benchmarks/FaceDetection.Benchmarks -c Release -- 15
```

The optional argument is the measured iteration count (1–1000, default 15). It
prints JSON lines with runtime/SIMD metadata, first-call timing, warmed median/
minimum/maximum, allocation counts, and workspace. Only the first resolution can
include cold shared model initialization; later first calls are not process-cold.
See `benchmarks/README.md` for details.

Build Release and record runtime version, CPU, image dimensions, iteration count,
hardware/scalar selection, and tiering settings with any timing. Separate cold
first-call measurements from warmed throughput. Keep warmups outside timing and
repeat runs under representative application load.

Managed allocation counters exclude native allocations. Retained workspace counts
array payload, not total managed heap or resident memory. Process RSS includes the
runtime, JIT, input, model, scratch, and allocator retention. Single-image timings
exclude image decoding, engine readback, display, and application scheduling.
No performance threshold or cross-platform speed guarantee is implied.
