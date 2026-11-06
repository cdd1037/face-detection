# Regression tests

Run the dependency-free console test runner from the repository root:

```sh
dotnet run -c Release --project tests/FaceDetection.Tests
```

A nonzero exit code indicates failure. The runner accepts no switches, reports named checks, and continues with the remaining suites after a suite fails. It references the actual production assembly using `InternalsVisibleTo`; no production source is linked, copied, or stubbed.

Suites run in this order:

- **API:** synchronized first model access, fixture detections and landmarks, RGB/BGR/RGBA parity, row padding and odd dimensions, input immutability, argument and buffer bounds, zero-allocation warm empty detection, workspace release, idempotent disposal, independent and shared-instance concurrency.
- **Kernels:** independent double-accumulation convolution oracle, pooling/upsampling/image conversion, fused-path parity and fallback, input immutability and spare-capacity canaries, NaN/infinity/signed-zero behavior, production/public SIMD diagnostic agreement.
- **Postprocess:** ordering and tie stability, top-K and output limits, result-array-only warm allocation, scratch release, cross-scale suppression, IoU/confidence threshold boundaries, and degenerate boxes. Synthetic heads use the real production `Tensor` type.

The API suite runs first so no kernel or postprocess setup initializes the shared model before the first-use concurrency check. Concurrency participants start behind an explicit gate; overlap and timing depend on the host scheduler.

To exercise the compile-time scalar fallback:

```sh
dotnet run -c Release --project tests/FaceDetection.Tests -p:FaceDetectionScalar=true
```

The fixture's origin and license are documented in [fixtures/README.md](fixtures/README.md).
