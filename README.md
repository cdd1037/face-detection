# FaceDetection

A standalone, pure C# face detector targeting .NET 8. It runs the existing
53-layer libfacedetection/YuNet model and returns rectangles, five landmarks,
and floating-point scores. The public namespace and assembly are `FaceDetection`; the NuGet package ID is
`Cdd.FaceDetection`, matching the original project.

- No native library, P/Invoke, ONNX Runtime, image codec, or third-party runtime package.
- Embedded model; no runtime downloads or external model setup.
- Hardware-selected SIMD kernels and a scalar fallback.
- Instance-owned reusable workspace and raw BGR24, RGB24, or RGBA32 input.

This is an independent managed implementation derived from the existing inference
algorithm and model, not a newly trained detector. The package is a preview release.
Linux x64 and Windows x64 have been exercised; other platforms and Godot exports require validation.

## Build and test

Install the .NET 8 SDK, then run from this directory:

```sh
dotnet build FaceDetection.sln -c Release
dotnet run --project tests/FaceDetection.Tests -c Release
```

The tests are a dependency-free console executable, not an xUnit/VSTest project.
Use `dotnet run`, not `dotnet test`, to execute assertions. Python and the original
native repository are not needed for ordinary builds or tests. The SDK may restore
Microsoft build-time tools for AOT/trim analysis; these are not runtime dependencies.

## Use in an application

Install the NuGet preview package:

```sh
dotnet add package Cdd.FaceDetection --version 0.1.0-preview.4
```

Or add a local project reference (replace the example paths):

```sh
dotnet add /path/to/App.csproj reference /path/to/FaceDetection/src/FaceDetection/FaceDetection.csproj
```

Alternatively, build a local package and install it from that local feed:

```sh
dotnet pack src/FaceDetection -c Release -o artifacts/packages
dotnet add /path/to/App.csproj package Cdd.FaceDetection --source /absolute/path/to/artifacts/packages --prerelease
```

```csharp
using FaceDetection;

// Pixels are already decoded, interleaved, and in top-down row order.
// Keep one detector per inference worker and reuse it across frames.
using var detector = new FaceDetector();
DetectedFace[] faces = detector.Detect(
    rgbaPixels, width, height, width * 4, PixelFormat.Rgba32);
foreach (var face in faces)
    Console.WriteLine($"{face.Bounds}, score={face.Score}, nose={face.Landmarks.Nose}");

// Optional: drop retained scratch after a large one-off image.
detector.ReleaseWorkspace();
```

The [raw-pixel console sample](samples/DetectFaces) demonstrates file input without
introducing an image-decoding dependency:

```sh
dotnet run --project samples/DetectFaces -c Release -- tests/fixtures/astronaut.rgb 256 256 rgb24
```

## RGB and RGBA input

Use `PixelFormat.Rgb24` for R, G, B bytes and `PixelFormat.Rgba32` for R, G, B, A
bytes. Pass the actual row stride: normally `width * 3` or `width * 4`, respectively.
Alpha is ignored, including fully transparent pixels. Premultiplied RGB values are
not unpremultiplied, so the caller should supply straight color channels.
BGR24 remains available and is the default.

```csharp
DetectedFace[] rgbFaces = detector.Detect(
    rgbPixels, width, height, width * 3, PixelFormat.Rgb24);
DetectedFace[] rgbaFaces = detector.Detect(
    rgbaPixels, width, height, width * 4, PixelFormat.Rgba32);
```

## Input and output

- Input channels are unsigned 8-bit values. RGBA alpha is ignored; premultiplied
  colors are not unpremultiplied. JPEG and PNG files must be decoded by the caller.
- Each dimension is 1–4096; total pixels must not exceed 4,194,304.
- Stride is positive and at least `width * bytesPerPixel`. Required input length is
  `(height - 1) * stride + width * bytesPerPixel`; final-row padding is optional.
- Input is neither retained nor modified. Do not modify it during inference.
- Rectangles and landmarks use original-image integer pixels and are not clipped.
  Left/right landmark names refer to the subject, not the viewer.
- Scores are uncalibrated model scores. Fixed thresholds are score >= 0.2 and
  NMS IoU 0.45, with 1000 candidates and at most 512 returned faces.
- Invalid input throws argument exceptions. Valid calls after disposal throw
  `ObjectDisposedException`. Allocation failures may throw `OutOfMemoryException`.

## Lifetime and memory

Calls, workspace release, and disposal on one instance serialize. For parallel
inference, use a bounded pool of independent instances. Each owns reusable scratch;
model weights are shared and immutable. Returned arrays belong to the caller.
After warmup at a stable size, nonempty calls allocate their result array; empty
results reuse `Array.Empty<DetectedFace>()`.

Workspace capacity grows with demand and stays until release, disposal, or garbage
collection. Release drops references, not necessarily process resident memory.
`RetainedWorkspaceBytes` and `SharedModelBytes` report array payload, excluding
object headers, runtime overhead, and caller-owned buffers.

## Scalar builds and portability

```sh
dotnet run --project tests/FaceDetection.Tests -c Release -p:FaceDetectionScalar=true
```

This selects scalar detector algorithms, including for AOT builds. It does not
forbid runtime copies or scalar floating-point instructions using vector registers.
Rebuild without the property to restore hardware-selected kernels.
`FaceDetector.IsSimdEnabled` and `SimdFloatLanes` report the detector selection.

See [architecture](docs/design.md), [testing](docs/testing.md),
[platform and Godot notes](docs/portability.md), and [model provenance](docs/provenance.md).

## License

BSD-3-Clause. Preserve [LICENSE](LICENSE), [third-party notices](THIRD-PARTY-NOTICES.txt),
and the notices under [licenses/libfacedetection](licenses/libfacedetection) when
redistributing code or model weights. The test portrait has separate public-domain
provenance in [tests/fixtures/README.md](tests/fixtures/README.md).
