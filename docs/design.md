# Architecture

## Boundaries

The library accepts decoded pixels and returns detections. Image codecs, UI/engine
integration, scheduling, and native comparison tools are outside its runtime boundary.
The public `FaceDetection` API remains small; implementation types are internal.
There are no direct native allocations, third-party runtime package dependencies,
or consumer model-generation hooks.

`FaceDetector` validates input and serializes access to its instance-owned network.
The network executes the fixed graph using reusable tensor storage. Model loading
validates the layer schema and prepares shared immutable weights. Postprocessing
performs decoding, stable score ordering, and greedy IoU suppression.

## Graph and arithmetic

Input is zero-padded to a multiple of 32. A stride-two 3×3 BGR patch transform gives
27 channels and five zero padding channels before the first pointwise convolution.
RGB/RGBA channel selection occurs during this transform. Supported SIMD widths fuse
the transform with the first convolution; other paths retain the explicit transform.

The 53-layer graph retains the original layer order and ReLU placement: backbone
pointwise/depthwise units and max pools, three feature levels at strides 8/16/32,
nearest-neighbor upsample/add lateral connections, and classification, box,
objectness, and five-landmark heads. Postprocessing uses clamped sigmoid,
square-root classification/objectness scores, exponential box decoding, and NMS.
Ties retain scale/row/column order. Landmarks are decoded only for surviving boxes.

This organizational refactor preserves the existing graph, model bytes, and kernel
arithmetic. It does not claim bitwise equality across hardware: reduction ordering,
FMA, and transcendental implementations can affect values near decision boundaries.

## Kernels and storage

Tensors use contiguous NHWC float storage. Pointwise kernels pack larger output
channel groups once and reuse broadcast inputs across accumulator tiles. Small
heads use dot products. Depthwise kernels share weights across adjacent pixels;
borders and tails have separate paths. Pooling and addition have vector paths.

`System.Numerics.Vector<float>` selects vector width; guarded BCL FMA/ARM intrinsics
are optional. Unsupported hardware falls back to portable vector or scalar paths.
`FaceDetectionScalar=true` forces the detector's scalar route at build time.
Scalar convolutions retain per-output accumulation order while sharing input loads.
Ref-based `Unsafe.Add` and vector load/store helpers rely on validated internal shape
and loop bounds; they do not require C# unsafe blocks or native buffers.

Scratch grows on demand and is overwritten before use. Failed inference discards
that instance's scratch. Shared model initialization is synchronized; a failed
initialization can be retried. Array-payload diagnostics are not measurements of
process RSS or per-call allocation. Limits protect layouts, not latency or memory
budgets: applications should bound image sizes and worker count for their devices.

## Package and API identity

The project, namespace and assembly are named `FaceDetection`. The NuGet package
ID remains `Cdd.FaceDetection`, matching the original project. This aligns package
identity; it does not make the instance-based managed API a binary-compatible
replacement for the original static native wrapper. Use `new FaceDetector()` and
retain/dispose its workspace as described in the README. Earlier standalone
prototype consumers must update their namespace/project references to `FaceDetection`.

The scalar build switch is `FaceDetectionScalar`. The embedded model retains its
existing `MFD1` binary format marker so renaming does not change trained weights or
model interpretation.
