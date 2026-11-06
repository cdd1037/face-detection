namespace FaceDetection;

/// <summary>
/// Pure C# detector with reusable, instance-owned scratch. Calls on the same instance serialize;
/// use a bounded pool of instances for parallel inference. Dispose releases scratch references.
/// </summary>
public sealed class FaceDetector : IDisposable
{
    /// <summary>Largest accepted image width or height in pixels.</summary>
    public const int MaximumDimension = 4096;
    /// <summary>Largest accepted product of image width and height.</summary>
    public const int MaximumPixelCount = 4_194_304;
    /// <summary>Maximum number of faces returned by one detection.</summary>
    public const int MaximumFaceCount = 512;
    private readonly object gate = new();
    private Network? network;
    private bool disposed;

    /// <summary>Creates a detector. Model and workspace allocation are deferred until first use.</summary>
    public FaceDetector() { }

    /// <summary>Whether this library build and runtime enable SIMD inference kernels.</summary>
    public static bool IsSimdEnabled => Kernels.SimdEnabled;
    /// <summary>Float lanes used by inference, or one when scalar inference is selected.</summary>
    public static int SimdFloatLanes => IsSimdEnabled ? System.Numerics.Vector<float>.Count : 1;

    /// <summary>Current retained tensor and postprocessing array payload, excluding shared weights and object headers.</summary>
    public long RetainedWorkspaceBytes { get { lock (gate) return network?.RetainedBytes ?? 0; } }
    /// <summary>Shared immutable model array payload, including packed weights; loaded on first access.</summary>
    public static long SharedModelBytes => Model.Instance.RetainedBytes;

    /// <summary>
    /// Detect faces in decoded pixels. Input must not be modified during the call. No caller memory is retained.
    /// Final-row padding is optional. Results own their memory. Empty results share an empty array.
    /// </summary>
    /// <param name="pixels">Decoded, interleaved, unsigned eight-bit channels in top-down rows.</param>
    /// <param name="width">Image width, from 1 through <see cref="MaximumDimension"/>.</param>
    /// <param name="height">Image height, from 1 through <see cref="MaximumDimension"/>.</param>
    /// <param name="stride">Positive row stride in bytes, at least the packed row length.</param>
    /// <param name="format">Channel order. RGBA alpha is ignored; colors are not unpremultiplied.</param>
    /// <returns>Faces in descending score order. Coordinates refer to the input image and are not clipped.</returns>
    /// <exception cref="ArgumentOutOfRangeException">Dimensions, total pixel count, stride or format are invalid.</exception>
    /// <exception cref="ArgumentException">The input buffer does not cover the declared image layout.</exception>
    /// <exception cref="ObjectDisposedException">This detector is disposed and the supplied layout is valid.</exception>
    public DetectedFace[] Detect(ReadOnlySpan<byte> pixels, int width, int height, int stride,
        PixelFormat format = PixelFormat.Bgr24)
    {
        ValidateInput(pixels.Length, width, height, stride, format);
        lock (gate)
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            // Publish a network only after model initialization succeeds. Allocation failures can be retried.
            network ??= new Network(Model.Instance);
            try { return network.Run(pixels, width, height, stride, format); }
            catch { network = null; throw; }
        }
    }

    /// <summary>Release retained scratch without disposing the detector. Shared immutable weights stay loaded.</summary>
    public void ReleaseWorkspace() { lock (gate) { ObjectDisposedException.ThrowIf(disposed, this); network = null; } }

    /// <summary>Releases workspace references and prevents further detection. Repeated calls are harmless.</summary>
    public void Dispose() { lock (gate) { network = null; disposed = true; } }

    private static void ValidateInput(int length, int width, int height, int stride, PixelFormat format)
    {
        if (width <= 0 || width > MaximumDimension) throw new ArgumentOutOfRangeException(nameof(width));
        if (height <= 0 || height > MaximumDimension || (long)width * height > MaximumPixelCount)
            throw new ArgumentOutOfRangeException(nameof(height));
        if (format is not PixelFormat.Bgr24 and not PixelFormat.Rgb24 and not PixelFormat.Rgba32)
            throw new ArgumentOutOfRangeException(nameof(format));
        int bytesPerPixel = format == PixelFormat.Rgba32 ? 4 : 3;
        if (stride < width * bytesPerPixel) throw new ArgumentOutOfRangeException(nameof(stride));
        if ((long)(height - 1) * stride + width * bytesPerPixel > length)
            throw new ArgumentException("Pixel buffer is shorter than the image layout requires.", "pixels");
    }
}
