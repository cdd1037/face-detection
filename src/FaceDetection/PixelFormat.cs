namespace FaceDetection;

/// <summary>Interleaved eight-bit channels in top-down rows.</summary>
public enum PixelFormat
{
    /// <summary>Blue, green, red; three bytes per pixel.</summary>
    Bgr24 = 0,
    /// <summary>Red, green, blue; three bytes per pixel.</summary>
    Rgb24 = 1,
    /// <summary>Red, green, blue, alpha; four bytes per pixel. Alpha is ignored.</summary>
    Rgba32 = 2
}
