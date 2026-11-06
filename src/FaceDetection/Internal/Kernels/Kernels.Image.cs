using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;

namespace FaceDetection;

internal static partial class Kernels
{
    internal static bool ImageConvolution(ReadOnlySpan<byte> pixels, int width, int height, int stride,
        PixelFormat format, Filter filter, Tensor output)
    {
        // Fuse the fixed first-layer patch expansion and 1x1 convolution. Five padded
        // channels are always zero, so their multiplications and the 32-channel input
        // tensor can be omitted. Keep the portable existing transform as a fallback.
        if (!SimdEnabled || Vector<float>.Count is not 4 and not 8) return false;
        int rows = ((height + 31) / 32) * 16, cols = ((width + 31) / 32) * 16;
        output.Resize(rows, cols, 16);
        int bytes = format == PixelFormat.Rgba32 ? 4 : 3;
        int blue = format == PixelFormat.Bgr24 ? 0 : 2, red = 2 - blue;
        float[] weights = filter.Packed, biases = filter.Biases, dst = output.Data;
        ref byte source = ref MemoryMarshal.GetReference(pixels);
        if (Vector<float>.Count == 8)
        {
            for (int y = 0; y < rows; y++)
                for (int x = 0; x < cols; x++)
                {
                    var a0 = Load(biases, 0);
                    var a1 = Load(biases, 8);
                    int centerY = y * 2, centerX = x * 2;
                    for (int dy = -1; dy <= 1; dy++)
                    {
                        int sy = centerY + dy;
                        if ((uint)sy >= (uint)height) continue;
                        for (int dx = -1; dx <= 1; dx++)
                        {
                            int sx = centerX + dx;
                            if ((uint)sx >= (uint)width) continue;
                            int si = sy * stride + sx * bytes;
                            int wi = ((dy + 1) * 3 + dx + 1) * 3 * 16;
                            var v0 = new Vector<float>(Unsafe.Add(ref source, si + blue));
                            a0 = MultiplyAdd(v0, Load(weights, wi + 0), a0);
                            a1 = MultiplyAdd(v0, Load(weights, wi + 8), a1);
                            var v1 = new Vector<float>(Unsafe.Add(ref source, si + 1));
                            a0 = MultiplyAdd(v1, Load(weights, wi + 16), a0);
                            a1 = MultiplyAdd(v1, Load(weights, wi + 24), a1);
                            var v2 = new Vector<float>(Unsafe.Add(ref source, si + red));
                            a0 = MultiplyAdd(v2, Load(weights, wi + 32), a0);
                            a1 = MultiplyAdd(v2, Load(weights, wi + 40), a1);
                        }
                    }
                    int di = (y * cols + x) * 16;
                    Store(Vector.Max(a0, Vector<float>.Zero), dst, di + 0);
                    Store(Vector.Max(a1, Vector<float>.Zero), dst, di + 8);
                }
            return true;
        }
        if (Vector<float>.Count == 4)
        {
            for (int y = 0; y < rows; y++)
                for (int x = 0; x < cols; x++)
                {
                    var a0 = Load(biases, 0);
                    var a1 = Load(biases, 4);
                    var a2 = Load(biases, 8);
                    var a3 = Load(biases, 12);
                    int centerY = y * 2, centerX = x * 2;
                    for (int dy = -1; dy <= 1; dy++)
                    {
                        int sy = centerY + dy;
                        if ((uint)sy >= (uint)height) continue;
                        for (int dx = -1; dx <= 1; dx++)
                        {
                            int sx = centerX + dx;
                            if ((uint)sx >= (uint)width) continue;
                            int si = sy * stride + sx * bytes;
                            int wi = ((dy + 1) * 3 + dx + 1) * 3 * 16;
                            var v0 = new Vector<float>(Unsafe.Add(ref source, si + blue));
                            a0 = MultiplyAdd(v0, Load(weights, wi + 0), a0);
                            a1 = MultiplyAdd(v0, Load(weights, wi + 4), a1);
                            a2 = MultiplyAdd(v0, Load(weights, wi + 8), a2);
                            a3 = MultiplyAdd(v0, Load(weights, wi + 12), a3);
                            var v1 = new Vector<float>(Unsafe.Add(ref source, si + 1));
                            a0 = MultiplyAdd(v1, Load(weights, wi + 16), a0);
                            a1 = MultiplyAdd(v1, Load(weights, wi + 20), a1);
                            a2 = MultiplyAdd(v1, Load(weights, wi + 24), a2);
                            a3 = MultiplyAdd(v1, Load(weights, wi + 28), a3);
                            var v2 = new Vector<float>(Unsafe.Add(ref source, si + red));
                            a0 = MultiplyAdd(v2, Load(weights, wi + 32), a0);
                            a1 = MultiplyAdd(v2, Load(weights, wi + 36), a1);
                            a2 = MultiplyAdd(v2, Load(weights, wi + 40), a2);
                            a3 = MultiplyAdd(v2, Load(weights, wi + 44), a3);
                        }
                    }
                    int di = (y * cols + x) * 16;
                    Store(Vector.Max(a0, Vector<float>.Zero), dst, di + 0);
                    Store(Vector.Max(a1, Vector<float>.Zero), dst, di + 4);
                    Store(Vector.Max(a2, Vector<float>.Zero), dst, di + 8);
                    Store(Vector.Max(a3, Vector<float>.Zero), dst, di + 12);
                }
            return true;
        }
        return false;
    }

    internal static void ImageToTensor(ReadOnlySpan<byte> pixels, int width, int height, int stride, PixelFormat format, Tensor output)
    {
        int rows = ((height + 31) / 32) * 16, cols = ((width + 31) / 32) * 16;
        output.Resize(rows, cols, 32);
        float[] dst = output.Data;
        int bytes = format == PixelFormat.Rgba32 ? 4 : 3;
        int blue = format == PixelFormat.Bgr24 ? 0 : 2, red = 2 - blue;
        for (int row = 0; row < rows; row++)
            for (int col = 0; col < cols; col++)
            {
                int start = (row * cols + col) * 32;
                int centerY = row * 2, centerX = col * 2;
                if (centerY > 0 && centerY + 1 < height && centerX > 0 && centerX + 1 < width)
                {
                    // Interior pixels cover the common case. All nine source pixels are valid,
                    // so avoid clearing and rechecking 27 channels that will be overwritten.
                    ref byte source = ref MemoryMarshal.GetReference(pixels);
                    ref float target = ref Unsafe.Add(ref MemoryMarshal.GetArrayDataReference(dst), start);
                    int top = (centerY - 1) * stride + (centerX - 1) * bytes;
                    Unsafe.Add(ref target, 0) = Unsafe.Add(ref source, top + 0 * stride + 0 * bytes + blue);
                    Unsafe.Add(ref target, 1) = Unsafe.Add(ref source, top + 0 * stride + 0 * bytes + 1);
                    Unsafe.Add(ref target, 2) = Unsafe.Add(ref source, top + 0 * stride + 0 * bytes + red);
                    Unsafe.Add(ref target, 3) = Unsafe.Add(ref source, top + 0 * stride + 1 * bytes + blue);
                    Unsafe.Add(ref target, 4) = Unsafe.Add(ref source, top + 0 * stride + 1 * bytes + 1);
                    Unsafe.Add(ref target, 5) = Unsafe.Add(ref source, top + 0 * stride + 1 * bytes + red);
                    Unsafe.Add(ref target, 6) = Unsafe.Add(ref source, top + 0 * stride + 2 * bytes + blue);
                    Unsafe.Add(ref target, 7) = Unsafe.Add(ref source, top + 0 * stride + 2 * bytes + 1);
                    Unsafe.Add(ref target, 8) = Unsafe.Add(ref source, top + 0 * stride + 2 * bytes + red);
                    Unsafe.Add(ref target, 9) = Unsafe.Add(ref source, top + 1 * stride + 0 * bytes + blue);
                    Unsafe.Add(ref target, 10) = Unsafe.Add(ref source, top + 1 * stride + 0 * bytes + 1);
                    Unsafe.Add(ref target, 11) = Unsafe.Add(ref source, top + 1 * stride + 0 * bytes + red);
                    Unsafe.Add(ref target, 12) = Unsafe.Add(ref source, top + 1 * stride + 1 * bytes + blue);
                    Unsafe.Add(ref target, 13) = Unsafe.Add(ref source, top + 1 * stride + 1 * bytes + 1);
                    Unsafe.Add(ref target, 14) = Unsafe.Add(ref source, top + 1 * stride + 1 * bytes + red);
                    Unsafe.Add(ref target, 15) = Unsafe.Add(ref source, top + 1 * stride + 2 * bytes + blue);
                    Unsafe.Add(ref target, 16) = Unsafe.Add(ref source, top + 1 * stride + 2 * bytes + 1);
                    Unsafe.Add(ref target, 17) = Unsafe.Add(ref source, top + 1 * stride + 2 * bytes + red);
                    Unsafe.Add(ref target, 18) = Unsafe.Add(ref source, top + 2 * stride + 0 * bytes + blue);
                    Unsafe.Add(ref target, 19) = Unsafe.Add(ref source, top + 2 * stride + 0 * bytes + 1);
                    Unsafe.Add(ref target, 20) = Unsafe.Add(ref source, top + 2 * stride + 0 * bytes + red);
                    Unsafe.Add(ref target, 21) = Unsafe.Add(ref source, top + 2 * stride + 1 * bytes + blue);
                    Unsafe.Add(ref target, 22) = Unsafe.Add(ref source, top + 2 * stride + 1 * bytes + 1);
                    Unsafe.Add(ref target, 23) = Unsafe.Add(ref source, top + 2 * stride + 1 * bytes + red);
                    Unsafe.Add(ref target, 24) = Unsafe.Add(ref source, top + 2 * stride + 2 * bytes + blue);
                    Unsafe.Add(ref target, 25) = Unsafe.Add(ref source, top + 2 * stride + 2 * bytes + 1);
                    Unsafe.Add(ref target, 26) = Unsafe.Add(ref source, top + 2 * stride + 2 * bytes + red);
                    Unsafe.Add(ref target, 27) = 0; Unsafe.Add(ref target, 28) = 0; Unsafe.Add(ref target, 29) = 0;
                    Unsafe.Add(ref target, 30) = 0; Unsafe.Add(ref target, 31) = 0;
                    continue;
                }
                Array.Clear(dst, start, 32);
                for (int dy = -1; dy <= 1; dy++)
                {
                    int y = row * 2 + dy;
                    if ((uint)y >= (uint)height) continue;
                    for (int dx = -1; dx <= 1; dx++)
                    {
                        int x = col * 2 + dx;
                        if ((uint)x >= (uint)width) continue;
                        int src = y * stride + x * bytes, target = start + ((dy + 1) * 3 + dx + 1) * 3;
                        dst[target] = pixels[src + blue]; dst[target + 1] = pixels[src + 1]; dst[target + 2] = pixels[src + red];
                    }
                }
            }
    }
}
