using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;

namespace FaceDetection;

internal static partial class Kernels
{
    internal static void Convolution(Tensor input, Filter filter, bool relu, Tensor output)
    {
        if (input.Channels != filter.InputChannels) throw new InvalidOperationException("Model and tensor channel counts differ.");
        output.Resize(input.Rows, input.Cols, filter.OutputChannels);
        if (!SimdEnabled)
        {
            if (filter.Depthwise) ScalarKernels.Depthwise(input, filter, relu, output);
            else ScalarKernels.Pointwise(input, filter, relu, output);
        }
        else if (filter.Depthwise) Depthwise(input, filter, relu, output);
        else if (filter.Packed.Length != 0 && SimdEnabled && filter.OutputChannels % Vector<float>.Count == 0) PointwisePacked(input, filter, relu, output);
        else PointwiseDot(input, filter, relu, output);
    }

    private static void PointwisePacked(Tensor input, Filter filter, bool relu, Tensor output)
    {
        int icount = input.Channels, ocount = output.Channels, lanes = Vector<float>.Count, pixels = input.Rows * input.Cols;
        float[] src = input.Data, dst = output.Data, weights = filter.Packed, biases = filter.Biases;
        if (ocount == 2 * lanes)
        {
            // First layers: two pixels and two output vectors, sharing all weights.
            int p = 0;
            for (; p + 1 < pixels; p += 2)
            {
                int si = p * icount, di = p * ocount;
                var a0 = Load(biases, 0); var a1 = Load(biases, lanes);
                var b0 = a0; var b1 = a1;
                for (int ic = 0, wi = 0; ic < icount; ic++, wi += ocount)
                {
                    var w0 = Load(weights, wi); var w1 = Load(weights, wi + lanes);
                    var v0 = new Vector<float>(Read(src, si + ic));
                    var v1 = new Vector<float>(Read(src, si + icount + ic));
                    a0 = MultiplyAdd(v0, w0, a0); a1 = MultiplyAdd(v0, w1, a1);
                    b0 = MultiplyAdd(v1, w0, b0); b1 = MultiplyAdd(v1, w1, b1);
                }
                if (relu) { a0 = Vector.Max(a0, Vector<float>.Zero); a1 = Vector.Max(a1, Vector<float>.Zero); b0 = Vector.Max(b0, Vector<float>.Zero); b1 = Vector.Max(b1, Vector<float>.Zero); }
                Store(a0, dst, di); Store(a1, dst, di + lanes);
                Store(b0, dst, di + ocount); Store(b1, dst, di + ocount + lanes);
            }
            for (; p < pixels; p++)
                for (int oc = 0; oc < ocount; oc += lanes)
                {
                    var sum = Load(biases, oc);
                    for (int ic = 0; ic < icount; ic++) sum = MultiplyAdd(new Vector<float>(Read(src, p * icount + ic)), Load(weights, ic * ocount + oc), sum);
                    Store(relu ? Vector.Max(sum, Vector<float>.Zero) : sum, dst, p * ocount + oc);
                }
            return;
        }
        for (int p = 0; p < pixels; p++)
        {
            int si = p * icount, di = p * ocount, oc = 0;
            for (; oc + 8 * lanes <= ocount; oc += 8 * lanes)
            {
                var a0 = Load(biases, oc + 0 * lanes);
                var a1 = Load(biases, oc + 1 * lanes);
                var a2 = Load(biases, oc + 2 * lanes);
                var a3 = Load(biases, oc + 3 * lanes);
                var a4 = Load(biases, oc + 4 * lanes);
                var a5 = Load(biases, oc + 5 * lanes);
                var a6 = Load(biases, oc + 6 * lanes);
                var a7 = Load(biases, oc + 7 * lanes);
                for (int ic = 0, wi = oc; ic < icount; ic++, wi += ocount)
                {
                    var value = new Vector<float>(Read(src, si + ic));
                    a0 = MultiplyAdd(value, Load(weights, wi + 0 * lanes), a0);
                    a1 = MultiplyAdd(value, Load(weights, wi + 1 * lanes), a1);
                    a2 = MultiplyAdd(value, Load(weights, wi + 2 * lanes), a2);
                    a3 = MultiplyAdd(value, Load(weights, wi + 3 * lanes), a3);
                    a4 = MultiplyAdd(value, Load(weights, wi + 4 * lanes), a4);
                    a5 = MultiplyAdd(value, Load(weights, wi + 5 * lanes), a5);
                    a6 = MultiplyAdd(value, Load(weights, wi + 6 * lanes), a6);
                    a7 = MultiplyAdd(value, Load(weights, wi + 7 * lanes), a7);
                }
                Store(relu ? Vector.Max(a0, Vector<float>.Zero) : a0, dst, di + oc + 0 * lanes);
                Store(relu ? Vector.Max(a1, Vector<float>.Zero) : a1, dst, di + oc + 1 * lanes);
                Store(relu ? Vector.Max(a2, Vector<float>.Zero) : a2, dst, di + oc + 2 * lanes);
                Store(relu ? Vector.Max(a3, Vector<float>.Zero) : a3, dst, di + oc + 3 * lanes);
                Store(relu ? Vector.Max(a4, Vector<float>.Zero) : a4, dst, di + oc + 4 * lanes);
                Store(relu ? Vector.Max(a5, Vector<float>.Zero) : a5, dst, di + oc + 5 * lanes);
                Store(relu ? Vector.Max(a6, Vector<float>.Zero) : a6, dst, di + oc + 6 * lanes);
                Store(relu ? Vector.Max(a7, Vector<float>.Zero) : a7, dst, di + oc + 7 * lanes);
            }
            for (; oc + 4 * lanes <= ocount; oc += 4 * lanes)
            {
                var a0 = Load(biases, oc + 0 * lanes);
                var a1 = Load(biases, oc + 1 * lanes);
                var a2 = Load(biases, oc + 2 * lanes);
                var a3 = Load(biases, oc + 3 * lanes);
                for (int ic = 0, wi = oc; ic < icount; ic++, wi += ocount)
                {
                    var value = new Vector<float>(Read(src, si + ic));
                    a0 = MultiplyAdd(value, Load(weights, wi + 0 * lanes), a0);
                    a1 = MultiplyAdd(value, Load(weights, wi + 1 * lanes), a1);
                    a2 = MultiplyAdd(value, Load(weights, wi + 2 * lanes), a2);
                    a3 = MultiplyAdd(value, Load(weights, wi + 3 * lanes), a3);
                }
                Store(relu ? Vector.Max(a0, Vector<float>.Zero) : a0, dst, di + oc + 0 * lanes);
                Store(relu ? Vector.Max(a1, Vector<float>.Zero) : a1, dst, di + oc + 1 * lanes);
                Store(relu ? Vector.Max(a2, Vector<float>.Zero) : a2, dst, di + oc + 2 * lanes);
                Store(relu ? Vector.Max(a3, Vector<float>.Zero) : a3, dst, di + oc + 3 * lanes);
            }
            for (; oc + 2 * lanes <= ocount; oc += 2 * lanes)
            {
                var a0 = Load(biases, oc + 0 * lanes);
                var a1 = Load(biases, oc + 1 * lanes);
                for (int ic = 0, wi = oc; ic < icount; ic++, wi += ocount)
                {
                    var value = new Vector<float>(Read(src, si + ic));
                    a0 = MultiplyAdd(value, Load(weights, wi + 0 * lanes), a0);
                    a1 = MultiplyAdd(value, Load(weights, wi + 1 * lanes), a1);
                }
                Store(relu ? Vector.Max(a0, Vector<float>.Zero) : a0, dst, di + oc + 0 * lanes);
                Store(relu ? Vector.Max(a1, Vector<float>.Zero) : a1, dst, di + oc + 1 * lanes);
            }
            for (; oc + 1 * lanes <= ocount; oc += 1 * lanes)
            {
                var a0 = Load(biases, oc + 0 * lanes);
                for (int ic = 0, wi = oc; ic < icount; ic++, wi += ocount)
                {
                    var value = new Vector<float>(Read(src, si + ic));
                    a0 = MultiplyAdd(value, Load(weights, wi + 0 * lanes), a0);
                }
                Store(relu ? Vector.Max(a0, Vector<float>.Zero) : a0, dst, di + oc + 0 * lanes);
            }
        }
    }

    private static void PointwiseDot(Tensor input, Filter filter, bool relu, Tensor output)
    {
        int icount = input.Channels, ocount = output.Channels, lanes = Vector<float>.Count, pixels = input.Rows * input.Cols;
        float[] src = input.Data, dst = output.Data, weights = filter.Weights, biases = filter.Biases;
        for (int p = 0; p < pixels; p++)
            for (int oc = 0; oc < ocount; oc++)
            {
                int si = p * icount, wi = oc * icount, ic = 0;
                float sum = 0;
                if (SimdEnabled)
                {
                    var accum = Vector<float>.Zero;
                    for (; ic + lanes <= icount; ic += lanes) accum = MultiplyAdd(Load(src, si + ic), Load(weights, wi + ic), accum);
                    sum = Vector.Sum(accum);
                }
                for (; ic < icount; ic++) sum += src[si + ic] * weights[wi + ic];
                sum += biases[oc];
                dst[p * ocount + oc] = relu ? MathF.Max(sum, 0) : sum;
            }
    }

    private static void Depthwise(Tensor input, Filter filter, bool relu, Tensor output)
    {
        int rows = input.Rows, cols = input.Cols, channels = input.Channels, lanes = Vector<float>.Count;
        float[] src = input.Data, dst = output.Data, weights = filter.Weights, biases = filter.Biases;
        int rowStep = cols * channels;
        for (int y = 0; y < rows; y++)
            for (int x = 0; x < cols; x++)
            {
                int index = (y * cols + x) * channels, c = 0;
                bool interior = y > 0 && y + 1 < rows && x > 0 && x + 1 < cols;
                if (SimdEnabled && interior && x + 2 < cols && channels % lanes == 0)
                {
                    // Adjacent output pixels share each weight and the two overlapping input
                    // vectors in each row, while keeping two independent FMA chains.
                    for (; c < channels; c += lanes)
                    {
                        int p = index - rowStep - channels + c;
                        var a = Load(biases, c); var b = a;
                        {
                            int r = p + 0 * rowStep;
                            var w0 = Load(weights, 0 * channels + c);
                            var w1 = Load(weights, 1 * channels + c);
                            var w2 = Load(weights, 2 * channels + c);
                            var v0 = Load(src, r); var v1 = Load(src, r + channels);
                            var v2 = Load(src, r + 2 * channels); var v3 = Load(src, r + 3 * channels);
                            a = MultiplyAdd(v0, w0, a); b = MultiplyAdd(v1, w0, b);
                            a = MultiplyAdd(v1, w1, a); b = MultiplyAdd(v2, w1, b);
                            a = MultiplyAdd(v2, w2, a); b = MultiplyAdd(v3, w2, b);
                        }
                        {
                            int r = p + 1 * rowStep;
                            var w0 = Load(weights, 3 * channels + c);
                            var w1 = Load(weights, 4 * channels + c);
                            var w2 = Load(weights, 5 * channels + c);
                            var v0 = Load(src, r); var v1 = Load(src, r + channels);
                            var v2 = Load(src, r + 2 * channels); var v3 = Load(src, r + 3 * channels);
                            a = MultiplyAdd(v0, w0, a); b = MultiplyAdd(v1, w0, b);
                            a = MultiplyAdd(v1, w1, a); b = MultiplyAdd(v2, w1, b);
                            a = MultiplyAdd(v2, w2, a); b = MultiplyAdd(v3, w2, b);
                        }
                        {
                            int r = p + 2 * rowStep;
                            var w0 = Load(weights, 6 * channels + c);
                            var w1 = Load(weights, 7 * channels + c);
                            var w2 = Load(weights, 8 * channels + c);
                            var v0 = Load(src, r); var v1 = Load(src, r + channels);
                            var v2 = Load(src, r + 2 * channels); var v3 = Load(src, r + 3 * channels);
                            a = MultiplyAdd(v0, w0, a); b = MultiplyAdd(v1, w0, b);
                            a = MultiplyAdd(v1, w1, a); b = MultiplyAdd(v2, w1, b);
                            a = MultiplyAdd(v2, w2, a); b = MultiplyAdd(v3, w2, b);
                        }
                        Store(relu ? Vector.Max(a, Vector<float>.Zero) : a, dst, index + c);
                        Store(relu ? Vector.Max(b, Vector<float>.Zero) : b, dst, index + channels + c);
                    }
                    x++;
                    continue;
                }
                if (SimdEnabled && interior)
                    for (; c + lanes <= channels; c += lanes)
                    {
                        int p = index - rowStep - channels + c;
                        var sum = MultiplyAdd(Load(src, p), Load(weights, c), Load(biases, c));
                        sum = MultiplyAdd(Load(src, p + channels), Load(weights, channels + c), sum);
                        sum = MultiplyAdd(Load(src, p + 2 * channels), Load(weights, 2 * channels + c), sum);
                        p += rowStep;
                        sum = MultiplyAdd(Load(src, p), Load(weights, 3 * channels + c), sum);
                        sum = MultiplyAdd(Load(src, p + channels), Load(weights, 4 * channels + c), sum);
                        sum = MultiplyAdd(Load(src, p + 2 * channels), Load(weights, 5 * channels + c), sum);
                        p += rowStep;
                        sum = MultiplyAdd(Load(src, p), Load(weights, 6 * channels + c), sum);
                        sum = MultiplyAdd(Load(src, p + channels), Load(weights, 7 * channels + c), sum);
                        sum = MultiplyAdd(Load(src, p + 2 * channels), Load(weights, 8 * channels + c), sum);
                        Store(relu ? Vector.Max(sum, Vector<float>.Zero) : sum, dst, index + c);
                    }
                else if (SimdEnabled)
                    for (; c + lanes <= channels; c += lanes)
                    {
                        var sum = Load(biases, c);
                        for (int dy = -1; dy <= 1; dy++)
                            if ((uint)(y + dy) < (uint)rows)
                                for (int dx = -1; dx <= 1; dx++)
                                    if ((uint)(x + dx) < (uint)cols)
                                        sum = MultiplyAdd(Load(src, index + dy * rowStep + dx * channels + c), Load(weights, ((dy + 1) * 3 + dx + 1) * channels + c), sum);
                        Store(relu ? Vector.Max(sum, Vector<float>.Zero) : sum, dst, index + c);
                    }
                for (; c < channels; c++)
                {
                    float sum = biases[c];
                    for (int dy = -1; dy <= 1; dy++)
                        if ((uint)(y + dy) < (uint)rows)
                            for (int dx = -1; dx <= 1; dx++)
                                if ((uint)(x + dx) < (uint)cols)
                                    sum += src[index + dy * rowStep + dx * channels + c] * weights[((dy + 1) * 3 + dx + 1) * channels + c];
                    dst[index + c] = relu ? MathF.Max(sum, 0) : sum;
                }
            }
    }
}
