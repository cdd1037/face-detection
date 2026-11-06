using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace FaceDetection;

// Scalar arithmetic remains scalar: independent output accumulators hide dependency
// latency, but no vector or FMA intrinsic is used. Input channels are accumulated in
// their original order; pointwise bias is added after the reduction, as before.
internal static class ScalarKernels
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float Read(float[] array, int offset) => Unsafe.Add(ref MemoryMarshal.GetArrayDataReference(array), offset);
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Write(float[] array, int offset, float value) => Unsafe.Add(ref MemoryMarshal.GetArrayDataReference(array), offset) = value;
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float Relu(float value) => value > 0 ? value : float.IsNaN(value) ? value : 0;

    internal static void Pointwise(Tensor input, Filter filter, bool relu, Tensor output)
    {
        int channels = input.Channels, outputs = output.Channels, pixels = input.Rows * input.Cols;
        float[] source = input.Data, destination = output.Data, biases = filter.Biases;
        if (filter.Packed.Length != 0)
        {
            float[] weights = filter.Packed;
            for (int pixel = 0; pixel < pixels; pixel++)
            {
                int si = pixel * channels, di = pixel * outputs, oc = 0;
                // Eight independent scalar outputs reuse every input value. The fixed
                // model's packed layers have 16/32/64 outputs, with a tail for safety.
                for (; oc + 8 <= outputs; oc += 8)
                {
                    float a0 = 0, a1 = 0, a2 = 0, a3 = 0, a4 = 0, a5 = 0, a6 = 0, a7 = 0;
                    for (int ic = 0, wi = oc; ic < channels; ic++, wi += outputs)
                    {
                        float value = Read(source, si + ic);
                        a0 += value * Read(weights, wi);
                        a1 += value * Read(weights, wi + 1);
                        a2 += value * Read(weights, wi + 2);
                        a3 += value * Read(weights, wi + 3);
                        a4 += value * Read(weights, wi + 4);
                        a5 += value * Read(weights, wi + 5);
                        a6 += value * Read(weights, wi + 6);
                        a7 += value * Read(weights, wi + 7);
                    }
                    a0 += Read(biases, oc); a1 += Read(biases, oc + 1);
                    a2 += Read(biases, oc + 2); a3 += Read(biases, oc + 3);
                    a4 += Read(biases, oc + 4); a5 += Read(biases, oc + 5);
                    a6 += Read(biases, oc + 6); a7 += Read(biases, oc + 7);
                    Write(destination, di + oc, relu ? Relu(a0) : a0);
                    Write(destination, di + oc + 1, relu ? Relu(a1) : a1);
                    Write(destination, di + oc + 2, relu ? Relu(a2) : a2);
                    Write(destination, di + oc + 3, relu ? Relu(a3) : a3);
                    Write(destination, di + oc + 4, relu ? Relu(a4) : a4);
                    Write(destination, di + oc + 5, relu ? Relu(a5) : a5);
                    Write(destination, di + oc + 6, relu ? Relu(a6) : a6);
                    Write(destination, di + oc + 7, relu ? Relu(a7) : a7);
                }
                for (; oc < outputs; oc++)
                {
                    float sum = 0;
                    for (int ic = 0, wi = oc; ic < channels; ic++, wi += outputs)
                        sum += Read(source, si + ic) * Read(weights, wi);
                    sum += Read(biases, oc);
                    Write(destination, di + oc, relu ? Relu(sum) : sum);
                }
            }
            return;
        }
        float[] unpacked = filter.Weights;
        for (int pixel = 0; pixel < pixels; pixel++)
            for (int oc = 0; oc < outputs; oc++)
            {
                float sum = 0;
                int si = pixel * channels, wi = oc * channels;
                for (int ic = 0; ic < channels; ic++) sum += Read(source, si + ic) * Read(unpacked, wi + ic);
                sum += Read(biases, oc);
                Write(destination, pixel * outputs + oc, relu ? Relu(sum) : sum);
            }
    }
    internal static void Depthwise(Tensor input, Filter filter, bool relu, Tensor output)
    {
        int rows = input.Rows, cols = input.Cols, channels = input.Channels;
        int rowStep = cols * channels;
        float[] source = input.Data, destination = output.Data, weights = filter.Weights, biases = filter.Biases;
        for (int y = 0; y < rows; y++)
            for (int x = 0; x < cols; x++)
            {
                int index = (y * cols + x) * channels;
                bool interior = y > 0 && y + 1 < rows && x > 0 && x + 1 < cols;
                if (interior && x + 2 < cols)
                {
                    // Two scalar output pixels share weights and overlapping inputs.
                    // Each accumulator still sees the original nine row-major taps.
                    int start = index - rowStep - channels;
                    for (int c = 0; c < channels; c++)
                    {
                        float a = Read(biases, c), b = a;
                        {
                            int p = start + 0 * rowStep + c;
                            float w0 = Read(weights, 0 * channels + c);
                            float w1 = Read(weights, 1 * channels + c);
                            float w2 = Read(weights, 2 * channels + c);
                            float v0 = Read(source, p), v1 = Read(source, p + channels);
                            float v2 = Read(source, p + 2 * channels), v3 = Read(source, p + 3 * channels);
                            a += v0 * w0; b += v1 * w0;
                            a += v1 * w1; b += v2 * w1;
                            a += v2 * w2; b += v3 * w2;
                        }
                        {
                            int p = start + 1 * rowStep + c;
                            float w0 = Read(weights, 3 * channels + c);
                            float w1 = Read(weights, 4 * channels + c);
                            float w2 = Read(weights, 5 * channels + c);
                            float v0 = Read(source, p), v1 = Read(source, p + channels);
                            float v2 = Read(source, p + 2 * channels), v3 = Read(source, p + 3 * channels);
                            a += v0 * w0; b += v1 * w0;
                            a += v1 * w1; b += v2 * w1;
                            a += v2 * w2; b += v3 * w2;
                        }
                        {
                            int p = start + 2 * rowStep + c;
                            float w0 = Read(weights, 6 * channels + c);
                            float w1 = Read(weights, 7 * channels + c);
                            float w2 = Read(weights, 8 * channels + c);
                            float v0 = Read(source, p), v1 = Read(source, p + channels);
                            float v2 = Read(source, p + 2 * channels), v3 = Read(source, p + 3 * channels);
                            a += v0 * w0; b += v1 * w0;
                            a += v1 * w1; b += v2 * w1;
                            a += v2 * w2; b += v3 * w2;
                        }
                        Write(destination, index + c, relu ? Relu(a) : a);
                        Write(destination, index + channels + c, relu ? Relu(b) : b);
                    }
                    x++;
                    continue;
                }
                if (interior)
                {
                    int start = index - rowStep - channels;
                    for (int c = 0; c < channels; c++)
                    {
                        float sum = Read(biases, c);
                        sum += Read(source, start + 0 * rowStep + 0 * channels + c) * Read(weights, 0 * channels + c);
                        sum += Read(source, start + 0 * rowStep + 1 * channels + c) * Read(weights, 1 * channels + c);
                        sum += Read(source, start + 0 * rowStep + 2 * channels + c) * Read(weights, 2 * channels + c);
                        sum += Read(source, start + 1 * rowStep + 0 * channels + c) * Read(weights, 3 * channels + c);
                        sum += Read(source, start + 1 * rowStep + 1 * channels + c) * Read(weights, 4 * channels + c);
                        sum += Read(source, start + 1 * rowStep + 2 * channels + c) * Read(weights, 5 * channels + c);
                        sum += Read(source, start + 2 * rowStep + 0 * channels + c) * Read(weights, 6 * channels + c);
                        sum += Read(source, start + 2 * rowStep + 1 * channels + c) * Read(weights, 7 * channels + c);
                        sum += Read(source, start + 2 * rowStep + 2 * channels + c) * Read(weights, 8 * channels + c);
                        Write(destination, index + c, relu ? Relu(sum) : sum);
                    }
                    continue;
                }
                // Border ranges are computed once per pixel, not for every channel/tap.
                int top = Math.Max(0, y - 1), bottom = Math.Min(rows, y + 2);
                int left = Math.Max(0, x - 1), right = Math.Min(cols, x + 2);
                for (int c = 0; c < channels; c++)
                {
                    float sum = Read(biases, c);
                    for (int sy = top; sy < bottom; sy++)
                    {
                        int si = (sy * cols + left) * channels + c;
                        int wi = ((sy - y + 1) * 3 + left - x + 1) * channels + c;
                        for (int sx = left; sx < right; sx++, si += channels, wi += channels)
                            sum += Read(source, si) * Read(weights, wi);
                    }
                    Write(destination, index + c, relu ? Relu(sum) : sum);
                }
            }
    }

}
