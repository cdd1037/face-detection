using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;

namespace FaceDetection;

internal static partial class Kernels
{
    internal static void MaxPool(Tensor input, Tensor output)
    {
        int rows = input.Rows / 2, cols = input.Cols / 2, channels = input.Channels, lanes = Vector<float>.Count;
        output.Resize(rows, cols, channels);
        float[] src = input.Data, dst = output.Data;
        int rowStep = input.Cols * channels;
        for (int y = 0; y < rows; y++)
            for (int x = 0; x < cols; x++)
            {
                int si = (y * 2 * input.Cols + x * 2) * channels, di = (y * cols + x) * channels, c = 0;
                if (SimdEnabled)
                    for (; c + lanes <= channels; c += lanes)
                        Store(Vector.Max(Vector.Max(Load(src, si + c), Load(src, si + channels + c)),
                            Vector.Max(Load(src, si + rowStep + c), Load(src, si + rowStep + channels + c))), dst, di + c);
                for (; c < channels; c++) dst[di + c] = MathF.Max(MathF.Max(src[si + c], src[si + channels + c]), MathF.Max(src[si + rowStep + c], src[si + rowStep + channels + c]));
            }
    }

    internal static void UpsampleAdd(Tensor input, Tensor lateral, Tensor output)
    {
        output.Resize(lateral.Rows, lateral.Cols, lateral.Channels);
        int channels = output.Channels, lanes = Vector<float>.Count;
        float[] src = input.Data, lat = lateral.Data, dst = output.Data;
        for (int y = 0; y < output.Rows; y++)
            for (int x = 0; x < output.Cols; x++)
            {
                int si = ((y / 2) * input.Cols + x / 2) * channels, di = (y * output.Cols + x) * channels, c = 0;
                if (SimdEnabled)
                    for (; c + lanes <= channels; c += lanes) Store(Load(src, si + c) + Load(lat, di + c), dst, di + c);
                for (; c < channels; c++) dst[di + c] = src[si + c] + lat[di + c];
            }
    }
}
