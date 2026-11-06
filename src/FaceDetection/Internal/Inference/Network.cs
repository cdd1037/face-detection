namespace FaceDetection;

// The 53-layer YuNet/libfacedetection graph. No reflection-based graph builder, native runtime,
// external model files, or image codec is needed. Scratch has explicit, bounded ownership.
internal sealed class Network
{
    private readonly Filter[] filters;
    private Tensor input = new(), fx = new(), pointwise = new(), block = new(), output = new();
    private Tensor fb1 = new(), fb2 = new(), fb3 = new(), branch = new(), headPointwise = new();
    private readonly Tensor[] cls = CreateHeads(), reg = CreateHeads(), kps = CreateHeads(), obj = CreateHeads();
    private readonly Postprocess postprocess = new();
    internal Network(Model model) => filters = model.Filters;
    private static Tensor[] CreateHeads() => new Tensor[] { new(), new(), new() };
    internal long RetainedBytes
    {
        get
        {
            long count = (long)input.Data.Length + fx.Data.Length + pointwise.Data.Length + block.Data.Length + output.Data.Length +
                fb1.Data.Length + fb2.Data.Length + fb3.Data.Length + branch.Data.Length + headPointwise.Data.Length;
            for (int i = 0; i < 3; i++) count += cls[i].Data.Length + reg[i].Data.Length + kps[i].Data.Length + obj[i].Data.Length;
            return count * sizeof(float) + postprocess.RetainedBytes;
        }
    }
    internal DetectedFace[] Run(ReadOnlySpan<byte> pixels, int width, int height, int stride, PixelFormat format)
    {
        if (!Kernels.ImageConvolution(pixels, width, height, stride, format, filters[0], fx))
        {
            Kernels.ImageToTensor(pixels, width, height, stride, format, input);
            Conv(input, 0, true, fx);
        }
        Pair(fx, 1, true, pointwise, block); Swap(ref fx, ref block);
        Kernels.MaxPool(fx, block); Swap(ref fx, ref block);
        Unit(fx, 3, output); Swap(ref fx, ref output);
        Unit(fx, 7, output); Swap(ref fx, ref output);
        Kernels.MaxPool(fx, block); Swap(ref fx, ref block);
        Unit(fx, 11, fb1);
        Kernels.MaxPool(fb1, fx); Unit(fx, 15, fb2);
        Kernels.MaxPool(fb2, fx); Unit(fx, 19, fb3);

        Pair(fb3, 27, true, headPointwise, branch); Swap(ref fb3, ref branch);
        Head(fb3, 2);
        Kernels.UpsampleAdd(fb3, fb2, branch); Swap(ref fb2, ref branch);
        Pair(fb2, 25, true, headPointwise, branch); Swap(ref fb2, ref branch);
        Head(fb2, 1);
        Kernels.UpsampleAdd(fb2, fb1, branch); Swap(ref fb1, ref branch);
        Pair(fb1, 23, true, headPointwise, branch); Swap(ref fb1, ref branch);
        Head(fb1, 0);
        return postprocess.Run(cls, reg, kps, obj);
    }
    private void Head(Tensor feature, int level)
    {
        Pair(feature, 29 + level * 2, false, headPointwise, cls[level]);
        Pair(feature, 35 + level * 2, false, headPointwise, reg[level]);
        Pair(feature, 47 + level * 2, false, headPointwise, kps[level]);
        Pair(feature, 41 + level * 2, false, headPointwise, obj[level]);
    }
    private void Conv(Tensor source, int index, bool relu, Tensor target) => Kernels.Convolution(source, filters[index], relu, target);
    private void Pair(Tensor source, int index, bool relu, Tensor scratch, Tensor target)
    {
        Conv(source, index, false, scratch);
        Conv(scratch, index + 1, relu, target);
    }
    private void Unit(Tensor source, int index, Tensor target)
    {
        Pair(source, index, true, pointwise, block);
        Pair(block, index + 2, true, pointwise, target);
    }
    private static void Swap(ref Tensor a, ref Tensor b) => (a, b) = (b, a);
}
