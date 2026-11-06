using FaceDetection;
using System.Numerics;

namespace FaceDetection.Tests;

internal static class KernelTests
{
    internal static void Run()
    {
        if (FaceDetector.IsSimdEnabled != Kernels.SimdEnabled ||
            FaceDetector.SimdFloatLanes != (Kernels.SimdEnabled ? Vector<float>.Count : 1))
            throw new Exception("Public SIMD diagnostics disagree with production kernels");
        #if FD_SCALAR
        if (Kernels.SimdEnabled) throw new Exception("Build-time scalar switch failed");
        #endif
        Console.WriteLine($"SIMD diagnostics: platform_hardware={Vector.IsHardwareAccelerated}, kernel_simd={Kernels.SimdEnabled}, public_simd={FaceDetector.IsSimdEnabled}, public_lanes={FaceDetector.SimdFloatLanes}");
        
        const float canary = 123456.25f;
        var random = new Random(173);
        int convolutionCases = 0, poolCases = 0, upsampleCases = 0, imageCases = 0, fusedCases = 0, fusedFallbacks = 0;
        float maxConvolutionError = 0, maxFusedError = 0;
        int[] channelsList = [1, 4, 10, 16, 32, 64];
        (int Rows, int Cols)[] shapes = [(1,1), (1,7), (7,1), (2,2), (3,3), (5,7), (7,5), (16,17), (17,16)];
        var reused = new Tensor();
        
        Tensor Input(int rows, int cols, int channels)
        {
            var tensor = new Tensor(); tensor.Resize(rows, cols, channels);
            for (int i = 0; i < tensor.Data.Length; i++) tensor.Data[i] = (float)(random.NextDouble() * 4 - 2);
            return tensor;
        }
        Filter FilterFor(int input, int output, bool depthwise, float scale = .125f)
        {
            var weights = new float[input * (depthwise ? 9 : output)];
            var biases = new float[output];
            for (int i = 0; i < weights.Length; i++) weights[i] = (float)(random.NextDouble() * 2 - 1) * scale;
            for (int i = 0; i < biases.Length; i++) biases[i] = (float)(random.NextDouble() * 2 - 1);
            return new Filter(input, output, depthwise, weights, biases);
        }
        void Prepare(Tensor target, int logical)
        {
            // Never shrink: all retained spare capacity, not just a fixed suffix, becomes a canary.
            if (target.Data.Length < logical + 19) target.Data = new float[logical + 19];
            Array.Fill(target.Data, canary);
        }
        float Verify(Tensor actual, float[] expected, int rows, int cols, int channels, string context, float tolerance = 1e-5f)
        {
            if (actual.Rows != rows || actual.Cols != cols || actual.Channels != channels)
                throw new Exception($"{context}: shape changed unexpectedly");
            float maximum = 0;
            for (int i = 0; i < expected.Length; i++)
            {
                float error = MathF.Abs(actual.Data[i] - expected[i]);
                if (!float.IsFinite(actual.Data[i]) || error > (tolerance == 0 ? 0 : tolerance + 1e-6f * MathF.Abs(expected[i])))
                    throw new Exception($"{context}: element {i}, expected {expected[i]:G9}, actual {actual.Data[i]:G9}, error {error:G9}");
                maximum = MathF.Max(maximum, error);
            }
            for (int i = expected.Length; i < actual.Data.Length; i++)
                if (actual.Data[i] != canary) throw new Exception($"{context}: output tail overwritten at {i}");
            return maximum;
        }
        float[] ReferenceConvolution(Tensor input, Filter filter, bool relu)
        {
            // Scalar double accumulation intentionally avoids production SIMD/FMA/reduction code.
            var result = new float[input.Rows * input.Cols * filter.OutputChannels];
            for (int y = 0; y < input.Rows; y++)
                for (int x = 0; x < input.Cols; x++)
                    for (int oc = 0; oc < filter.OutputChannels; oc++)
                    {
                        double sum = filter.Biases[oc];
                        if (filter.Depthwise)
                        {
                            for (int ky = 0; ky < 3; ky++)
                                for (int kx = 0; kx < 3; kx++)
                                {
                                    int iy = y + ky - 1, ix = x + kx - 1;
                                    if (iy < 0 || iy >= input.Rows || ix < 0 || ix >= input.Cols) continue;
                                    sum += (double)input.Data[(iy * input.Cols + ix) * input.Channels + oc] *
                                        filter.Weights[(ky * 3 + kx) * input.Channels + oc];
                                }
                        }
                        else
                            for (int ic = 0; ic < input.Channels; ic++)
                                sum += (double)input.Data[(y * input.Cols + x) * input.Channels + ic] * filter.Weights[oc * input.Channels + ic];
                        result[(y * input.Cols + x) * filter.OutputChannels + oc] = (float)(relu ? Math.Max(sum, 0) : sum);
                    }
            return result;
        }
        foreach (var shape in shapes)
        foreach (int channels in channelsList)
        foreach (bool relu in new[] { false, true })
        {
            var input = Input(shape.Rows, shape.Cols, channels);
            var unchanged = (float[])input.Data.Clone();
            foreach (int outputChannels in channelsList)
            {
                var filter = FilterFor(channels, outputChannels, false);
                var expected = ReferenceConvolution(input, filter, relu);
                Prepare(reused, expected.Length);
                Kernels.Convolution(input, filter, relu, reused);
                maxConvolutionError = Math.Max(maxConvolutionError,
                    Verify(reused, expected, shape.Rows, shape.Cols, outputChannels, $"pointwise {shape}/{channels}->{outputChannels}/{relu}"));
                convolutionCases++;
            }
            var depthwise = FilterFor(channels, channels, true);
            var depthwiseExpected = ReferenceConvolution(input, depthwise, relu);
            Prepare(reused, depthwiseExpected.Length);
            Kernels.Convolution(input, depthwise, relu, reused);
            maxConvolutionError = Math.Max(maxConvolutionError,
                Verify(reused, depthwiseExpected, shape.Rows, shape.Cols, channels, $"depthwise {shape}/{channels}/{relu}"));
            convolutionCases++;
            if (!input.Data.SequenceEqual(unchanged)) throw new Exception("Convolution mutated input");
        }
        foreach (var (rows, cols) in new[] { (2,2), (2,14), (14,2), (6,10), (16,18) })
        foreach (int channels in channelsList)
        {
            var input = Input(rows, cols, channels);
            var unchanged = (float[])input.Data.Clone();
            var expected = new float[rows / 2 * (cols / 2) * channels];
            for (int y = 0; y < rows / 2; y++)
                for (int x = 0; x < cols / 2; x++)
                    for (int c = 0; c < channels; c++)
                    {
                        float value = float.NegativeInfinity;
                        for (int dy = 0; dy < 2; dy++)
                            for (int dx = 0; dx < 2; dx++)
                                value = Math.Max(value, input.Data[((y * 2 + dy) * cols + x * 2 + dx) * channels + c]);
                        expected[(y * (cols / 2) + x) * channels + c] = value;
                    }
            Prepare(reused, expected.Length); Kernels.MaxPool(input, reused);
            Verify(reused, expected, rows / 2, cols / 2, channels, "pool", 0);
            if (!input.Data.SequenceEqual(unchanged)) throw new Exception("Pool mutated input");
            poolCases++;
        }
        foreach (var shape in shapes)
        foreach (int channels in channelsList)
        {
            var input = Input(shape.Rows, shape.Cols, channels);
            var lateral = Input(shape.Rows * 2, shape.Cols * 2, channels);
            var unchanged = (float[])input.Data.Clone(); var lateralUnchanged = (float[])lateral.Data.Clone();
            var expected = new float[lateral.Data.Length];
            for (int y = 0; y < lateral.Rows; y++)
                for (int x = 0; x < lateral.Cols; x++)
                    for (int c = 0; c < channels; c++)
                    {
                        int target = (y * lateral.Cols + x) * channels + c;
                        expected[target] = lateral.Data[target] + input.Data[((y / 2) * input.Cols + x / 2) * channels + c];
                    }
            Prepare(reused, expected.Length); Kernels.UpsampleAdd(input, lateral, reused);
            Verify(reused, expected, lateral.Rows, lateral.Cols, channels, "upsample", 0);
            if (!input.Data.SequenceEqual(unchanged) || !lateral.Data.SequenceEqual(lateralUnchanged)) throw new Exception("Upsample mutated input");
            upsampleCases++;
        }
        
        var patch = new Tensor(); var legacy = new Tensor(); var fused = new Tensor();
        var syntheticFirst = FilterFor(32, 16, false, .01f);
        // In particular, padded input channels 27..31 have nonzero weights and must contribute zero.
        if (Enumerable.Range(0,16).Any(oc => Enumerable.Range(27,5).Any(ic => syntheticFirst.Weights[oc * 32 + ic] == 0)))
            throw new Exception("Synthetic padded weight unexpectedly zero");
        foreach (var (width, height) in new[] { (1,1), (1,33), (33,1), (3,3), (5,7), (16,17), (31,33), (33,31), (63,65) })
        foreach (var format in Enum.GetValues<PixelFormat>())
        foreach (int padding in new[] { 0, 7 })
        {
            int bytes = format == PixelFormat.Rgba32 ? 4 : 3;
            int stride = width * bytes + padding;
            byte[] pixels = new byte[(height - 1) * stride + width * bytes];
            random.NextBytes(pixels); var unchanged = (byte[])pixels.Clone();
            int rows = (height + 31) / 32 * 16, cols = (width + 31) / 32 * 16;
            var expected = new float[rows * cols * 32];
            for (int y = 0; y < rows; y++)
                for (int x = 0; x < cols; x++)
                    for (int channel = 0; channel < 27; channel++)
                    {
                        int iy = y * 2 + channel / 9 - 1, ix = x * 2 + channel / 3 % 3 - 1;
                        if (iy < 0 || iy >= height || ix < 0 || ix >= width) continue;
                        int color = channel % 3;
                        int sourceColor = format == PixelFormat.Bgr24 ? color : 2 - color;
                        expected[(y * cols + x) * 32 + channel] = pixels[iy * stride + ix * bytes + sourceColor];
                    }
            Prepare(patch, expected.Length); Kernels.ImageToTensor(pixels, width, height, stride, format, patch);
            Verify(patch, expected, rows, cols, 32, $"image {width}x{height}/{format}/{padding}", 0);
            imageCases++;
            foreach (var filter in new[] { syntheticFirst, Model.Instance.Filters[0] })
            {
                Prepare(legacy, rows * cols * 16); Kernels.Convolution(patch, filter, true, legacy);
                // Check logical output and canaries against a separate scalar convolution as well.
                var scalarExpected = ReferenceConvolution(patch, filter, true);
                Verify(legacy, scalarExpected, rows, cols, 16, "legacy first-layer scalar reference", 1e-5f);
                Prepare(fused, rows * cols * 16);
                bool ran = Kernels.ImageConvolution(pixels, width, height, stride, format, filter, fused);
                if (ran)
                {
                    var legacyLogical = legacy.Data.AsSpan(0, rows * cols * 16).ToArray();
                    maxFusedError = Math.Max(maxFusedError, Verify(fused, legacyLogical, rows, cols, 16, $"fused {width}x{height}/{format}/{padding}"));
                    fusedCases++;
                }
                else
                {
                    if (fused.Data.Any(value => value != canary)) throw new Exception("Unavailable fused path wrote output");
                    fusedFallbacks++;
                }
            }
            if (!pixels.SequenceEqual(unchanged)) throw new Exception("Image kernel mutated pixels or row padding");
        }
        
        // Pooling uses the retained Kernels.MaxPool with one channel, which always takes
        // its scalar tail, even in SIMD builds. ReLU tests call ScalarKernels directly.
        // Vector.Max is deliberately not used as their reference.
        float negativeZero = BitConverter.Int32BitsToSingle(unchecked((int)0x80000000));
        float[] specials = [float.NegativeInfinity, -2, negativeZero, 0, 2, float.PositiveInfinity,
            BitConverter.Int32BitsToSingle(0x7fc12345), BitConverter.Int32BitsToSingle(unchecked((int)0xffc54321))];
        int specialPoolCases = 0, specialReluCases = 0;
        void VerifyBits(float actual, float expected, string context)
        {
            int a = BitConverter.SingleToInt32Bits(actual), b = BitConverter.SingleToInt32Bits(expected);
            if (a != b) throw new Exception($"{context}: bits {a:X8}, expected {b:X8}");
        }
        void VerifyMaximumSemantics(float actual, float expected, string context)
        {
            // MathF.Max guarantees NaN propagation, not which NaN payload wins when both
            // operands are NaN. Hardware/JIT lowering can select different valid payloads.
            if (float.IsNaN(expected))
            {
                if (!float.IsNaN(actual)) throw new Exception($"{context}: expected NaN");
                return;
            }
            VerifyBits(actual, expected, context);
        }
        void VerifyTail(Tensor output, int logical, string context)
        {
            for (int i = logical; i < output.Data.Length; i++)
                if (output.Data[i] != canary) throw new Exception($"{context}: tail overwritten at {i}");
        }
        var poolInput = new Tensor(); poolInput.Resize(2, 2, 1);
        foreach (float a in specials)
        foreach (float b in specials)
        foreach (float c in specials)
        foreach (float d in specials)
        {
            poolInput.Data[0] = a; poolInput.Data[1] = b; poolInput.Data[2] = c; poolInput.Data[3] = d;
            float expected = MathF.Max(MathF.Max(a, b), MathF.Max(c, d));
            Prepare(reused, 1); reused.Resize(1, 1, 1);
            Kernels.MaxPool(poolInput, reused);
            VerifyMaximumSemantics(reused.Data[0], expected, "scalar special-value pool");
            VerifyTail(reused, 1, "scalar special-value pool");
            VerifyBits(poolInput.Data[0], a, "pool source a"); VerifyBits(poolInput.Data[1], b, "pool source b");
            VerifyBits(poolInput.Data[2], c, "pool source c"); VerifyBits(poolInput.Data[3], d, "pool source d");
            specialPoolCases++;
        }
        foreach (float bias in specials)
        {
            foreach (int outputs in new[] { 1, 16 })
            {
                var source = new Tensor(); source.Resize(1, 1, 1); source.Data[0] = negativeZero;
                var filter = new Filter(1, outputs, false, Enumerable.Repeat(1f, outputs).ToArray(), Enumerable.Repeat(bias, outputs).ToArray());
                float sum = 0; sum += source.Data[0]; sum += bias;
                float expected = MathF.Max(sum, 0);
                Prepare(reused, outputs); reused.Resize(1, 1, outputs);
                ScalarKernels.Pointwise(source, filter, true, reused);
                for (int c = 0; c < outputs; c++) VerifyMaximumSemantics(reused.Data[c], expected, "scalar pointwise ReLU");
                VerifyTail(reused, outputs, "scalar pointwise ReLU");
                specialReluCases++;
            }
            foreach (var (rows, cols) in new[] { (1,1), (3,3), (3,4) })
            {
                var source = new Tensor(); source.Resize(rows, cols, 1); Array.Fill(source.Data, negativeZero);
                var filter = new Filter(1, 1, true, Enumerable.Repeat(1f, 9).ToArray(), [bias]);
                Prepare(reused, rows * cols); reused.Resize(rows, cols, 1);
                ScalarKernels.Depthwise(source, filter, true, reused);
                for (int y = 0; y < rows; y++)
                    for (int x = 0; x < cols; x++)
                    {
                        float sum = bias;
                        for (int iy = Math.Max(0, y - 1); iy <= Math.Min(rows - 1, y + 1); iy++)
                            for (int ix = Math.Max(0, x - 1); ix <= Math.Min(cols - 1, x + 1); ix++) sum += negativeZero;
                        VerifyMaximumSemantics(reused.Data[y * cols + x], MathF.Max(sum, 0), "scalar depthwise ReLU");
                    }
                VerifyTail(reused, rows * cols, "scalar depthwise ReLU");
                specialReluCases++;
            }
        }
        TestRunner.Pass("Kernels/scalar special values and signed-zero semantics");
        Console.WriteLine($"PASS scalar special values: pool={specialPoolCases}, ReLU={specialReluCases}; NaN propagation, infinities, bit-exact signed zeros, canary tails.");
        Console.WriteLine($"PASS convolution={convolutionCases}, pool={poolCases}, upsample={upsampleCases}, image={imageCases}, fused={fusedCases}, fused_fallback={fusedFallbacks}; vector_lanes={Vector<float>.Count}, hardware={Vector.IsHardwareAccelerated}; max_convolution_error={maxConvolutionError:G9}, max_fused_vs_legacy_error={maxFusedError:G9}; reused-output canary tails and input immutability passed.");
        
        TestRunner.Pass("Kernels/convolution, pooling, upsampling, image conversion and fused parity");
    }
}
