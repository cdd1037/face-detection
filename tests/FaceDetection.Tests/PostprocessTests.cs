using FaceDetection;

namespace FaceDetection.Tests;

internal static class PostprocessTests
{
    internal static void Run()
    {
        
        var random = new Random(42);
        int cases = 0;
        foreach (int pixels in new[] { 0, 1, 2, 3, 4, 5, 20, 341, 1001 })
        foreach (bool tied in new[] { false, true })
        {
            var cls = new Tensor[3]; var reg = new Tensor[3]; var kps = new Tensor[3]; var obj = new Tensor[3];
            var expected = new List<(float Score, int Id)>();
            for (int scale = 0; scale < 3; scale++)
            {
                int stride = 8 << scale;
                cls[scale] = NewTensor(pixels, 1); reg[scale] = NewTensor(pixels, 4);
                kps[scale] = NewTensor(pixels, 10); obj[scale] = NewTensor(pixels, 1);
                for (int pixel = 0; pixel < pixels; pixel++)
                {
                    float logit = tied ? 0 : random.Next(-4, 5);
                    cls[scale].Data[pixel] = obj[scale].Data[pixel] = logit;
                    float sig = 1f / (1f + MathF.Exp(-logit));
                    float score = MathF.Sqrt(sig * sig);
                    int id = scale * pixels + pixel;
                    if (score >= .2f) expected.Add((score, id));
                    reg[scale].Data[4 * pixel] = 10000 * scale;
                    reg[scale].Data[4 * pixel + 2] = reg[scale].Data[4 * pixel + 3] = -4;
                    kps[scale].Data[10 * pixel] = (float)(id - pixel * stride) / stride;
                }
            }
            var post = new Postprocess();
            var actual = post.Run(cls, reg, kps, obj);
            var sorted = expected.OrderByDescending(v => v.Score).Take(1000).Take(512).ToArray();
            if (actual.Length != sorted.Length) throw new Exception($"Count mismatch {pixels}/{tied}");
            for (int i = 0; i < sorted.Length; i++)
                if (actual[i].Score != sorted[i].Score || actual[i].Landmarks.RightEye.X != sorted[i].Id)
                    throw new Exception($"Order mismatch {pixels}/{tied} at {i}");
            for (int i = 0; i < 20; i++)
            {
                GC.KeepAlive(post.Run(cls, reg, kps, obj));
                GC.KeepAlive(NewResult(actual.Length));
            }
            const int iterations = 100;
            // Keep GC-triggered allocation-context transitions outside both counter windows.
            // The largest pair allocates under 6 MiB (100 * 512-face arrays twice).
            // Failure to reserve or maintain the region fails explicitly; never accept
            // an unmeasured case or subtract unexplained allocation overhead.
            if (!GC.TryStartNoGCRegion(16 * 1024 * 1024))
                throw new InvalidOperationException("Cannot reserve the 16 MiB no-GC region required for exact allocation checks.");
            long allocatedTotal, outputTotal;
            try
            {
                allocatedTotal = MeasureRunAllocation(post, cls, reg, kps, obj, iterations);
                outputTotal = MeasureResultAllocation(actual.Length, iterations);
            }
            finally { GC.EndNoGCRegion(); }
            if (allocatedTotal != outputTotal)
                throw new Exception($"Allocation mismatch over {iterations} calls: {allocatedTotal} != {outputTotal} bytes");
            // Divide only for display: even one unexpected byte must fail the assertion.
            long allocated = allocatedTotal / iterations;
            long outputBytes = outputTotal / iterations;
            Console.WriteLine($"PASS pixels/scale={pixels}, tied={tied}, results={actual.Length}, allocation={allocated} B/call (result array={outputBytes} B)");
            post.Release();
            if (post.RetainedBytes != 0) throw new Exception("Release retained scratch payload.");
            TestRunner.Pass($"Postprocess/ordering-allocation/pixels={pixels}/tied={tied}");
            cases++;
        }
        Console.WriteLine($"Passed {cases} ordering/tie/size cases; warm Run allocation equals result array only in every case.");
        
        int boundaryCases = 0;
        
        // The same box appears in all three output scales. NMS must be global, and
        // equal scores must keep the earliest scale rather than whichever heap node wins.
        foreach (bool tied in new[] { false, true })
        {
            var heads = CreateHeads(1, 1, 1);
            SetHead(heads, 0, 0, 0, 0, 32, 32, 0, 10);
            SetHead(heads, 1, 0, 0, 0, 32, 32, tied ? 0 : 2, 20);
            SetHead(heads, 2, 0, 0, 0, 32, 32, tied ? 0 : 1, 30);
            var result = RunHeads(heads);
            Assert(result.Length == 1 && result[0].Landmarks.RightEye.X == (tied ? 10 : 20),
                $"Cross-scale suppression/tie selection failed (tied={tied}).");
            PassBoundary($"cross-scale overlapping suppression, tied={tied}");
        }
        
        // A contained 18-by-8 box in a 40-by-8 box has IoU exactly 0.45f.
        // Verify the decoded floating-point geometry itself, before relying on output
        // integer truncation, which intentionally loses subpixel differences.
        foreach (float smallWidth in new[] { 17.999f, 18f, 18.001f })
        {
            var heads = CreateHeads(2, 0, 0);
            SetHead(heads, 0, 0, 20, 0, 40, 8, 2, 100);
            SetHead(heads, 0, 1, 20, 0, smallWidth, 8, 1, 200);
            var large = DecodedBox(heads.Reg[0], 0, 8);
            var small = DecodedBox(heads.Reg[0], 1, 8);
            float intersection = (small.XMax - small.XMin) * (small.YMax - small.YMin);
            float largeArea = (large.XMax - large.XMin) * (large.YMax - large.YMin);
            float iou = intersection / (largeArea + intersection - intersection);
            Assert(smallWidth < 18 ? iou < .45f : smallWidth > 18 ? iou > .45f : iou == .45f,
                $"IoU fixture is not on its intended threshold side: {iou:R}.");
            var result = RunHeads(heads);
            int expectedCount = iou > .45f ? 1 : 2;
            Assert(result.Length == expectedCount && result[0].Landmarks.RightEye.X == 100,
                $"IoU threshold failed: overlap={iou:R}, expected {expectedCount}, got {result.Length}.");
            PassBoundary($"IoU boundary {iou:R}, survivors={result.Length}");
        }
        
        float thresholdLogit = MathF.Log(.2f / (1f - .2f));
        foreach (float logit in new[] { thresholdLogit - .0001f, thresholdLogit, thresholdLogit + .0001f })
        {
            var heads = CreateHeads(1, 0, 0);
            SetHead(heads, 0, 0, 0, 0, 8, 8, logit, 300);
            float sigmoid = 1f / (1f + MathF.Exp(-logit));
            float score = MathF.Sqrt(sigmoid * sigmoid);
            Assert(logit < thresholdLogit ? score < .2f : logit > thresholdLogit ? score > .2f : score == .2f,
                $"Confidence fixture is not on its intended threshold side: {score:R}.");
            var result = RunHeads(heads);
            Assert(result.Length == (score >= .2f ? 1 : 0),
                $"Confidence threshold failed at {score:R}.");
            if (result.Length != 0) Assert(result[0].Score == score, "Surviving score changed.");
            PassBoundary($"confidence boundary {score:R}, survivors={result.Length}");
        }
        
        // Zero-area boxes must have zero overlap, even when coincident with each other
        // or contained in a higher-scored ordinary box. -Infinity encodes exp=0 exactly.
        foreach (var size in new[] { (Width: 0f, Height: 8f), (Width: 8f, Height: 0f), (Width: 0f, Height: 0f) })
        {
            var heads = CreateHeads(3, 0, 0);
            SetHead(heads, 0, 0, 0, 0, 32, 32, 2, 400);
            SetHead(heads, 0, 1, 0, 0, size.Width, size.Height, 1, 500);
            SetHead(heads, 0, 2, 0, 0, size.Width, size.Height, 1, 600);
            var result = RunHeads(heads);
            Assert(result.Length == 3 && result[1].Landmarks.RightEye.X == 500 && result[2].Landmarks.RightEye.X == 600,
                $"Zero-area overlap failed for width={size.Width}, height={size.Height}.");
            Assert(result[1].Bounds.Width == (int)size.Width && result[1].Bounds.Height == (int)size.Height,
                "Degenerate box dimensions were changed.");
            PassBoundary($"degenerate boxes width={size.Width}, height={size.Height}, survivors={result.Length}");
        }
        Console.WriteLine($"Passed {boundaryCases} NMS/confidence/degenerate boundary cases; total={cases + boundaryCases}.");
        
        void PassBoundary(string description)
        {
            boundaryCases++;
            TestRunner.Pass($"Postprocess/boundary/{description}");
        }
        
        static void Assert(bool condition, string message)
        {
            if (!condition) throw new Exception(message);
        }
        
        static (Tensor[] Cls, Tensor[] Reg, Tensor[] Kps, Tensor[] Obj) CreateHeads(params int[] pixels)
        {
            var heads = (Cls: new Tensor[3], Reg: new Tensor[3], Kps: new Tensor[3], Obj: new Tensor[3]);
            for (int scale = 0; scale < 3; scale++)
            {
                heads.Cls[scale] = NewTensor(pixels[scale], 1);
                heads.Reg[scale] = NewTensor(pixels[scale], 4);
                heads.Kps[scale] = NewTensor(pixels[scale], 10);
                heads.Obj[scale] = NewTensor(pixels[scale], 1);
            }
            return heads;
        }
        
        static void SetHead((Tensor[] Cls, Tensor[] Reg, Tensor[] Kps, Tensor[] Obj) heads,
            int scale, int pixel, float cx, float cy, float width, float height, float logit, int id)
        {
            int stride = 8 << scale;
            heads.Cls[scale].Data[pixel] = heads.Obj[scale].Data[pixel] = logit;
            float[] reg = heads.Reg[scale].Data;
            reg[pixel * 4] = (cx - pixel * stride) / stride;
            reg[pixel * 4 + 1] = cy / stride;
            reg[pixel * 4 + 2] = MathF.Log(width / stride);
            reg[pixel * 4 + 3] = MathF.Log(height / stride);
            heads.Kps[scale].Data[pixel * 10] = (float)(id - pixel * stride) / stride;
        }
        
        static DetectedFace[] RunHeads((Tensor[] Cls, Tensor[] Reg, Tensor[] Kps, Tensor[] Obj) heads)
            => new Postprocess().Run(heads.Cls, heads.Reg, heads.Kps, heads.Obj);
        
        static (float XMin, float YMin, float XMax, float YMax) DecodedBox(Tensor reg, int pixel, int stride)
        {
            float cx = reg.Data[4 * pixel] * stride + pixel * stride;
            float cy = reg.Data[4 * pixel + 1] * stride;
            float width = MathF.Exp(reg.Data[4 * pixel + 2]) * stride;
            float height = MathF.Exp(reg.Data[4 * pixel + 3]) * stride;
            return (cx - width / 2f, cy - height / 2f, cx + width / 2f, cy + height / 2f);
        }
        
        static Tensor NewTensor(int cols, int channels)
        {
            var tensor = new Tensor();
            tensor.Resize(1, cols, channels);
            // Real Tensor storage is uninitialized; unused synthetic-head fields must be zero.
            Array.Clear(tensor.Data);
            return tensor;
        }
    }

    // Separate, eagerly optimized measurement frames prevent tiered OSR of the large
    // test driver from occurring between allocation-counter reads. Production methods
    // retain their normal compilation policy and the assertion remains byte-exact.
    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining |
        System.Runtime.CompilerServices.MethodImplOptions.AggressiveOptimization)]
    private static long MeasureRunAllocation(Postprocess post, Tensor[] cls, Tensor[] reg,
        Tensor[] kps, Tensor[] obj, int iterations)
    {
        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < iterations; i++) GC.KeepAlive(post.Run(cls, reg, kps, obj));
        return GC.GetAllocatedBytesForCurrentThread() - before;
    }

    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining |
        System.Runtime.CompilerServices.MethodImplOptions.AggressiveOptimization)]
    private static long MeasureResultAllocation(int length, int iterations)
    {
        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < iterations; i++) GC.KeepAlive(NewResult(length));
        return GC.GetAllocatedBytesForCurrentThread() - before;
    }

    // Compare against actual array allocation rather than assuming object-header sizes.
    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private static DetectedFace[] NewResult(int length)
        => length == 0 ? Array.Empty<DetectedFace>() : new DetectedFace[length];
}
