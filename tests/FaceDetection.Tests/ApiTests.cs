using FaceDetection;

namespace FaceDetection.Tests;

internal static class ApiTests
{
    internal static void Run()
    {
        var rgb = File.ReadAllBytes(Path.Combine(AppContext.BaseDirectory, "astronaut.rgb"));
        const int width = 256, height = 256;
        using var detector = new FaceDetector();
        var original = (byte[])rgb.Clone();
        // Construct all eight sessions before any model access. Explicit threads and a
        // shared start gate avoid Parallel.For silently serializing on a pinned CPU.
        var firstDetectors = Enumerable.Range(0, 8).Select(_ => new FaceDetector()).ToArray();
        try
        {
            ConcurrentWave("independent-first-use", firstDetectors.Length, i =>
                Check(firstDetectors[i].Detect(rgb, width, height, width * 3, PixelFormat.Rgb24).Length > 0,
                    "Concurrent independent first detection failed"));
        }
        finally { foreach (FaceDetector first in firstDetectors) first.Dispose(); }
        TestRunner.Pass("API/independent first-use concurrency");
        var faces = detector.Detect(rgb, width, height, width * 3, PixelFormat.Rgb24);
        Check(faces.Length is > 0 and <= FaceDetector.MaximumFaceCount, "Expected astronaut face");
        var face = faces.OrderByDescending(f => f.Score).First();
        Check(face.Score is > 0.8f and <= 1f, "Invalid score");
        Check(face.Bounds.X is > 70 and < 115 && face.Bounds.Y is > 20 and < 55 &&
              face.Bounds.Width is > 35 and < 75 && face.Bounds.Height is > 35 and < 80, "Unexpected face box");
        Check(face.Landmarks.RightEye.X < face.Landmarks.LeftEye.X &&
              face.Landmarks.Nose.Y > face.Landmarks.RightEye.Y &&
              face.Landmarks.RightMouthCorner.Y > face.Landmarks.Nose.Y, "Invalid landmark order");
        Check(rgb.SequenceEqual(original), "Input was mutated");
        
        TestRunner.Pass("API/face geometry, landmarks and input immutability");
        // Keep RGB fixed while changing only alpha: fully transparent pixels still
        // contain ordinary, unpremultiplied RGB and must match the real RGB fixture.
        var packedRgba = new byte[width * height * 4];
        for (int pixel = 0; pixel < width * height; pixel++)
            Array.Copy(rgb, pixel * 3, packedRgba, pixel * 4, 3);
        foreach (string alphaMode in new[] { "transparent", "opaque", "varying" })
        {
            for (int pixel = 0; pixel < width * height; pixel++)
                packedRgba[pixel * 4 + 3] = alphaMode switch
                {
                    "transparent" => (byte)0,
                    "opaque" => (byte)255,
                    _ => (byte)(pixel % 256)
                };
            var rgbaBefore = (byte[])packedRgba.Clone();
            Check(faces.SequenceEqual(detector.Detect(packedRgba, width, height, width * 4, PixelFormat.Rgba32)),
                $"Packed RGB/RGBA mismatch with {alphaMode} alpha");
            Check(packedRgba.SequenceEqual(rgbaBefore), $"RGBA input mutated with {alphaMode} alpha");
        }
        Check(rgb.SequenceEqual(original), "RGB fixture mutated during alpha tests");
        TestRunner.Pass("API/packed RGB and RGBA equivalence with ignored alpha");
        var bgr = (byte[])rgb.Clone();
        for (int i = 0; i < bgr.Length; i += 3) (bgr[i], bgr[i + 2]) = (bgr[i + 2], bgr[i]);
        Check(faces.SequenceEqual(detector.Detect(bgr, width, height, width * 3)), "RGB/BGR mismatch");
        const int stride = width * 3 + 13;
        var padded = new byte[(height - 1) * stride + width * 3];
        Array.Fill(padded, (byte)123);
        for (int y = 0; y < height; y++) Array.Copy(bgr, y * width * 3, padded, y * stride, width * 3);
        Check(faces.SequenceEqual(detector.Detect(padded, width, height, stride)), "Padded rows mismatch");
        // Alternate odd, non-square resolutions to exercise input tensor edge handling.
        // RGB rows include padding except the final row, whose last pixel ends the buffer.
        foreach (var (testWidth, testHeight) in new[] { (257, 193), (129, 161) })
        {
            int testStride = testWidth * 3 + 11;
            var packedBgr = new byte[testWidth * testHeight * 3];
            var paddedRgb = new byte[(testHeight - 1) * testStride + testWidth * 3];
            Array.Fill(paddedRgb, (byte)173);
            for (int y = 0; y < testHeight; y++)
            {
                for (int x = 0; x < testWidth; x++)
                {
                    int source = ((y * height / testHeight) * width + x * width / testWidth) * 3;
                    int packedOffset = (y * testWidth + x) * 3;
                    int paddedOffset = y * testStride + x * 3;
                    for (int channel = 0; channel < 3; channel++)
                    {
                        packedBgr[packedOffset + channel] = rgb[source + 2 - channel];
                        paddedRgb[paddedOffset + channel] = rgb[source + channel];
                    }
                }
            }
            int rgbaStride = testWidth * 4 + 7;
            var paddedRgba = new byte[(testHeight - 1) * rgbaStride + testWidth * 4];
            Array.Fill(paddedRgba, (byte)91);
            for (int y = 0; y < testHeight; y++)
            {
                for (int x = 0; x < testWidth; x++)
                {
                    int rgbOffset = y * testStride + x * 3;
                    int rgbaOffset = y * rgbaStride + x * 4;
                    Array.Copy(paddedRgb, rgbOffset, paddedRgba, rgbaOffset, 3);
                    // Include transparent and partially transparent pixels. Alpha must not
                    // alter RGB values or trigger implicit unpremultiplication.
                    paddedRgba[rgbaOffset + 3] = (byte)((x + y) % 256);
                }
            }
            var expectedRgba = (byte[])paddedRgba.Clone();
            var expectedBgr = (byte[])packedBgr.Clone();
            var expectedRgb = (byte[])paddedRgb.Clone();
            var shapeFaces = detector.Detect(packedBgr, testWidth, testHeight, testWidth * 3);
            Check(shapeFaces.Length > 0, $"Expected resized astronaut face at {testWidth}x{testHeight}");
            Check(shapeFaces.SequenceEqual(detector.Detect(paddedRgb, testWidth, testHeight, testStride, PixelFormat.Rgb24)),
                $"Padded RGB/BGR mismatch at {testWidth}x{testHeight}");
            Check(shapeFaces.SequenceEqual(detector.Detect(paddedRgba, testWidth, testHeight, rgbaStride, PixelFormat.Rgba32)),
                $"Padded RGBA/BGR mismatch at {testWidth}x{testHeight}");
            Check(packedBgr.SequenceEqual(expectedBgr) && paddedRgb.SequenceEqual(expectedRgb) && paddedRgba.SequenceEqual(expectedRgba),
                $"Input or padding mutated at {testWidth}x{testHeight}");
        }
        Check(faces.SequenceEqual(detector.Detect(rgb, width, height, width * 3, PixelFormat.Rgb24)),
            "Returning to the original resolution changed results");
        Check(rgb.SequenceEqual(original), "Original RGB input was mutated");
        TestRunner.Pass("API/pixel formats, padded strides and changing dimensions");
        Check(detector.Detect(new byte[32 * 32 * 3], 32, 32, 96).Length == 0, "Blank image has faces");
        var tiny = new byte[3];
        var empty = detector.Detect(tiny, 1, 1, 3);
        Check(empty.Length == 0, "Tiny image failed");
        Check(detector.Detect(new byte[] { 0, 0, 0, 255 }, 1, 1, 4, PixelFormat.Rgba32).Length == 0,
            "Tiny RGBA image failed");
        Check(ReferenceEquals(empty, detector.Detect(tiny, 1, 1, 3)), "Empty results should reuse an array");
        // Warm inference and JIT before measuring managed allocations on this thread.
        for (int i = 0; i < 32; i++) detector.Detect(tiny, 1, 1, 3);
        long allocatedBefore = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < 128; i++) detector.Detect(tiny, 1, 1, 3);
        Check(GC.GetAllocatedBytesForCurrentThread() == allocatedBefore, "Empty detection allocated managed memory");
        TestRunner.Pass("API/empty results and zero-allocation warm inference");
        Reject<ArgumentOutOfRangeException>(() => detector.Detect(rgb, 0, height, 768));
        Reject<ArgumentOutOfRangeException>(() => detector.Detect(rgb, int.MaxValue, height, int.MaxValue));
        Reject<ArgumentOutOfRangeException>(() => detector.Detect(rgb, 4096, 4096, 12288));
        Reject<ArgumentOutOfRangeException>(() => detector.Detect(rgb, width, height, -768));
        Reject<ArgumentOutOfRangeException>(() => detector.Detect(rgb, width, height, 767));
        Reject<ArgumentOutOfRangeException>(() => detector.Detect(rgb, width, height, 768, (PixelFormat)42));
        Reject<ArgumentException>(() => detector.Detect(new byte[1], width, height, 768));
        Reject<ArgumentException>(() => detector.Detect(rgb, width, height, int.MaxValue));
        Reject<ArgumentOutOfRangeException>(() => detector.Detect(rgb, width, height, width * 4 - 1, PixelFormat.Rgba32));
        Reject<ArgumentException>(() => detector.Detect(new byte[width * height * 4 - 1], width, height, width * 4, PixelFormat.Rgba32));
        Reject<ArgumentException>(() => detector.Detect(new byte[3], 1, 1, 4, PixelFormat.Rgba32));
        Reject<ArgumentException>(() => detector.Detect(rgb, width, height, int.MaxValue, PixelFormat.Rgba32));
        // Exercise identical layout boundaries explicitly for every public format.
        foreach (PixelFormat format in new[] { PixelFormat.Bgr24, PixelFormat.Rgb24, PixelFormat.Rgba32 })
        {
            const int testWidth = 3, testHeight = 2;
            int rowBytes = testWidth * (format == PixelFormat.Rgba32 ? 4 : 3);
            int paddedStride = rowBytes + 5;
            var packedBuffer = new byte[testHeight * rowBytes];
            Reject<ArgumentOutOfRangeException>(() => detector.Detect(packedBuffer, testWidth, testHeight, 0, format));
            Reject<ArgumentOutOfRangeException>(() => detector.Detect(packedBuffer, testWidth, testHeight, -rowBytes, format));
            Reject<ArgumentOutOfRangeException>(() => detector.Detect(packedBuffer, testWidth, testHeight, rowBytes - 1, format));
            Reject<ArgumentException>(() => detector.Detect(new byte[testHeight * rowBytes - 1],
                testWidth, testHeight, rowBytes, format));
            // The final row needs its pixels, but not its trailing padding.
            Reject<ArgumentException>(() => detector.Detect(new byte[paddedStride + rowBytes - 1],
                testWidth, testHeight, paddedStride, format));
            Reject<ArgumentException>(() => detector.Detect(packedBuffer, testWidth, testHeight, int.MaxValue, format));
        }
        TestRunner.Pass("API/argument validation and buffer bounds");
        Check(detector.RetainedWorkspaceBytes > 0, "Workspace should be retained after inference");
        detector.ReleaseWorkspace();
        Check(detector.RetainedWorkspaceBytes == 0, "ReleaseWorkspace did not release references");
        Check(faces.SequenceEqual(detector.Detect(rgb, width, height, width * 3, PixelFormat.Rgb24)), "Workspace release changed results");
        using (var disposed = new FaceDetector())
        {
            disposed.Detect(rgb, width, height, width * 3, PixelFormat.Rgb24);
            disposed.Dispose();
            Check(disposed.RetainedWorkspaceBytes == 0, "Dispose did not release references");
            disposed.Dispose();
            Reject<ObjectDisposedException>(() => disposed.ReleaseWorkspace());
            Reject<ObjectDisposedException>(() => disposed.Detect(tiny, 1, 1, 3));
        }
        TestRunner.Pass("API/workspace release and disposal");
        ConcurrentWave("independent-reuse", 4, _ =>
        {
            using var independent = new FaceDetector();
            Check(faces.SequenceEqual(independent.Detect(rgb, width, height, width * 3, PixelFormat.Rgb24)), "Independent detector mismatch");
        });
        ConcurrentWave("shared-instance", 8, _ => Check(faces.SequenceEqual(detector.Detect(rgb, width, height, width * 3, PixelFormat.Rgb24)), "Same-instance result mismatch"));
        TestRunner.Pass("API/independent reuse and shared-instance concurrency");
        Console.WriteLine($"PASS: real face, landmarks, score, RGB/BGR/RGBA, padded RGB/RGBA, changing resolutions, input immutability, stride, buffer bounds, concurrency; {face}");
    }

    static void Check(bool value, string message)
    {
        if (!value) throw new Exception(message);
    }
    static void Reject<T>(Action action) where T : Exception
    {
        try { action(); } catch (T) { return; }
        throw new Exception($"Expected {typeof(T).Name}");
    }
    
    static void ConcurrentWave(string name, int count, Action<int> action)
    {
        using var ready = new CountdownEvent(count);
        using var start = new ManualResetEventSlim(false);
        var errors = new System.Collections.Concurrent.ConcurrentQueue<Exception>();
        var participants = new System.Collections.Concurrent.ConcurrentDictionary<int, byte>();
        int active = 0, peak = 0;
        var threads = new Thread[count];
        for (int i = 0; i < count; i++)
        {
            int worker = i;
            threads[i] = new Thread(() =>
            {
                participants.TryAdd(Environment.CurrentManagedThreadId, 0);
                ready.Signal();
                start.Wait();
                int current = Interlocked.Increment(ref active);
                int observed;
                do
                {
                    observed = Volatile.Read(ref peak);
                    if (current <= observed) break;
                } while (Interlocked.CompareExchange(ref peak, current, observed) != observed);
                try { action(worker); }
                catch (Exception error) { errors.Enqueue(error); }
                finally { Interlocked.Decrement(ref active); }
            }) { IsBackground = true, Name = $"{name}-{worker}" };
            threads[i].Start();
        }
        bool allReady = ready.Wait(TimeSpan.FromSeconds(30));
        start.Set();
        Check(allReady, $"{name}: workers did not reach the start gate");
        foreach (Thread thread in threads)
            Check(thread.Join(TimeSpan.FromSeconds(30)), $"{name}: worker did not finish");
        if (!errors.IsEmpty) throw new AggregateException(errors);
        Check(participants.Count == count, $"{name}: expected {count} distinct threads, saw {participants.Count}");
        Console.WriteLine($"CONCURRENCY {name}: participating_threads={participants.Count}, peak_inflight_calls={peak}, processor_count={Environment.ProcessorCount}");
    }
    
}
