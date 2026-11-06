using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text.Json;
using FaceDetection;

if (args.Length > 1 || (args.Length == 1 && (!int.TryParse(args[0], out int value) || value < 1 || value > 1000)))
{
    Console.Error.WriteLine("Usage: FaceDetection.Benchmarks [iterations: 1..1000] (default: 15)");
    return 2;
}
int iterations = args.Length == 0 ? 15 : int.Parse(args[0]);
byte[] source = File.ReadAllBytes(Path.Combine(AppContext.BaseDirectory, "astronaut.rgb"));
Console.WriteLine(JsonSerializer.Serialize(new
{
    runtime = RuntimeInformation.FrameworkDescription,
    architecture = RuntimeInformation.ProcessArchitecture.ToString(),
    simd = FaceDetector.IsSimdEnabled,
    simd_lanes = FaceDetector.SimdFloatLanes,
    iterations
}));
foreach ((int width, int height) in new[] { (256, 256), (512, 512), (640, 480) })
{
    var input = new byte[width * height * 3];
    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++)
            Array.Copy(source, ((y * 256 / height) * 256 + x * 256 / width) * 3,
                input, (y * width + x) * 3, 3);
    using var detector = new FaceDetector();
    long firstStart = Stopwatch.GetTimestamp();
    int faces = detector.Detect(input, width, height, width * 3, PixelFormat.Rgb24).Length;
    double firstMilliseconds = Stopwatch.GetElapsedTime(firstStart).TotalMilliseconds;
    if (faces == 0) throw new InvalidOperationException("Fixture detection failed.");
    for (int i = 0; i < 5; i++) detector.Detect(input, width, height, width * 3, PixelFormat.Rgb24);
    var samples = new double[iterations];
    long allocatedBefore = GC.GetAllocatedBytesForCurrentThread();
    for (int i = 0; i < iterations; i++)
    {
        long start = Stopwatch.GetTimestamp();
        var result = detector.Detect(input, width, height, width * 3, PixelFormat.Rgb24);
        samples[i] = Stopwatch.GetElapsedTime(start).TotalMilliseconds;
        if (result.Length != faces) throw new InvalidOperationException("Detection count changed between iterations.");
    }
    long allocated = GC.GetAllocatedBytesForCurrentThread() - allocatedBefore;
    Array.Sort(samples);
    double median = iterations % 2 == 0 ? (samples[iterations / 2 - 1] + samples[iterations / 2]) / 2 : samples[iterations / 2];
    Console.WriteLine(JsonSerializer.Serialize(new
    {
        width, height, faces,
        first_call_ms = firstMilliseconds,
        median_ms = median,
        min_ms = samples[0],
        max_ms = samples[^1],
        allocated_bytes_per_call = (double)allocated / iterations,
        retained_workspace_bytes = detector.RetainedWorkspaceBytes
    }));
}
return 0;
