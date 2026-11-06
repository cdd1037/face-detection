namespace FaceDetection.Tests;

internal static class Program
{
    private static int Main(string[] args)
    {
        if (args.Length != 0)
        {
            Console.Error.WriteLine("This executable runs all regression tests and accepts no arguments. Use benchmarks/ for performance measurements.");
            return 2;
        }
        // API must run first: its synchronized first-use test exercises model publication.
        TestRunner.RunSuite("API", ApiTests.Run);
        TestRunner.RunSuite("Kernels", KernelTests.Run);
        TestRunner.RunSuite("Postprocess", PostprocessTests.Run);
        Console.WriteLine($"{TestRunner.Passed} named checks passed; {TestRunner.Failed} suites failed.");
        return TestRunner.Failed == 0 ? 0 : 1;
    }
}

internal static class TestRunner
{
    internal static int Passed { get; private set; }
    internal static int Failed { get; private set; }

    internal static void Pass(string name)
    {
        Passed++;
        Console.WriteLine($"PASS {name}");
    }

    internal static void RunSuite(string name, Action suite)
    {
        Console.WriteLine($"SUITE {name}");
        try { suite(); }
        catch (Exception error)
        {
            Failed++;
            Console.Error.WriteLine($"FAIL {name}: {error}");
        }
    }
}
