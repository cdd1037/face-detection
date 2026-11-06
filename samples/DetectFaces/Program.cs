using System.Globalization;
using System.Text.Json;
using System.Text.Json.Serialization;
using FaceDetection;

if (args.Length is < 4 or > 5)
{
    Console.Error.WriteLine("Usage: DetectFaces <raw-file> <width> <height> <bgr24|rgb24|rgba32> [stride]");
    return 2;
}

try
{
    int width = int.Parse(args[1], NumberStyles.None, CultureInfo.InvariantCulture);
    int height = int.Parse(args[2], NumberStyles.None, CultureInfo.InvariantCulture);
    PixelFormat format = args[3].ToLowerInvariant() switch
    {
        "bgr24" => PixelFormat.Bgr24,
        "rgb24" => PixelFormat.Rgb24,
        "rgba32" => PixelFormat.Rgba32,
        _ => throw new ArgumentException("Format must be bgr24, rgb24 or rgba32.")
    };
    int stride = args.Length == 5
        ? int.Parse(args[4], NumberStyles.None, CultureInfo.InvariantCulture)
        : checked(width * (format == PixelFormat.Rgba32 ? 4 : 3));
    byte[] pixels = File.ReadAllBytes(args[0]);
    using var detector = new FaceDetector();
    DetectedFace[] faces = detector.Detect(pixels, width, height, stride, format);
    Console.WriteLine(JsonSerializer.Serialize(faces, FaceJsonContext.Default.DetectedFaceArray));
    return 0;
}
catch (Exception error) when (error is ArgumentException or FormatException or OverflowException
    or IOException or UnauthorizedAccessException)
{
    Console.Error.WriteLine(error.Message);
    return 1;
}

[JsonSerializable(typeof(DetectedFace[]))]
internal partial class FaceJsonContext : JsonSerializerContext;
