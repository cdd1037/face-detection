namespace FaceDetection;

internal sealed class Model
{
    private static readonly object Gate = new();
    private static Model? instance;
    internal static Model Instance
    {
        get
        {
            var current = Volatile.Read(ref instance);
            if (current is not null) return current;
            lock (Gate)
            {
                current = instance;
                if (current is null) { current = Load(); Volatile.Write(ref instance, current); }
                return current;
            }
        }
    }
    internal readonly Filter[] Filters;
    internal readonly long RetainedBytes;
    private Model(Filter[] filters)
    {
        Filters = filters;
        foreach (var f in filters) RetainedBytes += ((long)f.Weights.Length + f.Biases.Length + f.Packed.Length) * sizeof(float);
    }
    // Fixed graph contract also protects the unchecked SIMD loads from a mismatched model.
    private static ReadOnlySpan<int> ExpectedInputs => [32, 16, 16, 16, 16, 16, 32, 32, 32, 32, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 1, 64, 1, 64, 1, 64, 4, 64, 4, 64, 4, 64, 1, 64, 1, 64, 1, 64, 10, 64, 10, 64, 10];
    private static ReadOnlySpan<int> ExpectedOutputs => [16, 16, 16, 16, 16, 32, 32, 32, 32, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 1, 1, 1, 1, 1, 1, 4, 4, 4, 4, 4, 4, 1, 1, 1, 1, 1, 1, 10, 10, 10, 10, 10, 10];

    private static Model Load()
    {
        using var stream = typeof(Model).Assembly.GetManifestResourceStream("FaceDetection.Model.yunet.bin")
            ?? throw new InvalidOperationException("Embedded detector model is missing.");
        using var reader = new BinaryReader(stream);
        if (reader.ReadUInt32() != 0x3144464d || reader.ReadInt32() != 53)
            throw new InvalidDataException("Unsupported embedded detector model.");
        var filters = new Filter[53];
        for (int i = 0; i < filters.Length; i++)
        {
            int input = reader.ReadInt32(), output = reader.ReadInt32(), depthwise = reader.ReadInt32();
            int weightsCount = reader.ReadInt32(), biasesCount = reader.ReadInt32();
            if (input is < 1 or > 64 || output is < 1 or > 64 || depthwise is < 0 or > 1 ||
                depthwise == 1 && input != output || weightsCount != input * (depthwise == 1 ? 9 : output) || biasesCount != output)
                throw new InvalidDataException("Invalid embedded layer dimensions.");
            if (input != ExpectedInputs[i] || output != ExpectedOutputs[i] || depthwise != (i >= 2 && i % 2 == 0 ? 1 : 0))
                throw new InvalidDataException("Embedded model does not match the fixed 53-layer graph.");
            float[] weights = ReadFloats(reader, weightsCount), biases = ReadFloats(reader, biasesCount);
            filters[i] = new(input, output, depthwise == 1, weights, biases);
        }
        if (stream.Position != stream.Length) throw new InvalidDataException("Unexpected trailing model data.");
        return new Model(filters);
    }
    private static float[] ReadFloats(BinaryReader reader, int count)
    {
        var data = new float[count];
        for (int i = 0; i < count; i++)
        {
            data[i] = reader.ReadSingle();
            if (!float.IsFinite(data[i])) throw new InvalidDataException("Non-finite embedded weight.");
        }
        return data;
    }
}
