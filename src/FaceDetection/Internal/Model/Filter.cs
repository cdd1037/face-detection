namespace FaceDetection;

internal sealed class Filter
{
    internal readonly int InputChannels, OutputChannels;
    internal readonly bool Depthwise;
    internal readonly float[] Weights, Biases, Packed;
    internal Filter(int input, int output, bool depthwise, float[] weights, float[] biases)
    {
        InputChannels = input; OutputChannels = output; Depthwise = depthwise;
        Weights = weights; Biases = biases;
        Packed = !depthwise && output >= 16 ? new float[weights.Length] : Array.Empty<float>();
        for (int oc = 0; oc < output && Packed.Length != 0; oc++)
            for (int ic = 0; ic < input; ic++) Packed[ic * output + oc] = weights[oc * input + ic];
    }
}
