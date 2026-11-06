namespace FaceDetection;

internal sealed class Tensor
{
    internal float[] Data = Array.Empty<float>();
    internal int Rows, Cols, Channels;
    internal void Resize(int rows, int cols, int channels)
    {
        int count = checked(rows * cols * channels);
        if (Data.Length < count) Data = GC.AllocateUninitializedArray<float>(count);
        Rows = rows; Cols = cols; Channels = channels;
    }
}
