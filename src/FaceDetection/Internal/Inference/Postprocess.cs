using System;

namespace FaceDetection;

/// <summary>Fused head decoding and greedy NMS, with reusable per-session scratch.</summary>
internal sealed class Postprocess
{
    private const float SigmoidLimit = 88.3762626647949f;
    private const float ConfidenceThreshold = 0.2f;
    private const float OverlapThreshold = 0.45f;
    private const int TopK = 1000;
    private const int KeepTopK = 512;

    // Landmarks are decoded only for survivors. Scale and pixel also provide the
    // original scan order, making the comparison equivalent to a stable score sort.
    private struct Candidate
    {
        internal float Score, XMin, YMin, XMax, YMax;
        internal int Scale, Pixel;
    }

    private static int Compare(Candidate a, Candidate b)
    {
        int score = b.Score.CompareTo(a.Score);
        if (score != 0) return score;
        int scale = a.Scale.CompareTo(b.Scale);
        return scale != 0 ? scale : a.Pixel.CompareTo(b.Pixel);
    }

    // In-place heap sort avoids the per-call comparison delegate allocated by
    // Array.Sort's IComparer overload. Unique scale/pixel tie keys preserve the
    // native stable-score ordering even though heap sort itself is not stable.
    private void SortCandidates(int count)
    {
        for (int root = count / 2 - 1; root >= 0; --root)
            SiftDown(root, count);
        for (int end = count - 1; end > 0; --end)
        {
            Candidate last = candidates[end];
            candidates[end] = candidates[0];
            candidates[0] = last;
            SiftDown(0, end);
        }
    }

    private void SiftDown(int root, int count)
    {
        Candidate value = candidates[root];
        // Nodes before count / 2 have a child; this also bounds 2 * root + 1.
        while (root < count / 2)
        {
            int child = 2 * root + 1;
            if (child + 1 < count && Compare(candidates[child], candidates[child + 1]) < 0)
                ++child;
            if (Compare(value, candidates[child]) >= 0) break;
            candidates[root] = candidates[child];
            root = child;
        }
        candidates[root] = value;
    }

    private Candidate[] candidates = Array.Empty<Candidate>();
    private int[] kept = Array.Empty<int>();

    // Payload only: five floats and two ints per candidate, one int per kept index.
    internal long RetainedBytes => (long)candidates.Length * 28 + (long)kept.Length * sizeof(int);

    internal void Release()
    {
        candidates = Array.Empty<Candidate>();
        kept = Array.Empty<int>();
    }

    internal DetectedFace[] Run(Tensor[] cls, Tensor[] reg, Tensor[] kps, Tensor[] obj)
    {
        int count = 0;
        for (int scale = 0; scale < 3; ++scale)
        {
            Tensor classification = cls[scale];
            Tensor regression = reg[scale];
            Tensor objectness = obj[scale];
            int stride = 8 << scale;
            for (int row = 0; row < classification.Rows; ++row)
            {
                float priorY = row * stride;
                for (int col = 0; col < classification.Cols; ++col)
                {
                    int pixel = row * classification.Cols + col;
                    float confidence = MathF.Sqrt(
                        Sigmoid(classification.Data[pixel * classification.Channels]) *
                        Sigmoid(objectness.Data[pixel * objectness.Channels]));
                    if (!(confidence >= ConfidenceThreshold)) continue;

                    int offset = pixel * regression.Channels;
                    float priorX = col * stride;
                    float cx = regression.Data[offset] * stride + priorX;
                    float cy = regression.Data[offset + 1] * stride + priorY;
                    float width = MathF.Exp(regression.Data[offset + 2]) * stride;
                    float height = MathF.Exp(regression.Data[offset + 3]) * stride;
                    if (count == candidates.Length)
                        Array.Resize(ref candidates, candidates.Length == 0 ? 256 : checked(candidates.Length * 2));
                    candidates[count++] = new Candidate
                    {
                        Score = confidence,
                        XMin = cx - width / 2.0f,
                        YMin = cy - height / 2.0f,
                        XMax = cx + width / 2.0f,
                        YMax = cy + height / 2.0f,
                        Scale = scale,
                        Pixel = pixel
                    };
                }
            }
        }

        if (count == 0) return Array.Empty<DetectedFace>();
        SortCandidates(count);
        int limit = Math.Min(count, TopK);
        if (kept.Length == 0) kept = new int[KeepTopK];
        int keptCount = 0;
        for (int i = 0; i < limit; ++i)
        {
            bool keep = true;
            for (int j = 0; j < keptCount; ++j)
            {
                // Use <= rather than negating > to preserve native NaN handling.
                if (!(Overlap(candidates[i], candidates[kept[j]]) <= OverlapThreshold))
                {
                    keep = false;
                    break;
                }
            }
            if (!keep) continue;
            kept[keptCount++] = i;
            // Later candidates cannot change the first 512 greedy survivors.
            if (keptCount == KeepTopK) break;
        }

        var result = new DetectedFace[keptCount];
        for (int i = 0; i < keptCount; ++i)
        {
            Candidate candidate = candidates[kept[i]];
            Tensor points = kps[candidate.Scale];
            int stride = 8 << candidate.Scale;
            float priorX = (candidate.Pixel % points.Cols) * stride;
            float priorY = (candidate.Pixel / points.Cols) * stride;
            int offset = candidate.Pixel * points.Channels;
            var landmarks = new FaceLandmarks(
                DecodePoint(points.Data, offset, stride, priorX, priorY),
                DecodePoint(points.Data, offset + 2, stride, priorX, priorY),
                DecodePoint(points.Data, offset + 4, stride, priorX, priorY),
                DecodePoint(points.Data, offset + 6, stride, priorX, priorY),
                DecodePoint(points.Data, offset + 8, stride, priorX, priorY));
            result[i] = new DetectedFace(
                new FaceRectangle((int)candidate.XMin, (int)candidate.YMin,
                    (int)(candidate.XMax - candidate.XMin), (int)(candidate.YMax - candidate.YMin)),
                landmarks, candidate.Score);
        }
        return result;
    }

    private static float Sigmoid(float value)
    {
        value = MathF.Min(value, SigmoidLimit);
        value = MathF.Max(value, -SigmoidLimit);
        return 1.0f / (1.0f + MathF.Exp(-value));
    }

    private static FacePoint DecodePoint(float[] data, int offset, int stride, float priorX, float priorY)
        => new((int)(data[offset] * stride + priorX), (int)(data[offset + 1] * stride + priorY));

    private static float Overlap(Candidate a, Candidate b)
    {
        if (b.XMin > a.XMax || b.XMax < a.XMin || b.YMin > a.YMax || b.YMax < a.YMin)
            return 0.0f;
        float width = MathF.Min(a.XMax, b.XMax) - MathF.Max(a.XMin, b.XMin);
        float height = MathF.Min(a.YMax, b.YMax) - MathF.Max(a.YMin, b.YMin);
        if (width <= 0.0f || height <= 0.0f) return 0.0f;
        float intersection = width * height;
        float areaA = (a.XMax - a.XMin) * (a.YMax - a.YMin);
        float areaB = (b.XMax - b.XMin) * (b.YMax - b.YMin);
        return intersection / (areaA + areaB - intersection);
    }
}
