namespace FaceDetection;

/// <summary>An axis-aligned rectangle in input-image pixels. Bounds are not clipped.</summary>
/// <param name="X">Left edge.</param>
/// <param name="Y">Top edge.</param>
/// <param name="Width">Rectangle width.</param>
/// <param name="Height">Rectangle height.</param>
public readonly record struct FaceRectangle(int X, int Y, int Width, int Height);
