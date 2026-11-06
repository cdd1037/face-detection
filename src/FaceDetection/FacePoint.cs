namespace FaceDetection;

/// <summary>A point in input-image pixel coordinates. Coordinates are not clipped.</summary>
/// <param name="X">Horizontal coordinate, measured from the left.</param>
/// <param name="Y">Vertical coordinate, measured from the top.</param>
public readonly record struct FacePoint(int X, int Y);
