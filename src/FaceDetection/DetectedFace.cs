namespace FaceDetection;

/// <summary>One detected face, its landmarks and an uncalibrated model score.</summary>
/// <param name="Bounds">Unclipped integer bounds in the original image.</param>
/// <param name="Landmarks">Unclipped integer landmarks in the original image.</param>
/// <param name="Score">Model confidence in [0, 1], not a calibrated probability.</param>
public readonly record struct DetectedFace(FaceRectangle Bounds, FaceLandmarks Landmarks, float Score);
