namespace FaceDetection;

/// <summary>Five facial landmarks. Left and right refer to the subject, not the viewer.</summary>
/// <param name="RightEye">Center of the subject's right eye.</param>
/// <param name="LeftEye">Center of the subject's left eye.</param>
/// <param name="Nose">Nose point.</param>
/// <param name="RightMouthCorner">Subject's right mouth corner.</param>
/// <param name="LeftMouthCorner">Subject's left mouth corner.</param>
public readonly record struct FaceLandmarks(FacePoint RightEye, FacePoint LeftEye, FacePoint Nose,
    FacePoint RightMouthCorner, FacePoint LeftMouthCorner);
