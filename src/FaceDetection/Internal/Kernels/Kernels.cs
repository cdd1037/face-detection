using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Runtime.Intrinsics.Arm;

namespace FaceDetection;

internal static partial class Kernels
{
    // Native AOT folds hardware feature checks at build time; runtime environment
    // switches are not sufficient to request scalar inference there.
    internal static bool SimdEnabled
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get
        {
#if FD_SCALAR
            return false;
#else
            return Vector.IsHardwareAccelerated;
#endif
        }
    }
    // Bounds are established by internal tensor dimensions. These BCL APIs do not pin arrays,
    // allocate native memory, or require /unsafe; callers cannot supply model/tensor dimensions.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector<float> Load(float[] array, int offset) => Vector.LoadUnsafe(ref MemoryMarshal.GetArrayDataReference(array), (nuint)offset);
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Store(Vector<float> value, float[] array, int offset) => value.StoreUnsafe(ref MemoryMarshal.GetArrayDataReference(array), (nuint)offset);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector<float> MultiplyAdd(Vector<float> a, Vector<float> b, Vector<float> sum)
    {
        // Optional BCL intrinsics, not native interop. The JIT folds these feature guards.
        if (Fma.IsSupported && Vector<float>.Count == 8)
        {
            var result = Fma.MultiplyAdd(Unsafe.As<Vector<float>, Vector256<float>>(ref a),
                Unsafe.As<Vector<float>, Vector256<float>>(ref b), Unsafe.As<Vector<float>, Vector256<float>>(ref sum));
            return Unsafe.As<Vector256<float>, Vector<float>>(ref result);
        }
        if (Fma.IsSupported && Vector<float>.Count == 4)
        {
            var result = Fma.MultiplyAdd(Unsafe.As<Vector<float>, Vector128<float>>(ref a),
                Unsafe.As<Vector<float>, Vector128<float>>(ref b), Unsafe.As<Vector<float>, Vector128<float>>(ref sum));
            return Unsafe.As<Vector128<float>, Vector<float>>(ref result);
        }
        if (AdvSimd.IsSupported && Vector<float>.Count == 4)
        {
            var result = AdvSimd.FusedMultiplyAdd(Unsafe.As<Vector<float>, Vector128<float>>(ref sum),
                Unsafe.As<Vector<float>, Vector128<float>>(ref a), Unsafe.As<Vector<float>, Vector128<float>>(ref b));
            return Unsafe.As<Vector128<float>, Vector<float>>(ref result);
        }
        return a * b + sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float Read(float[] array, int offset) => Unsafe.Add(ref MemoryMarshal.GetArrayDataReference(array), offset);
}
