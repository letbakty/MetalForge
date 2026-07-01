import Metal
import simd
import Foundation

// ===========================================================================
// DistortionFilters — v0.3.0 Distortion Pack.
//
// Eight UV-remap filters sharing DistortionKernels.metal. All are single-pass
// and reuse the `SinglePassColorFilter` base (SDR/HDR PSO pair + dispatch).
// ===========================================================================

// MARK: - BulgeDistortionFilter

private struct BulgeUniforms {
    var center: SIMD2<Float>
    var radius: Float
    var scale:  Float
}

/// Fisheye-style radial magnification inside a disc.
public final class BulgeDistortionFilter: SinglePassColorFilter, @unchecked Sendable {

    /// Normalised centre of the effect. Default frame centre.
    public var center: SIMD2<Float> = SIMD2(0.5, 0.5)
    /// Normalised radius of the affected disc. Clamped to `[0, 1]`. Default `0.25`.
    public var radius: Float = 0.25
    /// Magnification: `>0` bulges out, `<0` dents in. Clamped to `[-1, 1]`. Default `0.5`.
    public var scale: Float = 0.5

    public init(engine: MetalForgeEngine) throws {
        try super.init(engine: engine, kernel: "bulgeDistortionKernel", label: "BulgeDistortionFilter")
    }

    override func setUniforms(on encoder: MTLComputeCommandEncoder) {
        var u = BulgeUniforms(
            center: center,
            radius: min(max(radius, 0), 1),
            scale:  min(max(scale, -1), 1)
        )
        encoder.setBytes(&u, length: MemoryLayout<BulgeUniforms>.stride, index: 0)
    }
}

// MARK: - PinchDistortionFilter

private struct PinchUniforms {
    var center: SIMD2<Float>
    var radius: Float
    var scale:  Float
}

/// Radial squeeze toward the centre — the inverse feel of bulge.
public final class PinchDistortionFilter: SinglePassColorFilter, @unchecked Sendable {

    /// Normalised centre of the effect. Default frame centre.
    public var center: SIMD2<Float> = SIMD2(0.5, 0.5)
    /// Normalised radius of the affected disc. Clamped to `[0, 2]`. Default `0.5`.
    public var radius: Float = 0.5
    /// Pinch strength. Clamped to `[0, 2]`; `0` = identity. Default `0.5`.
    public var scale: Float = 0.5

    public init(engine: MetalForgeEngine) throws {
        try super.init(engine: engine, kernel: "pinchDistortionKernel", label: "PinchDistortionFilter")
    }

    override func setUniforms(on encoder: MTLComputeCommandEncoder) {
        var u = PinchUniforms(
            center: center,
            radius: min(max(radius, 0), 2),
            scale:  min(max(scale, 0), 2)
        )
        encoder.setBytes(&u, length: MemoryLayout<PinchUniforms>.stride, index: 0)
    }
}

// MARK: - StretchDistortionFilter

/// Centre-anchored stretch: the middle of the frame is magnified, the outer
/// band compressed (funhouse-mirror look).
public final class StretchDistortionFilter: SinglePassColorFilter, @unchecked Sendable {

    /// Normalised anchor of the stretch. Default frame centre.
    public var center: SIMD2<Float> = SIMD2(0.5, 0.5)

    public init(engine: MetalForgeEngine) throws {
        try super.init(engine: engine, kernel: "stretchDistortionKernel", label: "StretchDistortionFilter")
    }

    override func setUniforms(on encoder: MTLComputeCommandEncoder) {
        var c = center
        encoder.setBytes(&c, length: MemoryLayout<SIMD2<Float>>.stride, index: 0)
    }
}

// MARK: - SwirlFilter

private struct SwirlUniforms {
    var center: SIMD2<Float>
    var radius: Float
    var angle:  Float
}

/// Angular twist that is strongest at the centre and decays to zero at the
/// rim of the affected disc.
public final class SwirlFilter: SinglePassColorFilter, @unchecked Sendable {

    /// Normalised centre of the twist. Default frame centre.
    public var center: SIMD2<Float> = SIMD2(0.5, 0.5)
    /// Normalised radius of the affected disc. Clamped to `[0, 1]`. Default `0.5`.
    public var radius: Float = 0.5
    /// Twist at the centre, in radians. Clamped to `[-2π, 2π]`. Default `1`.
    public var angle: Float = 1.0

    public init(engine: MetalForgeEngine) throws {
        try super.init(engine: engine, kernel: "swirlDistortionKernel", label: "SwirlFilter")
    }

    override func setUniforms(on encoder: MTLComputeCommandEncoder) {
        var u = SwirlUniforms(
            center: center,
            radius: min(max(radius, 0), 1),
            angle:  min(max(angle, -2 * .pi), 2 * .pi)
        )
        encoder.setBytes(&u, length: MemoryLayout<SwirlUniforms>.stride, index: 0)
    }
}

// MARK: - Sphere refraction / glass sphere

private struct SphereUniforms {
    var center:          SIMD2<Float>
    var radius:          Float
    var refractiveIndex: Float
}

/// Refraction through a solid glass ball; everything outside the ball is black.
public final class SphereRefractionFilter: SinglePassColorFilter, @unchecked Sendable {

    /// Normalised centre of the sphere. Default frame centre.
    public var center: SIMD2<Float> = SIMD2(0.5, 0.5)
    /// Normalised radius of the sphere. Clamped to `[0, 1]`. Default `0.25`.
    public var radius: Float = 0.25
    /// Index of refraction ratio. Clamped to `[0.1, 1]`. Default `0.71` (glass).
    public var refractiveIndex: Float = 0.71

    public init(engine: MetalForgeEngine) throws {
        try super.init(engine: engine, kernel: "sphereRefractionKernel", label: "SphereRefractionFilter")
    }

    override func setUniforms(on encoder: MTLComputeCommandEncoder) {
        var u = SphereUniforms(
            center:          center,
            radius:          min(max(radius, 0), 1),
            refractiveIndex: min(max(refractiveIndex, 0.1), 1)
        )
        encoder.setBytes(&u, length: MemoryLayout<SphereUniforms>.stride, index: 0)
    }
}

/// A glass ball floating over the scene: refracted + mirrored image inside,
/// untouched scene outside, with a soft rim light.
public final class GlassSphereFilter: SinglePassColorFilter, @unchecked Sendable {

    /// Normalised centre of the sphere. Default frame centre.
    public var center: SIMD2<Float> = SIMD2(0.5, 0.5)
    /// Normalised radius of the sphere. Clamped to `[0, 1]`. Default `0.25`.
    public var radius: Float = 0.25
    /// Index of refraction ratio. Clamped to `[0.1, 1]`. Default `0.71` (glass).
    public var refractiveIndex: Float = 0.71

    public init(engine: MetalForgeEngine) throws {
        try super.init(engine: engine, kernel: "glassSphereKernel", label: "GlassSphereFilter")
    }

    override func setUniforms(on encoder: MTLComputeCommandEncoder) {
        var u = SphereUniforms(
            center:          center,
            radius:          min(max(radius, 0), 1),
            refractiveIndex: min(max(refractiveIndex, 0.1), 1)
        )
        encoder.setBytes(&u, length: MemoryLayout<SphereUniforms>.stride, index: 0)
    }
}

// MARK: - CropFilter

private struct CropUniforms {
    var origin:   SIMD2<Float>
    var cropSize: SIMD2<Float>
}

/// Crops a normalised region and scales it back up to fill the frame
/// (crop-and-zoom — the pipeline's output size is fixed).
public final class CropFilter: SinglePassColorFilter, @unchecked Sendable {

    /// Normalised top-left corner of the crop region. Default `(0, 0)`.
    public var origin: SIMD2<Float> = SIMD2(0, 0)
    /// Normalised size of the crop region. Components clamped to `[0.01, 1]`.
    /// Default `(1, 1)` = identity.
    public var size: SIMD2<Float> = SIMD2(1, 1)

    public init(engine: MetalForgeEngine) throws {
        try super.init(engine: engine, kernel: "cropKernel", label: "CropFilter")
    }

    override func setUniforms(on encoder: MTLComputeCommandEncoder) {
        let clampedSize = simd_clamp(size, SIMD2(repeating: 0.01), SIMD2(repeating: 1))
        let clampedOrigin = simd_clamp(origin, SIMD2(repeating: 0), SIMD2(repeating: 1) - clampedSize)
        var u = CropUniforms(origin: clampedOrigin, cropSize: clampedSize)
        encoder.setBytes(&u, length: MemoryLayout<CropUniforms>.stride, index: 0)
    }
}

// MARK: - TransformFilter

private struct TransformUniforms {
    var matrix: simd_float3x3
}

/// Affine transform (rotate / scale / translate) around the frame centre.
/// Pixels mapped from outside the source come out transparent black.
public final class TransformFilter: SinglePassColorFilter, @unchecked Sendable {

    /// Rotation in radians, counter-clockwise. Default `0`.
    public var rotation: Float = 0.0
    /// Per-axis scale. Components clamped to `[0.05, 20]`. Default `(1, 1)`.
    public var scale: SIMD2<Float> = SIMD2(1, 1)
    /// Translation in normalised UV units. Default `(0, 0)`.
    public var translation: SIMD2<Float> = SIMD2(0, 0)

    public init(engine: MetalForgeEngine) throws {
        try super.init(engine: engine, kernel: "transformKernel", label: "TransformFilter")
    }

    override func setUniforms(on encoder: MTLComputeCommandEncoder) {
        // The kernel needs the *inverse* mapping (destination → source). For
        // T·R·S the inverse is S⁻¹·R⁻¹·T⁻¹, which is what we build directly.
        let s = simd_clamp(scale, SIMD2(repeating: 0.05), SIMD2(repeating: 20))
        let invS = SIMD2<Float>(1 / s.x, 1 / s.y)
        let cosA = cos(-rotation)
        let sinA = sin(-rotation)

        // Columns of the 3×3 matrix (simd is column-major).
        let col0 = SIMD3<Float>(cosA * invS.x, sinA * invS.y, 0)
        let col1 = SIMD3<Float>(-sinA * invS.x, cosA * invS.y, 0)
        let col2 = SIMD3<Float>(
            -(translation.x * cosA - translation.y * sinA) * invS.x,
            -(translation.x * sinA + translation.y * cosA) * invS.y,
            1
        )
        var u = TransformUniforms(matrix: simd_float3x3(col0, col1, col2))
        encoder.setBytes(&u, length: MemoryLayout<TransformUniforms>.stride, index: 0)
    }
}
