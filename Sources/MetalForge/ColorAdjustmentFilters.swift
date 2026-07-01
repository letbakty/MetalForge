import Metal
import simd
import Foundation

// ===========================================================================
// ColorAdjustmentFilters — v0.2.0 Color Pack (correction half).
//
// Twelve single-pass colour filters sharing ColorAdjustmentKernels.metal.
// Every filter follows the established MetalForge pattern: two isHDR-
// specialised PSOs selected by the source pixel format at encode time,
// SIMD-aligned non-uniform dispatch, uniforms via `setBytes`, identity
// defaults, and no per-frame allocation.
// ===========================================================================

// MARK: - Shared helpers

/// Compile one isHDR-specialised PSO. Internal so every effect pack can share
/// it instead of redeclaring the same helper.
func makeColorAdjustmentPSO(
    engine: MetalForgeEngine,
    kernel: String,
    isHDR: Bool
) throws -> MTLComputePipelineState {
    let constants = MTLFunctionConstantValues()
    var flag = isHDR
    constants.setConstantValue(&flag, type: .bool, index: 0)

    let function = try engine.makeFunction(name: kernel, constantValues: constants)
    do {
        return try engine.device.makeComputePipelineState(function: function)
    } catch {
        throw MetalForgeError.pipelineStateCreationFailed(error.localizedDescription)
    }
}

/// SIMD-aligned threadgroup dispatch — internal for reuse across packs.
func dispatchColorAdjustment(
    encoder: MTLComputeCommandEncoder,
    pso: MTLComputePipelineState,
    width: Int,
    height: Int
) {
    let simdWidth = pso.threadExecutionWidth
    let groupH    = pso.maxTotalThreadsPerThreadgroup / simdWidth
    encoder.dispatchThreads(
        MTLSize(width: width, height: height, depth: 1),
        threadsPerThreadgroup: MTLSize(width: simdWidth, height: groupH, depth: 1)
    )
}

/// Base class factoring the SDR/HDR PSO pair + encode plumbing shared by the
/// simple single-pass filters in this file. Subclasses provide the kernel name
/// and write their uniforms in `setUniforms(on:)`.
// @unchecked Sendable: mutable configuration is written by the caller between
// frames, matching every other MetalForge filter.
public class SinglePassColorFilter: @unchecked Sendable, MetalForgeFilter {

    private let label:  String
    private let sdrPSO: MTLComputePipelineState
    private let hdrPSO: MTLComputePipelineState

    init(engine: MetalForgeEngine, kernel: String, label: String) throws {
        self.label  = label
        self.sdrPSO = try makeColorAdjustmentPSO(engine: engine, kernel: kernel, isHDR: false)
        self.hdrPSO = try makeColorAdjustmentPSO(engine: engine, kernel: kernel, isHDR: true)
    }

    /// Subclass hook: upload the uniform buffer (if any) at buffer index 0 and
    /// any extra textures from index 2 upward.
    func setUniforms(on encoder: MTLComputeCommandEncoder) {}

    public func encode(
        source: MTLTexture,
        destination: MTLTexture,
        commandBuffer: MTLCommandBuffer
    ) {
        let pso = (source.pixelFormat == .rgba16Float) ? hdrPSO : sdrPSO

        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = label
        encoder.setComputePipelineState(pso)
        encoder.setTexture(source,      index: 0)
        encoder.setTexture(destination, index: 1)
        setUniforms(on: encoder)

        dispatchColorAdjustment(
            encoder: encoder, pso: pso,
            width: destination.width, height: destination.height
        )
        encoder.endEncoding()
    }
}

// MARK: - GammaFilter

/// Per-channel power curve. `gamma = 1` is the identity; values below 1
/// brighten the midtones, values above 1 darken them.
public final class GammaFilter: SinglePassColorFilter, @unchecked Sendable {

    /// Exponent applied to each channel. Clamped to `[0.05, 5]`. Default `1`.
    public var gamma: Float = 1.0

    public init(engine: MetalForgeEngine) throws {
        try super.init(engine: engine, kernel: "gammaKernel", label: "GammaFilter")
    }

    override func setUniforms(on encoder: MTLComputeCommandEncoder) {
        var g = min(max(gamma, 0.05), 5)
        encoder.setBytes(&g, length: MemoryLayout<Float>.stride, index: 0)
    }
}

// MARK: - LevelsFilter

private struct LevelsUniforms {
    var inMin:  SIMD3<Float>; var pad0: Float = 0
    var inMax:  SIMD3<Float>; var pad1: Float = 0
    var outMin: SIMD3<Float>; var pad2: Float = 0
    var outMax: SIMD3<Float>; var pad3: Float = 0
    var gamma:  SIMD3<Float>; var pad4: Float = 0
}

/// Photoshop-style levels: per-channel input black/white points, mid-tone
/// gamma, and output remapping. All defaults are the identity.
public final class LevelsFilter: SinglePassColorFilter, @unchecked Sendable {

    /// Input black point per channel. Default `(0, 0, 0)`.
    public var inputMin: SIMD3<Float> = SIMD3(0, 0, 0)
    /// Input white point per channel. Default `(1, 1, 1)`.
    public var inputMax: SIMD3<Float> = SIMD3(1, 1, 1)
    /// Output black point per channel. Default `(0, 0, 0)`.
    public var outputMin: SIMD3<Float> = SIMD3(0, 0, 0)
    /// Output white point per channel. Default `(1, 1, 1)`.
    public var outputMax: SIMD3<Float> = SIMD3(1, 1, 1)
    /// Mid-tone gamma per channel. Default `(1, 1, 1)`.
    public var gamma: SIMD3<Float> = SIMD3(1, 1, 1)

    public init(engine: MetalForgeEngine) throws {
        try super.init(engine: engine, kernel: "levelsKernel", label: "LevelsFilter")
    }

    override func setUniforms(on encoder: MTLComputeCommandEncoder) {
        var u = LevelsUniforms(
            inMin: inputMin, inMax: inputMax,
            outMin: outputMin, outMax: outputMax,
            gamma: simd_clamp(gamma, SIMD3(repeating: 0.05), SIMD3(repeating: 5))
        )
        encoder.setBytes(&u, length: MemoryLayout<LevelsUniforms>.stride, index: 0)
    }
}

// MARK: - HueRotateFilter

/// Rotates every hue around the colour wheel (YIQ chroma-plane rotation),
/// preserving luminance. `angleDegrees = 0` is the identity.
public final class HueRotateFilter: SinglePassColorFilter, @unchecked Sendable {

    /// Rotation in degrees. Not clamped — wraps naturally every 360°. Default `90`.
    public var angleDegrees: Float = 90.0

    public init(engine: MetalForgeEngine) throws {
        try super.init(engine: engine, kernel: "hueRotateKernel", label: "HueRotateFilter")
    }

    override func setUniforms(on encoder: MTLComputeCommandEncoder) {
        var radians = angleDegrees * .pi / 180
        encoder.setBytes(&radians, length: MemoryLayout<Float>.stride, index: 0)
    }
}

// MARK: - VibranceFilter

/// Smart saturation: boosts muted colours much more than already-vivid ones,
/// so skin and saturated subjects don't blow out.
public final class VibranceFilter: SinglePassColorFilter, @unchecked Sendable {

    /// Vibrance amount. Clamped to `[-1.2, 1.2]`; `0` is the identity. Default `0.5`.
    public var vibrance: Float = 0.5

    public init(engine: MetalForgeEngine) throws {
        try super.init(engine: engine, kernel: "vibranceKernel", label: "VibranceFilter")
    }

    override func setUniforms(on encoder: MTLComputeCommandEncoder) {
        var v = min(max(vibrance, -1.2), 1.2)
        encoder.setBytes(&v, length: MemoryLayout<Float>.stride, index: 0)
    }
}

// MARK: - WhiteBalanceFilter

private struct WhiteBalanceUniforms {
    var temperature: Float
    var tint:        Float
}

/// Temperature (blue ↔ amber) and tint (green ↔ magenta) balance.
/// Both `0` = identity.
public final class WhiteBalanceFilter: SinglePassColorFilter, @unchecked Sendable {

    /// `-1` (cool) … `+1` (warm). Clamped. Default `0`.
    public var temperature: Float = 0.0

    /// `-1` (green) … `+1` (magenta). Clamped. Default `0`.
    public var tint: Float = 0.0

    public init(engine: MetalForgeEngine) throws {
        try super.init(engine: engine, kernel: "whiteBalanceKernel", label: "WhiteBalanceFilter")
    }

    override func setUniforms(on encoder: MTLComputeCommandEncoder) {
        var u = WhiteBalanceUniforms(
            temperature: min(max(temperature, -1), 1),
            tint:        min(max(tint, -1), 1)
        )
        encoder.setBytes(&u, length: MemoryLayout<WhiteBalanceUniforms>.stride, index: 0)
    }
}

// MARK: - ToneCurveFilter

/// RGB tone curves driven by control points, like the Curves tool in photo
/// editors. Each curve is a list of `(in, out)` points in `[0, 1]`; a
/// Catmull-Rom spline through them is baked into a 256-entry LUT on the CPU
/// whenever the points change, then sampled per pixel on the GPU.
///
/// The per-channel curves (`redPoints` / `greenPoints` / `bluePoints`) are
/// applied first, then the composite `rgbPoints` curve on top — matching
/// the order used by GPUImage and Photoshop.
// @unchecked Sendable: curve points + the lazily rebuilt LUT texture are
// guarded by `lock`; configuration is written between frames per the
// MetalForge filter contract.
public final class ToneCurveFilter: @unchecked Sendable, MetalForgeFilter {

    /// Composite curve applied to all three channels. Default identity.
    public var rgbPoints: [SIMD2<Float>] = [SIMD2(0, 0), SIMD2(1, 1)] {
        didSet { markDirty() }
    }
    /// Red-channel curve. Default identity.
    public var redPoints: [SIMD2<Float>] = [SIMD2(0, 0), SIMD2(1, 1)] {
        didSet { markDirty() }
    }
    /// Green-channel curve. Default identity.
    public var greenPoints: [SIMD2<Float>] = [SIMD2(0, 0), SIMD2(1, 1)] {
        didSet { markDirty() }
    }
    /// Blue-channel curve. Default identity.
    public var bluePoints: [SIMD2<Float>] = [SIMD2(0, 0), SIMD2(1, 1)] {
        didSet { markDirty() }
    }

    private let device: MTLDevice
    private let sdrPSO: MTLComputePipelineState
    private let hdrPSO: MTLComputePipelineState

    private let lock = NSLock()
    private var curveTexture: MTLTexture?
    private var dirty = true

    public init(engine: MetalForgeEngine) throws {
        device = engine.device
        sdrPSO = try makeColorAdjustmentPSO(engine: engine, kernel: "toneCurveKernel", isHDR: false)
        hdrPSO = try makeColorAdjustmentPSO(engine: engine, kernel: "toneCurveKernel", isHDR: true)
    }

    private func markDirty() {
        lock.lock(); dirty = true; lock.unlock()
    }

    // MARK: Curve baking

    /// Evaluate a Catmull-Rom spline through `points` at 256 evenly spaced
    /// inputs. Points are sorted by x; outside the first/last point the curve
    /// extends flat. Two points degrade gracefully to a straight line.
    static func bakeCurve(_ points: [SIMD2<Float>]) -> [Float] {
        let sorted = points.sorted { $0.x < $1.x }
        guard sorted.count >= 2 else {
            return (0..<256).map { Float($0) / 255.0 }
        }

        func sample(_ x: Float) -> Float {
            if x <= sorted.first!.x { return sorted.first!.y }
            if x >= sorted.last!.x  { return sorted.last!.y }
            // Find the segment containing x.
            var k = 0
            while k < sorted.count - 2 && sorted[k + 1].x < x { k += 1 }
            let p1 = sorted[k], p2 = sorted[k + 1]
            // Linear extrapolation for the phantom endpoints keeps a 2-point
            // curve exactly straight (duplicating endpoints would bow it).
            let p0 = k > 0 ? sorted[k - 1] : (2 * p1 - p2)
            let p3 = k + 2 < sorted.count ? sorted[k + 2] : (2 * p2 - p1)
            let span = max(p2.x - p1.x, 1e-5)
            let t = (x - p1.x) / span
            let t2 = t * t, t3 = t2 * t
            // Standard uniform Catmull-Rom basis on the y values.
            let y = 0.5 * ((2 * p1.y)
                + (-p0.y + p2.y) * t
                + (2 * p0.y - 5 * p1.y + 4 * p2.y - p3.y) * t2
                + (-p0.y + 3 * p1.y - 3 * p2.y + p3.y) * t3)
            return min(max(y, 0), 1)
        }

        return (0..<256).map { sample(Float($0) / 255.0) }
    }

    private func rebuildCurveTexture() -> MTLTexture? {
        let composite = Self.bakeCurve(rgbPoints)
        let red   = Self.bakeCurve(redPoints)
        let green = Self.bakeCurve(greenPoints)
        let blue  = Self.bakeCurve(bluePoints)

        var bytes = [UInt8](repeating: 255, count: 256 * 4)
        for i in 0..<256 {
            // Channel curve first, composite on top: out = rgb(channel(x)).
            func through(_ channel: [Float]) -> UInt8 {
                let v = composite[Int((channel[i] * 255.0).rounded())]
                return UInt8((min(max(v, 0), 1) * 255.0).rounded())
            }
            bytes[i * 4 + 0] = through(red)
            bytes[i * 4 + 1] = through(green)
            bytes[i * 4 + 2] = through(blue)
        }

        let texture: MTLTexture?
        if let existing = curveTexture {
            texture = existing
        } else {
            let desc = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: .rgba8Unorm, width: 256, height: 1, mipmapped: false
            )
            #if os(macOS)
            desc.storageMode = .managed
            #else
            desc.storageMode = .shared
            #endif
            desc.usage = .shaderRead
            texture = device.makeTexture(descriptor: desc)
        }
        texture?.replace(
            region: MTLRegionMake2D(0, 0, 256, 1),
            mipmapLevel: 0, withBytes: bytes, bytesPerRow: 256 * 4
        )
        return texture
    }

    public func encode(
        source: MTLTexture,
        destination: MTLTexture,
        commandBuffer: MTLCommandBuffer
    ) {
        lock.lock()
        if dirty {
            curveTexture = rebuildCurveTexture()
            dirty = false
        }
        let curve = curveTexture
        lock.unlock()

        guard let curve else { return }
        let pso = (source.pixelFormat == .rgba16Float) ? hdrPSO : sdrPSO

        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "ToneCurveFilter"
        encoder.setComputePipelineState(pso)
        encoder.setTexture(source,      index: 0)
        encoder.setTexture(destination, index: 1)
        encoder.setTexture(curve,       index: 2)

        dispatchColorAdjustment(
            encoder: encoder, pso: pso,
            width: destination.width, height: destination.height
        )
        encoder.endEncoding()
    }
}

// MARK: - HighlightShadowFilter

private struct HighlightShadowUniforms {
    var shadows:    Float
    var highlights: Float
}

/// Lifts shadows and/or recovers highlights without touching the midtones.
public final class HighlightShadowFilter: SinglePassColorFilter, @unchecked Sendable {

    /// `0` = identity … `1` = full shadow lift. Clamped. Default `0`.
    public var shadows: Float = 0.0

    /// `1` = identity … `0` = full highlight recovery. Clamped. Default `1`.
    public var highlights: Float = 1.0

    public init(engine: MetalForgeEngine) throws {
        try super.init(engine: engine, kernel: "highlightShadowKernel", label: "HighlightShadowFilter")
    }

    override func setUniforms(on encoder: MTLComputeCommandEncoder) {
        var u = HighlightShadowUniforms(
            shadows:    min(max(shadows, 0), 1),
            highlights: min(max(highlights, 0), 1)
        )
        encoder.setBytes(&u, length: MemoryLayout<HighlightShadowUniforms>.stride, index: 0)
    }
}

// MARK: - HighlightShadowTintFilter

private struct HighlightShadowTintUniforms {
    var shadowTintColor:        SIMD3<Float>
    var shadowTintIntensity:    Float
    var highlightTintColor:     SIMD3<Float>
    var highlightTintIntensity: Float
}

/// Tints the shadow and highlight bands with independent colours — the
/// split-toning look. Intensities of `0` are the identity.
public final class HighlightShadowTintFilter: SinglePassColorFilter, @unchecked Sendable {

    /// Colour pushed into the shadows. Default red.
    public var shadowTintColor: SIMD3<Float> = SIMD3(1, 0, 0)
    /// Shadow tint strength. Clamped to `[0, 1]`. Default `0`.
    public var shadowTintIntensity: Float = 0.0
    /// Colour pushed into the highlights. Default blue.
    public var highlightTintColor: SIMD3<Float> = SIMD3(0, 0, 1)
    /// Highlight tint strength. Clamped to `[0, 1]`. Default `0`.
    public var highlightTintIntensity: Float = 0.0

    public init(engine: MetalForgeEngine) throws {
        try super.init(engine: engine, kernel: "highlightShadowTintKernel", label: "HighlightShadowTintFilter")
    }

    override func setUniforms(on encoder: MTLComputeCommandEncoder) {
        var u = HighlightShadowTintUniforms(
            shadowTintColor:        shadowTintColor,
            shadowTintIntensity:    min(max(shadowTintIntensity, 0), 1),
            highlightTintColor:     highlightTintColor,
            highlightTintIntensity: min(max(highlightTintIntensity, 0), 1)
        )
        encoder.setBytes(&u, length: MemoryLayout<HighlightShadowTintUniforms>.stride, index: 0)
    }
}

// MARK: - ColorMatrixFilter

private struct ColorMatrixUniforms {
    var matrix:    simd_float4x4
    var intensity: Float
}

/// Multiplies every pixel by an arbitrary 4×4 colour matrix, blended with the
/// original by `intensity`. The identity matrix is the no-op.
public final class ColorMatrixFilter: SinglePassColorFilter, @unchecked Sendable {

    /// The colour matrix applied to `(r, g, b, a)`. Default identity.
    public var matrix: simd_float4x4 = matrix_identity_float4x4

    /// `0` = original … `1` = full matrix. Clamped. Default `1`.
    public var intensity: Float = 1.0

    public init(engine: MetalForgeEngine) throws {
        try super.init(engine: engine, kernel: "colorMatrixKernel", label: "ColorMatrixFilter")
    }

    override func setUniforms(on encoder: MTLComputeCommandEncoder) {
        var u = ColorMatrixUniforms(
            matrix:    matrix,
            intensity: min(max(intensity, 0), 1)
        )
        encoder.setBytes(&u, length: MemoryLayout<ColorMatrixUniforms>.stride, index: 0)
    }
}

// MARK: - ColorInvertFilter

/// Photographic negative: `rgb → 1 − rgb`. No parameters.
public final class ColorInvertFilter: SinglePassColorFilter, @unchecked Sendable {

    public init(engine: MetalForgeEngine) throws {
        try super.init(engine: engine, kernel: "colorInvertKernel", label: "ColorInvertFilter")
    }
}

// MARK: - MonochromeFilter

private struct MonochromeUniforms {
    var filterColor: SIMD3<Float>
    var intensity:   Float
}

/// Single-colour monochrome: collapses to luminance, then re-tints through
/// `filterColor` with a soft-light-style response.
public final class MonochromeFilter: SinglePassColorFilter, @unchecked Sendable {

    /// The tint colour. Default a warm sepia-ish tone.
    public var filterColor: SIMD3<Float> = SIMD3(0.6, 0.45, 0.3)

    /// `0` = original … `1` = fully monochrome. Clamped. Default `1`.
    public var intensity: Float = 1.0

    public init(engine: MetalForgeEngine) throws {
        try super.init(engine: engine, kernel: "monochromeKernel", label: "MonochromeFilter")
    }

    override func setUniforms(on encoder: MTLComputeCommandEncoder) {
        var u = MonochromeUniforms(
            filterColor: filterColor,
            intensity:   min(max(intensity, 0), 1)
        )
        encoder.setBytes(&u, length: MemoryLayout<MonochromeUniforms>.stride, index: 0)
    }
}

// MARK: - FalseColorFilter

private struct FalseColorUniforms {
    var firstColor:  SIMD3<Float>; var pad0: Float = 0
    var secondColor: SIMD3<Float>; var pad1: Float = 0
}

/// Maps luminance onto a two-colour gradient (thermal-camera look):
/// black → `firstColor`, white → `secondColor`.
public final class FalseColorFilter: SinglePassColorFilter, @unchecked Sendable {

    /// Colour for darkest pixels. Default dark blue.
    public var firstColor: SIMD3<Float> = SIMD3(0.0, 0.0, 0.5)

    /// Colour for brightest pixels. Default red.
    public var secondColor: SIMD3<Float> = SIMD3(1.0, 0.0, 0.0)

    public init(engine: MetalForgeEngine) throws {
        try super.init(engine: engine, kernel: "falseColorKernel", label: "FalseColorFilter")
    }

    override func setUniforms(on encoder: MTLComputeCommandEncoder) {
        var u = FalseColorUniforms(firstColor: firstColor, secondColor: secondColor)
        encoder.setBytes(&u, length: MemoryLayout<FalseColorUniforms>.stride, index: 0)
    }
}
