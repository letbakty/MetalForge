import Metal
import simd
import Foundation

// ===========================================================================
// BlurMorphologyFilters — v0.7.0 Advanced Blur & Morphology Pack.
//
// Single-pass filters reuse `SinglePassColorFilter`; the separable/multi-pass
// ones (box blur, iOS blur, morphology) manage persistent intermediates via
// `TextureSlot`, reallocated only when the frame size or format changes.
// ===========================================================================

// MARK: - Persistent intermediate helper

/// Thread-safe lazily-(re)allocated private texture matching a reference —
/// the persistent-intermediate pattern shared by every multi-pass filter.
final class TextureSlot: @unchecked Sendable {

    private let lock = NSLock()
    private var texture: MTLTexture?
    private var format: MTLPixelFormat = .invalid
    private var width = 0
    private var height = 0

    func acquire(matching reference: MTLTexture, device: MTLDevice) -> MTLTexture? {
        lock.lock()
        defer { lock.unlock() }
        if texture == nil
            || format != reference.pixelFormat
            || width != reference.width
            || height != reference.height {
            let desc = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: reference.pixelFormat,
                width: reference.width, height: reference.height, mipmapped: false
            )
            desc.storageMode = .private
            desc.usage = [.shaderRead, .shaderWrite]
            texture = device.makeTexture(descriptor: desc)
            format = reference.pixelFormat
            width = reference.width
            height = reference.height
        }
        return texture
    }

    func clear() {
        lock.lock()
        texture = nil
        format = .invalid
        width = 0
        height = 0
        lock.unlock()
    }
}

// MARK: - BoxBlurFilter

/// Separable box blur — the cheapest large-radius blur there is.
// @unchecked Sendable: intermediate via TextureSlot; configuration written
// between frames per the MetalForge filter contract.
public final class BoxBlurFilter: @unchecked Sendable, MetalForgeFilter {

    /// Blur radius in pixels (taps each side). Clamped to `[1, 32]`. Default `4`.
    public var radius: Float = 4.0

    private let device: MTLDevice
    private let hPSO: MTLComputePipelineState
    private let vSdrPSO: MTLComputePipelineState
    private let vHdrPSO: MTLComputePipelineState
    private let slot = TextureSlot()

    public init(engine: MetalForgeEngine) throws {
        device = engine.device
        hPSO    = try makeColorAdjustmentPSO(engine: engine, kernel: "boxBlurHorizontalKernel", isHDR: false)
        vSdrPSO = try makeColorAdjustmentPSO(engine: engine, kernel: "boxBlurVerticalKernel", isHDR: false)
        vHdrPSO = try makeColorAdjustmentPSO(engine: engine, kernel: "boxBlurVerticalKernel", isHDR: true)
    }

    /// Drop the persistent intermediate (e.g. before a resolution change).
    public func clearHistory() { slot.clear() }

    public func encode(
        source: MTLTexture,
        destination: MTLTexture,
        commandBuffer: MTLCommandBuffer
    ) {
        guard let mid = slot.acquire(matching: source, device: device) else { return }
        var r = min(max(radius, 1), 32)

        if let enc = commandBuffer.makeComputeCommandEncoder() {
            enc.label = "BoxBlurFilter.horizontal"
            enc.setComputePipelineState(hPSO)
            enc.setTexture(source, index: 0)
            enc.setTexture(mid,    index: 1)
            enc.setBytes(&r, length: MemoryLayout<Float>.stride, index: 0)
            dispatchColorAdjustment(encoder: enc, pso: hPSO, width: source.width, height: source.height)
            enc.endEncoding()
        }
        let vPSO = (source.pixelFormat == .rgba16Float) ? vHdrPSO : vSdrPSO
        if let enc = commandBuffer.makeComputeCommandEncoder() {
            enc.label = "BoxBlurFilter.vertical"
            enc.setComputePipelineState(vPSO)
            enc.setTexture(mid,         index: 0)
            enc.setTexture(destination, index: 1)
            enc.setBytes(&r, length: MemoryLayout<Float>.stride, index: 0)
            dispatchColorAdjustment(encoder: enc, pso: vPSO, width: destination.width, height: destination.height)
            enc.endEncoding()
        }
    }
}

// MARK: - DirectionalMotionBlurFilter

private struct DirectionalBlurUniforms {
    var angle:  Float
    var radius: Float
}

/// Static directional (line) blur — every pixel is smeared along one angle.
/// For the temporal frame-accumulation effect see `MotionBlurFilter`.
public final class DirectionalMotionBlurFilter: SinglePassColorFilter, @unchecked Sendable {

    /// Streak direction in radians. Default `0` (horizontal).
    public var angle: Float = 0.0

    /// Streak half-length in pixels. Clamped to `[1, 32]`. Default `8`.
    public var radius: Float = 8.0

    public init(engine: MetalForgeEngine) throws {
        try super.init(engine: engine, kernel: "directionalBlurKernel", label: "DirectionalMotionBlurFilter")
    }

    override func setUniforms(on encoder: MTLComputeCommandEncoder) {
        var u = DirectionalBlurUniforms(
            angle:  angle,
            radius: min(max(radius, 1), 32)
        )
        encoder.setBytes(&u, length: MemoryLayout<DirectionalBlurUniforms>.stride, index: 0)
    }
}

// MARK: - ZoomBlurFilter

private struct ZoomBlurUniforms {
    var center:   SIMD2<Float>
    var strength: Float
}

/// Radial streaks toward a centre — the hyperspace-jump look.
public final class ZoomBlurFilter: SinglePassColorFilter, @unchecked Sendable {

    /// Normalised streak origin. Default frame centre.
    public var center: SIMD2<Float> = SIMD2(0.5, 0.5)

    /// Streak strength. Clamped to `[0, 1]`; `0` = identity. Default `0.5`.
    public var strength: Float = 0.5

    public init(engine: MetalForgeEngine) throws {
        try super.init(engine: engine, kernel: "zoomBlurKernel", label: "ZoomBlurFilter")
    }

    override func setUniforms(on encoder: MTLComputeCommandEncoder) {
        var u = ZoomBlurUniforms(
            center:   center,
            strength: min(max(strength, 0), 1)
        )
        encoder.setBytes(&u, length: MemoryLayout<ZoomBlurUniforms>.stride, index: 0)
    }
}

// MARK: - TiltShiftFilter

private struct TiltShiftUniforms {
    var focusCenter: Float
    var focusWidth:  Float
    var falloff:     Float
    var blurRadius:  Float
    var vertical:    Float
}

/// Miniature-faking tilt-shift: a sharp focus band with disc blur ramping up
/// on both sides.
public final class TiltShiftFilter: SinglePassColorFilter, @unchecked Sendable {

    /// Normalised position of the sharp band along the blur axis. Default `0.5`.
    public var focusCenter: Float = 0.5
    /// Half-width of the fully sharp band. Clamped to `[0, 0.5]`. Default `0.15`.
    public var focusWidth: Float = 0.15
    /// Width of the sharp→blurred ramp. Clamped to `[0.01, 0.5]`. Default `0.2`.
    public var falloff: Float = 0.2
    /// Maximum disc-blur radius in pixels. Clamped to `[0, 32]`. Default `8`.
    public var blurRadius: Float = 8.0
    /// `false` = horizontal band (landscape miniatures), `true` = vertical.
    public var isVertical: Bool = false

    public init(engine: MetalForgeEngine) throws {
        try super.init(engine: engine, kernel: "tiltShiftKernel", label: "TiltShiftFilter")
    }

    override func setUniforms(on encoder: MTLComputeCommandEncoder) {
        var u = TiltShiftUniforms(
            focusCenter: min(max(focusCenter, 0), 1),
            focusWidth:  min(max(focusWidth, 0), 0.5),
            falloff:     min(max(falloff, 0.01), 0.5),
            blurRadius:  min(max(blurRadius, 0), 32),
            vertical:    isVertical ? 1 : 0
        )
        encoder.setBytes(&u, length: MemoryLayout<TiltShiftUniforms>.stride, index: 0)
    }
}

// MARK: - BilateralBlurFilter

private struct BilateralUniforms {
    var radius:     Float
    var rangeSigma: Float
}

/// Edge-preserving Gaussian: spatial × colour-distance weights. The classic
/// skin-smoothing blur.
public final class BilateralBlurFilter: SinglePassColorFilter, @unchecked Sendable {

    /// Spatial radius in pixels. Clamped to `[1, 6]` (O(r²) cost). Default `4`.
    public var radius: Float = 4.0

    /// Colour-distance sigma; smaller preserves edges harder.
    /// Clamped to `[0.01, 1]`. Default `0.15`.
    public var rangeSigma: Float = 0.15

    public init(engine: MetalForgeEngine) throws {
        try super.init(engine: engine, kernel: "bilateralBlurKernel", label: "BilateralBlurFilter")
    }

    override func setUniforms(on encoder: MTLComputeCommandEncoder) {
        var u = BilateralUniforms(
            radius:     min(max(radius, 1), 6),
            rangeSigma: min(max(rangeSigma, 0.01), 1)
        )
        encoder.setBytes(&u, length: MemoryLayout<BilateralUniforms>.stride, index: 0)
    }
}

// MARK: - MedianBlurFilter

/// 3×3 median denoise (luminance-ranked, outputs a real input pixel —
/// no invented colours). Kills salt-and-pepper noise. No parameters.
public final class MedianBlurFilter: SinglePassColorFilter, @unchecked Sendable {

    public init(engine: MetalForgeEngine) throws {
        try super.init(engine: engine, kernel: "medianBlurKernel", label: "MedianBlurFilter")
    }
}

// MARK: - LensBlurFilter

private struct LensBlurUniforms {
    var radius:     Float
    var brightness: Float
}

/// Hexagonal-aperture bokeh: 37 taps over 3 hex rings with highlight-weighted
/// accumulation, so bright points bloom into hexagons instead of mushing out.
public final class LensBlurFilter: SinglePassColorFilter, @unchecked Sendable {

    /// Bokeh radius in pixels. Clamped to `[1, 32]`. Default `8`.
    public var radius: Float = 8.0

    /// Highlight-bloom emphasis. Clamped to `[0, 1]`; `0` = plain average.
    /// Default `0.5`.
    public var brightness: Float = 0.5

    public init(engine: MetalForgeEngine) throws {
        try super.init(engine: engine, kernel: "lensBlurKernel", label: "LensBlurFilter")
    }

    override func setUniforms(on encoder: MTLComputeCommandEncoder) {
        var u = LensBlurUniforms(
            radius:     min(max(radius, 1), 32),
            brightness: min(max(brightness, 0), 1)
        )
        encoder.setBytes(&u, length: MemoryLayout<LensBlurUniforms>.stride, index: 0)
    }
}

// MARK: - SurfaceBlurFilter

private struct SurfaceBlurUniforms {
    var radius:    Float
    var threshold: Float
}

/// Photoshop-style surface blur: only neighbours within a luminance threshold
/// participate in the average — texture smooths, hard edges survive intact.
public final class SurfaceBlurFilter: SinglePassColorFilter, @unchecked Sendable {

    /// Box radius in pixels. Clamped to `[1, 6]`. Default `4`.
    public var radius: Float = 4.0

    /// Maximum luminance difference to participate. Clamped to `[0.01, 1]`.
    /// Default `0.15`.
    public var threshold: Float = 0.15

    public init(engine: MetalForgeEngine) throws {
        try super.init(engine: engine, kernel: "surfaceBlurKernel", label: "SurfaceBlurFilter")
    }

    override func setUniforms(on encoder: MTLComputeCommandEncoder) {
        var u = SurfaceBlurUniforms(
            radius:    min(max(radius, 1), 6),
            threshold: min(max(threshold, 0.01), 1)
        )
        encoder.setBytes(&u, length: MemoryLayout<SurfaceBlurUniforms>.stride, index: 0)
    }
}

// MARK: - IOSBlurFilter

private struct FrostedUniforms {
    var saturationBoost: Float
    var whiteMix:        Float
}

/// UIKit-style frosted glass: two stacked box-blur passes (≈ triangle/Gaussian
/// response) followed by re-saturation and a light white mix.
// @unchecked Sendable: intermediates via TextureSlot; configuration written
// between frames per the MetalForge filter contract.
public final class IOSBlurFilter: @unchecked Sendable, MetalForgeFilter {

    /// Blur radius in pixels per box pass. Clamped to `[1, 32]`. Default `12`.
    public var radius: Float = 12.0

    /// Post-blur saturation boost. Clamped to `[1, 3]`. Default `1.8`.
    public var saturationBoost: Float = 1.8

    /// White mixed in for the "light material" look. Clamped to `[0, 0.6]`.
    /// Default `0.15`.
    public var whiteMix: Float = 0.15

    private let device: MTLDevice
    private let hPSO: MTLComputePipelineState
    private let vPSO: MTLComputePipelineState
    private let frostedSdrPSO: MTLComputePipelineState
    private let frostedHdrPSO: MTLComputePipelineState
    private let slotA = TextureSlot()
    private let slotB = TextureSlot()

    public init(engine: MetalForgeEngine) throws {
        device = engine.device
        hPSO = try makeColorAdjustmentPSO(engine: engine, kernel: "boxBlurHorizontalKernel", isHDR: false)
        vPSO = try makeColorAdjustmentPSO(engine: engine, kernel: "boxBlurVerticalKernel", isHDR: false)
        frostedSdrPSO = try makeColorAdjustmentPSO(engine: engine, kernel: "frostedGlassKernel", isHDR: false)
        frostedHdrPSO = try makeColorAdjustmentPSO(engine: engine, kernel: "frostedGlassKernel", isHDR: true)
    }

    /// Drop the persistent intermediates (e.g. before a resolution change).
    public func clearHistory() {
        slotA.clear()
        slotB.clear()
    }

    private func boxPass(
        _ pso: MTLComputePipelineState, label: String,
        from src: MTLTexture, to dst: MTLTexture,
        radius: inout Float, commandBuffer: MTLCommandBuffer
    ) {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.label = label
        enc.setComputePipelineState(pso)
        enc.setTexture(src, index: 0)
        enc.setTexture(dst, index: 1)
        enc.setBytes(&radius, length: MemoryLayout<Float>.stride, index: 0)
        dispatchColorAdjustment(encoder: enc, pso: pso, width: dst.width, height: dst.height)
        enc.endEncoding()
    }

    public func encode(
        source: MTLTexture,
        destination: MTLTexture,
        commandBuffer: MTLCommandBuffer
    ) {
        guard
            let midA = slotA.acquire(matching: source, device: device),
            let midB = slotB.acquire(matching: source, device: device)
        else { return }

        var r = min(max(radius, 1), 32)
        // Two box iterations ≈ triangle filter — visually close to Gaussian
        // at a fraction of the tap count.
        boxPass(hPSO, label: "IOSBlurFilter.h1", from: source, to: midA, radius: &r, commandBuffer: commandBuffer)
        boxPass(vPSO, label: "IOSBlurFilter.v1", from: midA,   to: midB, radius: &r, commandBuffer: commandBuffer)
        boxPass(hPSO, label: "IOSBlurFilter.h2", from: midB,   to: midA, radius: &r, commandBuffer: commandBuffer)
        boxPass(vPSO, label: "IOSBlurFilter.v2", from: midA,   to: midB, radius: &r, commandBuffer: commandBuffer)

        let pso = (source.pixelFormat == .rgba16Float) ? frostedHdrPSO : frostedSdrPSO
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.label = "IOSBlurFilter.frosted"
        enc.setComputePipelineState(pso)
        enc.setTexture(midB,        index: 0)
        enc.setTexture(destination, index: 1)
        var u = FrostedUniforms(
            saturationBoost: min(max(saturationBoost, 1), 3),
            whiteMix:        min(max(whiteMix, 0), 0.6)
        )
        enc.setBytes(&u, length: MemoryLayout<FrostedUniforms>.stride, index: 0)
        dispatchColorAdjustment(encoder: enc, pso: pso, width: destination.width, height: destination.height)
        enc.endEncoding()
    }
}

// MARK: - Morphology

private struct MorphologyUniforms {
    var radius:   Float
    var isDilate: Float
}

/// Shared engine for the four morphological operators: each operation is a
/// separable H+V min/max pass over a square structuring element.
// @unchecked Sendable: intermediates via TextureSlot; configuration written
// between frames per the MetalForge filter contract.
public class MorphologyFilterBase: @unchecked Sendable, MetalForgeFilter {

    /// Structuring-element half-size in pixels. Clamped to `[1, 16]`. Default `2`.
    public var radius: Float = 2.0

    /// The pass sequence: `true` = dilate (max), `false` = erode (min).
    private let operations: [Bool]
    private let label: String

    private let device: MTLDevice
    private let hPSO: MTLComputePipelineState
    private let vSdrPSO: MTLComputePipelineState
    private let vHdrPSO: MTLComputePipelineState
    private let slotA = TextureSlot()
    private let slotB = TextureSlot()

    init(engine: MetalForgeEngine, operations: [Bool], label: String) throws {
        self.operations = operations
        self.label = label
        device = engine.device
        hPSO    = try makeColorAdjustmentPSO(engine: engine, kernel: "morphologyHorizontalKernel", isHDR: false)
        vSdrPSO = try makeColorAdjustmentPSO(engine: engine, kernel: "morphologyVerticalKernel", isHDR: false)
        vHdrPSO = try makeColorAdjustmentPSO(engine: engine, kernel: "morphologyVerticalKernel", isHDR: true)
    }

    /// Drop the persistent intermediates (e.g. before a resolution change).
    public func clearHistory() {
        slotA.clear()
        slotB.clear()
    }

    public func encode(
        source: MTLTexture,
        destination: MTLTexture,
        commandBuffer: MTLCommandBuffer
    ) {
        guard
            let midA = slotA.acquire(matching: source, device: device),
            let midB = slotB.acquire(matching: source, device: device)
        else { return }

        let vPSO = (source.pixelFormat == .rgba16Float) ? vHdrPSO : vSdrPSO
        var input: MTLTexture = source

        for (index, isDilate) in operations.enumerated() {
            let isLast = index == operations.count - 1
            let output: MTLTexture = isLast ? destination : (input === midB ? midA : midB)
            var u = MorphologyUniforms(
                radius:   min(max(radius, 1), 16),
                isDilate: isDilate ? 1 : 0
            )

            if let enc = commandBuffer.makeComputeCommandEncoder() {
                enc.label = "\(label).h\(index)"
                enc.setComputePipelineState(hPSO)
                enc.setTexture(input, index: 0)
                enc.setTexture(midA === output ? midB : midA, index: 1)
                enc.setBytes(&u, length: MemoryLayout<MorphologyUniforms>.stride, index: 0)
                dispatchColorAdjustment(encoder: enc, pso: hPSO, width: source.width, height: source.height)
                enc.endEncoding()
            }
            if let enc = commandBuffer.makeComputeCommandEncoder() {
                enc.label = "\(label).v\(index)"
                enc.setComputePipelineState(vPSO)
                enc.setTexture(midA === output ? midB : midA, index: 0)
                enc.setTexture(output, index: 1)
                enc.setBytes(&u, length: MemoryLayout<MorphologyUniforms>.stride, index: 0)
                dispatchColorAdjustment(encoder: enc, pso: vPSO, width: output.width, height: output.height)
                enc.endEncoding()
            }
            input = output
        }
    }
}

/// Grows bright regions (per-channel max over the structuring element).
public final class DilationFilter: MorphologyFilterBase, @unchecked Sendable {
    public init(engine: MetalForgeEngine) throws {
        try super.init(engine: engine, operations: [true], label: "DilationFilter")
    }
}

/// Shrinks bright regions (per-channel min over the structuring element).
public final class ErosionFilter: MorphologyFilterBase, @unchecked Sendable {
    public init(engine: MetalForgeEngine) throws {
        try super.init(engine: engine, operations: [false], label: "ErosionFilter")
    }
}

/// Erosion followed by dilation — removes bright specks smaller than the
/// structuring element while preserving larger shapes.
public final class OpeningFilter: MorphologyFilterBase, @unchecked Sendable {
    public init(engine: MetalForgeEngine) throws {
        try super.init(engine: engine, operations: [false, true], label: "OpeningFilter")
    }
}

/// Dilation followed by erosion — fills dark holes smaller than the
/// structuring element while preserving larger shapes.
public final class ClosingFilter: MorphologyFilterBase, @unchecked Sendable {
    public init(engine: MetalForgeEngine) throws {
        try super.init(engine: engine, operations: [true, false], label: "ClosingFilter")
    }
}
