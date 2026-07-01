import Metal
import simd
import Foundation

// ===========================================================================
// TemporalPackFilters — v0.8.0 Temporal Video Effects Pack.
//
// Six frame-history filters sharing TemporalPackKernels.metal. They all
// follow the proven MotionBlurFilter lifecycle (seed on first frame, blit
// the chosen result back into the history texture), factored into
// `TemporalHistoryFilterBase`:
//
//   FrameBlendFilter         decaying max-blend light trails   (saves output)
//   LowPassFilter            EMA temporal smoothing            (saves output)
//   HighPassFilter           |frame − EMA| residual            (saves EMA)
//   MotionDetectorFilter     thresholded difference overlay    (saves input)
//   OpticalFlowWarpFilter    gradient-flow warp                (saves input)
//   FrameInterpolationFilter phase crossfade between frames    (saves input)
// ===========================================================================

// MARK: - Shared base

/// What gets blitted into the history texture after the compute pass.
enum TemporalHistorySource {
    case input         // previous *frame* semantics
    case output        // accumulation-buffer semantics
    case auxiliary     // kernel writes the history update to texture(3)
}

/// Common plumbing for one-history-texture temporal filters: persistent
/// buffer allocation, first-frame seeding, uniform upload hook, and the
/// history save blit.
// @unchecked Sendable: history guarded by `bufferLock`; configuration is
// written between frames per the MetalForge filter contract.
public class TemporalHistoryFilterBase: @unchecked Sendable, MetalForgeFilter {

    private let label: String
    private let historySource: TemporalHistorySource
    private let device: MTLDevice
    private let sdrPSO: MTLComputePipelineState
    private let hdrPSO: MTLComputePipelineState

    private let bufferLock = NSLock()
    private var history: MTLTexture?
    private var auxiliary: MTLTexture?
    private var historyFormat: MTLPixelFormat = .invalid
    private var historyWidth = 0
    private var historyHeight = 0

    init(
        engine: MetalForgeEngine,
        kernel: String,
        label: String,
        historySource: TemporalHistorySource
    ) throws {
        self.label = label
        self.historySource = historySource
        self.device = engine.device
        self.sdrPSO = try makeColorAdjustmentPSO(engine: engine, kernel: kernel, isHDR: false)
        self.hdrPSO = try makeColorAdjustmentPSO(engine: engine, kernel: kernel, isHDR: true)
    }

    /// Subclass hook: upload the uniform buffer at index 0.
    func setUniforms(on encoder: MTLComputeCommandEncoder) {}

    /// Drop the history buffer; the next frame re-seeds (output == identity
    /// behaviour for that single frame). Call when the user switches filters
    /// or the capture format changes. Safe from any thread.
    public func clearHistory() {
        bufferLock.lock()
        history = nil
        auxiliary = nil
        historyFormat = .invalid
        historyWidth = 0
        historyHeight = 0
        bufferLock.unlock()
    }

    private func makeBuffer(matching reference: MTLTexture) -> MTLTexture? {
        let desc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: reference.pixelFormat,
            width: reference.width, height: reference.height, mipmapped: false
        )
        desc.storageMode = .private
        desc.usage = [.shaderRead, .shaderWrite]
        return device.makeTexture(descriptor: desc)
    }

    public func encode(
        source: MTLTexture,
        destination: MTLTexture,
        commandBuffer: MTLCommandBuffer
    ) {
        // ----- Acquire / re-acquire history under the lock -----
        bufferLock.lock()
        let needsSeed =
            history == nil
            || historyFormat != source.pixelFormat
            || historyWidth != source.width
            || historyHeight != source.height
        if needsSeed {
            history = makeBuffer(matching: source)
            auxiliary = (historySource == .auxiliary) ? makeBuffer(matching: source) : nil
            historyFormat = source.pixelFormat
            historyWidth = source.width
            historyHeight = source.height
        }
        let prev = history
        let aux = auxiliary
        bufferLock.unlock()

        guard let prev else { return }
        if historySource == .auxiliary && aux == nil { return }

        // ----- First-frame seed: history starts as a copy of the source -----
        if needsSeed, let blit = commandBuffer.makeBlitCommandEncoder() {
            blit.label = "\(label).seedHistory"
            blit.copy(from: source, to: prev)
            blit.endEncoding()
        }

        // ----- Compute pass -----
        let pso = (source.pixelFormat == .rgba16Float) ? hdrPSO : sdrPSO
        if let enc = commandBuffer.makeComputeCommandEncoder() {
            enc.label = label
            enc.setComputePipelineState(pso)
            enc.setTexture(source,      index: 0)
            enc.setTexture(prev,        index: 1)
            enc.setTexture(destination, index: 2)
            if let aux { enc.setTexture(aux, index: 3) }
            setUniforms(on: enc)
            dispatchColorAdjustment(
                encoder: enc, pso: pso,
                width: destination.width, height: destination.height
            )
            enc.endEncoding()
        }

        // ----- Save the configured result into history for the next frame -----
        if let blit = commandBuffer.makeBlitCommandEncoder() {
            blit.label = "\(label).saveHistory"
            switch historySource {
            case .input:     blit.copy(from: source,      to: prev)
            case .output:    blit.copy(from: destination, to: prev)
            case .auxiliary: if let aux { blit.copy(from: aux, to: prev) }
            }
            blit.endEncoding()
        }
    }
}

// MARK: - FrameBlendFilter

/// Long-exposure light trails: bright content persists and decays over
/// `decay`, dark content refreshes instantly.
public final class FrameBlendFilter: TemporalHistoryFilterBase, @unchecked Sendable {

    /// Per-frame trail multiplier. Clamped to `[0.5, 0.995]`; higher = longer
    /// trails. Default `0.92`.
    public var decay: Float = 0.92

    public init(engine: MetalForgeEngine) throws {
        try super.init(
            engine: engine, kernel: "frameBlendKernel",
            label: "FrameBlendFilter", historySource: .output
        )
    }

    override func setUniforms(on encoder: MTLComputeCommandEncoder) {
        var d = min(max(decay, 0.5), 0.995)
        encoder.setBytes(&d, length: MemoryLayout<Float>.stride, index: 0)
    }
}

// MARK: - LowPassFilter

/// Temporal exponential moving average — smooths sensor noise and flicker at
/// the cost of motion ghosting.
public final class LowPassFilter: TemporalHistoryFilterBase, @unchecked Sendable {

    /// Smoothing strength: `0` = pass-through, `→1` = very slow average.
    /// Clamped to `[0, 0.98]`. Default `0.5`.
    public var strength: Float = 0.5

    public init(engine: MetalForgeEngine) throws {
        try super.init(
            engine: engine, kernel: "lowPassKernel",
            label: "LowPassFilter", historySource: .output
        )
    }

    override func setUniforms(on encoder: MTLComputeCommandEncoder) {
        var s = min(max(strength, 0), 0.98)
        encoder.setBytes(&s, length: MemoryLayout<Float>.stride, index: 0)
    }
}

// MARK: - HighPassFilter

/// Temporal high-pass: shows `|frame − EMA|`, i.e. only what *changes*.
/// Static scenery goes black; motion shows as ghostly residual.
public final class HighPassFilter: TemporalHistoryFilterBase, @unchecked Sendable {

    /// EMA persistence (same meaning as `LowPassFilter.strength`).
    /// Clamped to `[0, 0.98]`. Default `0.5`.
    public var strength: Float = 0.5

    public init(engine: MetalForgeEngine) throws {
        try super.init(
            engine: engine, kernel: "highPassKernel",
            label: "HighPassFilter", historySource: .auxiliary
        )
    }

    override func setUniforms(on encoder: MTLComputeCommandEncoder) {
        var s = min(max(strength, 0), 0.98)
        encoder.setBytes(&s, length: MemoryLayout<Float>.stride, index: 0)
    }
}

// MARK: - MotionDetectorFilter

private struct MotionDetectorUniforms {
    var highlightColor: SIMD3<Float>
    var threshold:      Float
    var dimming:        Float
    var pad0: Float = 0
    var pad1: Float = 0
    var pad2: Float = 0
}

/// Paints moving pixels with a highlight colour over a dimmed scene —
/// a security-camera-style motion visualiser.
public final class MotionDetectorFilter: TemporalHistoryFilterBase, @unchecked Sendable {

    /// Colour for moving regions. Default bright red.
    public var highlightColor: SIMD3<Float> = SIMD3(1, 0.1, 0.1)

    /// Frame-difference threshold. Clamped to `[0, 1]`. Default `0.1`.
    public var threshold: Float = 0.1

    /// Brightness of static scenery. Clamped to `[0, 1]`. Default `0.4`.
    public var dimming: Float = 0.4

    public init(engine: MetalForgeEngine) throws {
        try super.init(
            engine: engine, kernel: "motionDetectorKernel",
            label: "MotionDetectorFilter", historySource: .input
        )
    }

    override func setUniforms(on encoder: MTLComputeCommandEncoder) {
        var u = MotionDetectorUniforms(
            highlightColor: highlightColor,
            threshold:      min(max(threshold, 0), 1),
            dimming:        min(max(dimming, 0), 1)
        )
        encoder.setBytes(&u, length: MemoryLayout<MotionDetectorUniforms>.stride, index: 0)
    }
}

// MARK: - OpticalFlowWarpFilter

private struct FlowWarpUniforms {
    var strength:  Float
    var smoothing: Float
}

/// Warps the frame along a single-iteration gradient-based optical-flow
/// estimate — moving regions liquefy and smear, static regions stay put.
/// (A stylistic effect, not a metrology-grade flow field.)
public final class OpticalFlowWarpFilter: TemporalHistoryFilterBase, @unchecked Sendable {

    /// Warp distance multiplier in pixels. Clamped to `[0, 64]`. Default `8`.
    public var strength: Float = 8.0

    /// Per-axis flow clamp in pixels — limits how wild the warp gets.
    /// Clamped to `[0.1, 8]`. Default `2`.
    public var smoothing: Float = 2.0

    public init(engine: MetalForgeEngine) throws {
        try super.init(
            engine: engine, kernel: "opticalFlowWarpKernel",
            label: "OpticalFlowWarpFilter", historySource: .input
        )
    }

    override func setUniforms(on encoder: MTLComputeCommandEncoder) {
        var u = FlowWarpUniforms(
            strength:  min(max(strength, 0), 64),
            smoothing: min(max(smoothing, 0.1), 8)
        )
        encoder.setBytes(&u, length: MemoryLayout<FlowWarpUniforms>.stride, index: 0)
    }
}

// MARK: - FrameInterpolationFilter

/// Crossfades between the previous and current frame at `phase` — a temporal
/// resampling primitive (e.g. simulate shutter lag or sub-frame timing).
/// This is a blend interpolator, not motion-compensated synthesis.
public final class FrameInterpolationFilter: TemporalHistoryFilterBase, @unchecked Sendable {

    /// `0` = previous frame … `1` = current frame. Clamped. Default `0.5`.
    public var phase: Float = 0.5

    public init(engine: MetalForgeEngine) throws {
        try super.init(
            engine: engine, kernel: "frameInterpolationKernel",
            label: "FrameInterpolationFilter", historySource: .input
        )
    }

    override func setUniforms(on encoder: MTLComputeCommandEncoder) {
        var p = min(max(phase, 0), 1)
        encoder.setBytes(&p, length: MemoryLayout<Float>.stride, index: 0)
    }
}
