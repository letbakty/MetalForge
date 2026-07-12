import Metal
import simd
import Foundation

// ===========================================================================
// EdgeDetectionFilters — v0.5.0 Edge Detection & Convolution Pack.
//
// Seven single-pass filters reuse the `SinglePassColorFilter` base; Canny is
// two-pass with a persistent intermediate (same pattern as GaussianBlur).
// ===========================================================================

// MARK: - SobelEdgeDetectionFilter

/// Sobel gradient magnitude on luminance — white edges on black.
public final class SobelEdgeDetectionFilter: SinglePassColorFilter, @unchecked Sendable {

    /// Output gain. Clamped to `[0, 8]`. Default `1`.
    public var edgeStrength: Float = 1.0

    public init(engine: MetalForgeEngine) throws {
        try super.init(engine: engine, kernel: "sobelKernel", label: "SobelEdgeDetectionFilter")
    }

    override func setUniforms(on encoder: MTLComputeCommandEncoder) {
        var s = min(max(edgeStrength, 0), 8)
        encoder.setBytes(&s, length: MemoryLayout<Float>.stride, index: 0)
    }
}

// MARK: - PrewittEdgeDetectionFilter

/// Prewitt gradient magnitude — like Sobel with uniform row weighting,
/// slightly less centre-biased.
public final class PrewittEdgeDetectionFilter: SinglePassColorFilter, @unchecked Sendable {

    /// Output gain. Clamped to `[0, 8]`. Default `1`.
    public var edgeStrength: Float = 1.0

    public init(engine: MetalForgeEngine) throws {
        try super.init(engine: engine, kernel: "prewittKernel", label: "PrewittEdgeDetectionFilter")
    }

    override func setUniforms(on encoder: MTLComputeCommandEncoder) {
        var s = min(max(edgeStrength, 0), 8)
        encoder.setBytes(&s, length: MemoryLayout<Float>.stride, index: 0)
    }
}

// MARK: - Convolution3x3Filter

private struct Convolution3x3Uniforms {
    var kernelMatrix: simd_float3x3
    var bias:         Float
}

/// Arbitrary 3×3 convolution over RGB. `matrix` columns are kernel columns
/// (left→right), rows top→bottom. The identity kernel is the no-op.
public final class Convolution3x3Filter: SinglePassColorFilter, @unchecked Sendable {

    /// The convolution kernel. Default identity (centre 1).
    public var matrix: simd_float3x3 = simd_float3x3(
        SIMD3(0, 0, 0),
        SIMD3(0, 1, 0),
        SIMD3(0, 0, 0)
    )

    /// Added to the convolution result (use `0.5` for emboss-style kernels).
    /// Default `0`.
    public var bias: Float = 0.0

    public init(engine: MetalForgeEngine) throws {
        try super.init(engine: engine, kernel: "convolution3x3Kernel", label: "Convolution3x3Filter")
    }

    override func setUniforms(on encoder: MTLComputeCommandEncoder) {
        var u = Convolution3x3Uniforms(kernelMatrix: matrix, bias: bias)
        encoder.setBytes(&u, length: MemoryLayout<Convolution3x3Uniforms>.stride, index: 0)
    }
}

// MARK: - EmbossFilter

/// Classic emboss: a directional convolution biased to mid-grey, producing a
/// chiselled-relief look.
public final class EmbossFilter: SinglePassColorFilter, @unchecked Sendable {

    /// Relief strength. Clamped to `[0, 4]`. Default `1`.
    public var intensity: Float = 1.0

    public init(engine: MetalForgeEngine) throws {
        try super.init(engine: engine, kernel: "convolution3x3Kernel", label: "EmbossFilter")
    }

    override func setUniforms(on encoder: MTLComputeCommandEncoder) {
        let i = min(max(intensity, 0), 4)
        // Top-left → bottom-right emboss kernel scaled by intensity.
        // Columns left→right; rows top→bottom.
        var u = Convolution3x3Uniforms(
            kernelMatrix: simd_float3x3(
                SIMD3(-2 * i, -i, 0),
                SIMD3(-i, 1, i),
                SIMD3(0, i, 2 * i)
            ),
            bias: 0.5 * min(i, 1)
        )
        encoder.setBytes(&u, length: MemoryLayout<Convolution3x3Uniforms>.stride, index: 0)
    }
}

// MARK: - LaplacianFilter

/// 8-neighbour Laplacian on luminance, biased to mid-grey so both edge
/// polarities are visible.
public final class LaplacianFilter: SinglePassColorFilter, @unchecked Sendable {

    /// Output gain. Clamped to `[0, 8]`. Default `1`.
    public var strength: Float = 1.0

    public init(engine: MetalForgeEngine) throws {
        try super.init(engine: engine, kernel: "laplacianKernel", label: "LaplacianFilter")
    }

    override func setUniforms(on encoder: MTLComputeCommandEncoder) {
        var s = min(max(strength, 0), 8)
        encoder.setBytes(&s, length: MemoryLayout<Float>.stride, index: 0)
    }
}

// MARK: - CannyEdgeDetectionFilter

private struct CannyUniforms {
    var lowerThreshold: Float
    var upperThreshold: Float
}

/// Two-pass Canny edge detection: smoothed Sobel gradient + direction, then
/// non-maximum suppression along the gradient with a double threshold.
/// (The textbook hysteresis-connectivity walk is omitted — strong edges are
/// white, weak-but-plausible edges mid-grey.)
///
/// The intermediate gradient texture is persistent and reallocated only on
/// size/format change — no per-frame allocation.
// @unchecked Sendable: intermediate guarded by `bufferLock`; configuration
// written between frames per the MetalForge filter contract.
public final class CannyEdgeDetectionFilter: @unchecked Sendable, MetalForgeFilter {

    /// Gradient magnitude below this is never an edge. Clamped `[0, 1]`. Default `0.1`.
    public var lowerThreshold: Float = 0.1

    /// Gradient magnitude above this is always an edge. Clamped `[0, 1]`. Default `0.3`.
    public var upperThreshold: Float = 0.3

    private let device: MTLDevice
    private let gradientPSO: MTLComputePipelineState
    private let suppressSdrPSO: MTLComputePipelineState
    private let suppressHdrPSO: MTLComputePipelineState

    private let bufferLock = NSLock()
    private var intermediate: MTLTexture?
    private var intermediateWidth = 0
    private var intermediateHeight = 0

    public init(engine: MetalForgeEngine) throws {
        device = engine.device
        gradientPSO    = try makeColorAdjustmentPSO(engine: engine, kernel: "cannyGradientKernel", isHDR: false)
        suppressSdrPSO = try makeColorAdjustmentPSO(engine: engine, kernel: "cannySuppressKernel", isHDR: false)
        suppressHdrPSO = try makeColorAdjustmentPSO(engine: engine, kernel: "cannySuppressKernel", isHDR: true)
    }

    /// Drop the persistent gradient buffer (e.g. before a resolution change).
    public func clearHistory() {
        bufferLock.lock()
        intermediate = nil
        intermediateWidth = 0
        intermediateHeight = 0
        bufferLock.unlock()
    }

    public func encode(
        source: MTLTexture,
        destination: MTLTexture,
        commandBuffer: MTLCommandBuffer
    ) {
        bufferLock.lock()
        if intermediate == nil
            || intermediateWidth != source.width
            || intermediateHeight != source.height {
            // The gradient pack (mag + dir) always fits in rgba8.
            let desc = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: .rgba8Unorm,
                width: source.width, height: source.height, mipmapped: false
            )
            desc.storageMode = .private
            desc.usage = [.shaderRead, .shaderWrite]
            intermediate = device.makeTexture(descriptor: desc)
            intermediateWidth = source.width
            intermediateHeight = source.height
        }
        let mid = intermediate
        bufferLock.unlock()

        guard let mid else { return }

        // ----- Pass 1: gradient magnitude + direction -----
        if let enc = commandBuffer.makeComputeCommandEncoder() {
            enc.label = "CannyEdgeDetectionFilter.gradient"
            enc.setComputePipelineState(gradientPSO)
            enc.setTexture(source, index: 0)
            enc.setTexture(mid,    index: 1)
            dispatchColorAdjustment(
                encoder: enc, pso: gradientPSO,
                width: source.width, height: source.height
            )
            enc.endEncoding()
        }

        // ----- Pass 2: NMS + double threshold -----
        let pso = (destination.pixelFormat == .rgba16Float) ? suppressHdrPSO : suppressSdrPSO
        if let enc = commandBuffer.makeComputeCommandEncoder() {
            enc.label = "CannyEdgeDetectionFilter.suppress"
            enc.setComputePipelineState(pso)
            enc.setTexture(mid,         index: 0)
            enc.setTexture(destination, index: 1)
            var u = CannyUniforms(
                lowerThreshold: min(max(lowerThreshold, 0), 1),
                upperThreshold: min(max(upperThreshold, 0), 1)
            )
            enc.setBytes(&u, length: MemoryLayout<CannyUniforms>.stride, index: 0)
            dispatchColorAdjustment(
                encoder: enc, pso: pso,
                width: destination.width, height: destination.height
            )
            enc.endEncoding()
        }
    }
}

// MARK: - HarrisCornerDetectionFilter

private struct HarrisUniforms {
    var sensitivity: Float
    var threshold:   Float
}

/// Harris corner response over a 5×5 structure-tensor window — white dots at
/// detected corners on black.
public final class HarrisCornerDetectionFilter: SinglePassColorFilter, @unchecked Sendable {

    /// Harris `k`. Clamped to `[0.01, 0.5]`. Default `0.04`.
    public var sensitivity: Float = 0.04

    /// Response cut-off. Clamped to `[0, 1]`. Default `0.002`.
    public var threshold: Float = 0.002

    public init(engine: MetalForgeEngine) throws {
        try super.init(engine: engine, kernel: "harrisCornerKernel", label: "HarrisCornerDetectionFilter")
    }

    override func setUniforms(on encoder: MTLComputeCommandEncoder) {
        var u = HarrisUniforms(
            sensitivity: min(max(sensitivity, 0.01), 0.5),
            threshold:   min(max(threshold, 0), 1)
        )
        encoder.setBytes(&u, length: MemoryLayout<HarrisUniforms>.stride, index: 0)
    }
}

// MARK: - NonMaximumSuppressionFilter

/// Keeps a pixel only when its luminance is the strict maximum of its 3×3
/// neighbourhood (and above `threshold`); everything else goes black.
/// Typically chained after an edge or corner detector to thin responses.
public final class NonMaximumSuppressionFilter: SinglePassColorFilter, @unchecked Sendable {

    /// Minimum luminance for a pixel to survive at all. Clamped `[0, 1]`.
    /// Default `0.05`.
    public var threshold: Float = 0.05

    public init(engine: MetalForgeEngine) throws {
        try super.init(engine: engine, kernel: "nonMaximumSuppressionKernel", label: "NonMaximumSuppressionFilter")
    }

    override func setUniforms(on encoder: MTLComputeCommandEncoder) {
        var t = min(max(threshold, 0), 1)
        encoder.setBytes(&t, length: MemoryLayout<Float>.stride, index: 0)
    }
}
