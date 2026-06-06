import Metal
import simd
import Foundation

// ===========================================================================
// BlurFilters — GaussianBlurFilter (separable, two-pass) + SharpenFilter.
//
// Both follow the established MetalForge pattern: pre-compiled, isHDR-specialised
// MTLComputePipelineState objects selected at encode time by the source pixel
// format; non-uniform `dispatchThreads`; uniforms uploaded via `setBytes`.
// ===========================================================================

// MARK: - Shared helpers

/// Compile one PSO for a blur kernel. When `isHDR` is non-nil the function is
/// specialised against the `[[function_constant(0)]]` flag; pass nil for kernels
/// (the horizontal Gaussian pass) that don't reference the constant.
private func makeBlurPSO(
    engine: MetalForgeEngine,
    kernel: String,
    isHDR: Bool?
) throws -> MTLComputePipelineState {
    let function: MTLFunction
    if let isHDR {
        let constants = MTLFunctionConstantValues()
        var flag = isHDR
        constants.setConstantValue(&flag, type: .bool, index: 0)
        function = try engine.makeFunction(name: kernel, constantValues: constants)
    } else {
        function = try engine.makeFunction(name: kernel)
    }

    do {
        return try engine.device.makeComputePipelineState(function: function)
    } catch {
        throw MetalForgeError.pipelineStateCreationFailed(error.localizedDescription)
    }
}

/// SIMD-aligned threadgroup dispatch — identical shape to every other filter.
private func dispatchBlur(
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

// MARK: - GaussianBlurFilter

/// Uniform layout — must mirror `GaussianBlurUniforms` in `BlurKernels.metal`.
private struct GaussianBlurUniforms {
    var radius:    Float
    var intensity: Float
}

/// Separable Gaussian blur. The horizontal pass writes a filter-owned
/// intermediate texture; the vertical pass reads it back (plus the original
/// source) and blends by `intensity`.
///
/// The intermediate is allocated once and **reused** across frames — it is only
/// reallocated when the input dimensions or pixel format change (the same
/// persistent-buffer pattern used by the temporal filters). No per-frame texture
/// allocation occurs.
///
/// Configure `radius` / `intensity` between frames; reads happen inside `encode`.
// @unchecked Sendable: the protocol requires `Sendable`, but the configuration
// properties and the lazily-(re)allocated intermediate are mutable. The intermediate
// is guarded by `bufferLock`; configuration is expected to be written by the caller
// between frames, matching every other MetalForge filter.
public final class GaussianBlurFilter: @unchecked Sendable, MetalForgeFilter {

    /// Blur radius in pixels. Clamped to `[0, 64]` at encode time. Default `4`.
    public var radius: Float = 4.0

    /// Blend between original (`0`) and fully blurred (`1`). Clamped to `[0, 1]`.
    /// Default `1`.
    public var intensity: Float = 1.0

    private let device: MTLDevice
    private let horizontalPSO: MTLComputePipelineState
    private let verticalSdrPSO: MTLComputePipelineState
    private let verticalHdrPSO: MTLComputePipelineState

    // Persistent intermediate (horizontal-pass output). Guarded by the lock;
    // reallocated only when the source dims/format change.
    private let bufferLock = NSLock()
    private var intermediate: MTLTexture?
    private var intermediateFormat: MTLPixelFormat = .invalid
    private var intermediateWidth:  Int = 0
    private var intermediateHeight: Int = 0

    public init(engine: MetalForgeEngine) throws {
        device         = engine.device
        horizontalPSO  = try makeBlurPSO(engine: engine, kernel: "gaussianBlurHorizontalKernel", isHDR: nil)
        verticalSdrPSO = try makeBlurPSO(engine: engine, kernel: "gaussianBlurVerticalKernel", isHDR: false)
        verticalHdrPSO = try makeBlurPSO(engine: engine, kernel: "gaussianBlurVerticalKernel", isHDR: true)
    }

    /// Drop the intermediate texture (e.g. before a resolution change). The next
    /// `encode` reallocates it. Safe to call from any thread.
    public func clearHistory() {
        bufferLock.lock()
        intermediate       = nil
        intermediateFormat = .invalid
        intermediateWidth  = 0
        intermediateHeight = 0
        bufferLock.unlock()
    }

    public func encode(
        source: MTLTexture,
        destination: MTLTexture,
        commandBuffer: MTLCommandBuffer
    ) {
        // ----- Acquire / reuse the intermediate under the lock -----
        bufferLock.lock()
        let needsAlloc =
            intermediate == nil
            || intermediateFormat != source.pixelFormat
            || intermediateWidth  != source.width
            || intermediateHeight != source.height

        if needsAlloc {
            let desc = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: source.pixelFormat,
                width:       source.width,
                height:      source.height,
                mipmapped:   false
            )
            desc.storageMode = .private
            desc.usage       = [.shaderRead, .shaderWrite]
            intermediate       = device.makeTexture(descriptor: desc)
            intermediateFormat = source.pixelFormat
            intermediateWidth  = source.width
            intermediateHeight = source.height
        }
        let mid = intermediate
        bufferLock.unlock()

        guard let mid else { return }

        var uniforms = GaussianBlurUniforms(
            radius:    max(radius, 0),
            intensity: min(max(intensity, 0), 1)
        )
        let verticalPSO = (source.pixelFormat == .rgba16Float) ? verticalHdrPSO : verticalSdrPSO

        // ----- Pass 1: horizontal → intermediate -----
        if let enc = commandBuffer.makeComputeCommandEncoder() {
            enc.label = "GaussianBlurFilter.horizontal"
            enc.setComputePipelineState(horizontalPSO)
            enc.setTexture(source, index: 0)
            enc.setTexture(mid,    index: 1)
            enc.setBytes(&uniforms, length: MemoryLayout<GaussianBlurUniforms>.stride, index: 0)
            dispatchBlur(encoder: enc, pso: horizontalPSO, width: destination.width, height: destination.height)
            enc.endEncoding()
        }

        // ----- Pass 2: vertical (+ blend with original) → destination -----
        if let enc = commandBuffer.makeComputeCommandEncoder() {
            enc.label = "GaussianBlurFilter.vertical"
            enc.setComputePipelineState(verticalPSO)
            enc.setTexture(mid,         index: 0)
            enc.setTexture(source,      index: 1)
            enc.setTexture(destination, index: 2)
            enc.setBytes(&uniforms, length: MemoryLayout<GaussianBlurUniforms>.stride, index: 0)
            dispatchBlur(encoder: enc, pso: verticalPSO, width: destination.width, height: destination.height)
            enc.endEncoding()
        }
    }
}

// MARK: - SharpenFilter

/// Uniform layout — must mirror `SharpenUniforms` in `BlurKernels.metal`.
private struct SharpenUniforms {
    var amount: Float
}

/// Single-pass unsharp-mask sharpen using a 4-neighbour Laplacian.
///
/// `amount` is the edge-boost strength: `0` is the identity, larger values
/// crisp-up local contrast. Clamped to `[0, 5]` at encode time.
// @unchecked Sendable: mutable `amount` is written by the caller between frames,
// matching every other MetalForge filter. No per-frame allocation occurs.
public final class SharpenFilter: @unchecked Sendable, MetalForgeFilter {

    /// Sharpening strength. Clamped to `[0, 5]`. Default `0.5`.
    public var amount: Float = 0.5

    private let sdrPSO: MTLComputePipelineState
    private let hdrPSO: MTLComputePipelineState

    public init(engine: MetalForgeEngine) throws {
        sdrPSO = try makeBlurPSO(engine: engine, kernel: "sharpenKernel", isHDR: false)
        hdrPSO = try makeBlurPSO(engine: engine, kernel: "sharpenKernel", isHDR: true)
    }

    public func encode(
        source: MTLTexture,
        destination: MTLTexture,
        commandBuffer: MTLCommandBuffer
    ) {
        let pso = (source.pixelFormat == .rgba16Float) ? hdrPSO : sdrPSO

        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "SharpenFilter"
        encoder.setComputePipelineState(pso)
        encoder.setTexture(source,      index: 0)
        encoder.setTexture(destination, index: 1)

        var uniforms = SharpenUniforms(amount: min(max(amount, 0), 5))
        encoder.setBytes(&uniforms, length: MemoryLayout<SharpenUniforms>.stride, index: 0)

        dispatchBlur(encoder: encoder, pso: pso, width: destination.width, height: destination.height)
        encoder.endEncoding()
    }
}
