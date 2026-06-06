import Metal
import simd
import Foundation

// ===========================================================================
// StylizationFilters — VignetteFilter + ScanlineFilter + RGBSplitFilter.
//
// All three share StylizationKernels.metal and follow the established pattern:
// two isHDR-specialised PSOs selected by source pixel format, non-uniform
// dispatch, uniforms via `setBytes`. None allocate per-frame textures.
// ===========================================================================

// MARK: - Shared helpers

private func makeStylizationPSO(
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

private func dispatchStylization(
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

// MARK: - VignetteFilter

/// Uniform layout — must mirror `VignetteUniforms` in `StylizationKernels.metal`.
private struct VignetteUniforms {
    var intensity: Float
    var radius:    Float
    var softness:  Float
}

/// Radial edge darkening. `radius` is where the darkening starts (in
/// aspect-corrected normalised distance from centre), `softness` is the width of
/// the falloff band, and `intensity` is how dark the corners become.
// @unchecked Sendable: mutable configuration is written by the caller between
// frames, matching every other MetalForge filter. No per-frame allocation.
public final class VignetteFilter: @unchecked Sendable, MetalForgeFilter {

    /// Darkening strength. Clamped to `[0, 1]`. Default `0.5`.
    public var intensity: Float = 0.5

    /// Normalised distance where the falloff begins. Clamped to `[0, 1.5]`.
    /// Default `0.75`.
    public var radius: Float = 0.75

    /// Width of the falloff band. Clamped to `[0.001, 1.5]` so the smoothstep
    /// edges never coincide. Default `0.45`.
    public var softness: Float = 0.45

    private let sdrPSO: MTLComputePipelineState
    private let hdrPSO: MTLComputePipelineState

    public init(engine: MetalForgeEngine) throws {
        sdrPSO = try makeStylizationPSO(engine: engine, kernel: "vignetteKernel", isHDR: false)
        hdrPSO = try makeStylizationPSO(engine: engine, kernel: "vignetteKernel", isHDR: true)
    }

    public func encode(
        source: MTLTexture,
        destination: MTLTexture,
        commandBuffer: MTLCommandBuffer
    ) {
        let pso = (source.pixelFormat == .rgba16Float) ? hdrPSO : sdrPSO

        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "VignetteFilter"
        encoder.setComputePipelineState(pso)
        encoder.setTexture(source,      index: 0)
        encoder.setTexture(destination, index: 1)

        var uniforms = VignetteUniforms(
            intensity: min(max(intensity, 0), 1),
            radius:    min(max(radius, 0), 1.5),
            softness:  min(max(softness, 0.001), 1.5)
        )
        encoder.setBytes(&uniforms, length: MemoryLayout<VignetteUniforms>.stride, index: 0)

        dispatchStylization(encoder: encoder, pso: pso, width: destination.width, height: destination.height)
        encoder.endEncoding()
    }
}

// MARK: - ScanlineFilter

/// Uniform layout — must mirror `ScanlineUniforms` in `StylizationKernels.metal`.
private struct ScanlineUniforms {
    var intensity: Float
    var lineWidth: Float
    var timeSeed:  Float
}

/// Horizontal CRT-style scanlines. `lineWidth` is the band period in pixels,
/// `intensity` is how dark the troughs get, and `timeSeed` scrolls the bands
/// vertically (drive it from a frame clock for a rolling-CRT look).
// @unchecked Sendable: mutable configuration is written by the caller between
// frames, matching every other MetalForge filter. No per-frame allocation.
public final class ScanlineFilter: @unchecked Sendable, MetalForgeFilter {

    /// Scanline strength. Clamped to `[0, 1]`. Default `0.5`.
    public var intensity: Float = 0.5

    /// Band period in pixels. Clamped to `>= 1`. Default `2`.
    public var lineWidth: Float = 2.0

    /// Vertical scroll offset. Not clamped — set to a constant to freeze the
    /// pattern, or feed an advancing clock to scroll. Default `0`.
    public var timeSeed: Float = 0.0

    private let sdrPSO: MTLComputePipelineState
    private let hdrPSO: MTLComputePipelineState

    public init(engine: MetalForgeEngine) throws {
        sdrPSO = try makeStylizationPSO(engine: engine, kernel: "scanlineKernel", isHDR: false)
        hdrPSO = try makeStylizationPSO(engine: engine, kernel: "scanlineKernel", isHDR: true)
    }

    public func encode(
        source: MTLTexture,
        destination: MTLTexture,
        commandBuffer: MTLCommandBuffer
    ) {
        let pso = (source.pixelFormat == .rgba16Float) ? hdrPSO : sdrPSO

        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "ScanlineFilter"
        encoder.setComputePipelineState(pso)
        encoder.setTexture(source,      index: 0)
        encoder.setTexture(destination, index: 1)

        var uniforms = ScanlineUniforms(
            intensity: min(max(intensity, 0), 1),
            lineWidth: max(lineWidth, 1),
            timeSeed:  timeSeed
        )
        encoder.setBytes(&uniforms, length: MemoryLayout<ScanlineUniforms>.stride, index: 0)

        dispatchStylization(encoder: encoder, pso: pso, width: destination.width, height: destination.height)
        encoder.endEncoding()
    }
}

// MARK: - RGBSplitFilter

/// Uniform layout — must mirror `RGBSplitUniforms` in `StylizationKernels.metal`.
private struct RGBSplitUniforms {
    var redOffset:   SIMD2<Float>
    var greenOffset: SIMD2<Float>
    var blueOffset:  SIMD2<Float>
    var intensity:   Float
}

/// Per-channel UV displacement. Each colour channel is sampled at its own
/// normalised offset, scaled by a global `intensity`. Alpha comes from the
/// un-shifted centre. A generalisation of chromatic aberration that allows
/// arbitrary per-channel directions.
// @unchecked Sendable: mutable configuration is written by the caller between
// frames, matching every other MetalForge filter. No per-frame allocation.
public final class RGBSplitFilter: @unchecked Sendable, MetalForgeFilter {

    /// Normalised UV offset for the red channel. Default `(0.01, 0)`.
    public var redOffset: SIMD2<Float> = SIMD2(0.01, 0.0)

    /// Normalised UV offset for the green channel. Default `(0, 0)`.
    public var greenOffset: SIMD2<Float> = SIMD2(0.0, 0.0)

    /// Normalised UV offset for the blue channel. Default `(-0.01, 0)`.
    public var blueOffset: SIMD2<Float> = SIMD2(-0.01, 0.0)

    /// Global multiplier on all three offsets. Clamped to `[0, 1]`. Default `1`.
    public var intensity: Float = 1.0

    private let sdrPSO: MTLComputePipelineState
    private let hdrPSO: MTLComputePipelineState

    public init(engine: MetalForgeEngine) throws {
        sdrPSO = try makeStylizationPSO(engine: engine, kernel: "rgbSplitKernel", isHDR: false)
        hdrPSO = try makeStylizationPSO(engine: engine, kernel: "rgbSplitKernel", isHDR: true)
    }

    public func encode(
        source: MTLTexture,
        destination: MTLTexture,
        commandBuffer: MTLCommandBuffer
    ) {
        let pso = (source.pixelFormat == .rgba16Float) ? hdrPSO : sdrPSO

        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "RGBSplitFilter"
        encoder.setComputePipelineState(pso)
        encoder.setTexture(source,      index: 0)
        encoder.setTexture(destination, index: 1)

        var uniforms = RGBSplitUniforms(
            redOffset:   redOffset,
            greenOffset: greenOffset,
            blueOffset:  blueOffset,
            intensity:   min(max(intensity, 0), 1)
        )
        encoder.setBytes(&uniforms, length: MemoryLayout<RGBSplitUniforms>.stride, index: 0)

        dispatchStylization(encoder: encoder, pso: pso, width: destination.width, height: destination.height)
        encoder.endEncoding()
    }
}
