import Metal
import simd
import Foundation

// ===========================================================================
// BlendFilters — v0.4.0 Blend Modes API.
//
//   • BlendMode        — 23 compositing modes (Photoshop/W3C semantics).
//   • BlendModeFilter  — blends an overlay texture over the pipeline frame.
//   • AlphaBlendFilter — convenience: BlendModeFilter fixed to `.normal`.
//   • ChromaKeyFilter  — green-screen keying with optional background.
//   • MaskBlendFilter  — mask-texture-driven mix between frame and overlay.
//
// Overlay/mask textures are caller-provided `MTLTexture`s (sampled with
// normalised UVs, so their size need not match the frame). When a required
// texture is missing, the filters degrade to a pass-through blit — the
// pipeline keeps running.
// ===========================================================================

// MARK: - BlendMode

/// Compositing modes for `BlendModeFilter`. Raw values must stay in sync with
/// the `switch` in `BlendKernels.metal`.
public enum BlendMode: Int32, CaseIterable, Sendable {
    case normal      = 0
    case multiply    = 1
    case screen      = 2
    case overlay     = 3
    case hardLight   = 4
    case softLight   = 5
    case darken      = 6
    case lighten     = 7
    case colorDodge  = 8
    case colorBurn   = 9
    case linearBurn  = 10
    case difference  = 11
    case exclusion   = 12
    case subtract    = 13
    case divide      = 14
    case hue         = 15
    case saturation  = 16
    case color       = 17
    case luminosity  = 18
    case vividLight  = 19
    case pinLight    = 20
    case hardMix     = 21
    case dissolve    = 22
}

// MARK: - BlendModeFilter

private struct BlendModeUniforms {
    var mode:      Int32
    var intensity: Float
    var seed:      Float
}

/// Composites `overlayTexture` over the pipeline frame using a `BlendMode`.
///
/// The overlay is sampled with normalised coordinates, so any texture size
/// works (it is stretched to cover the frame). The overlay's alpha channel is
/// honoured, then `intensity` blends the final result against the original.
/// With no overlay set, the filter is a pass-through copy.
// @unchecked Sendable: mutable configuration (including the overlay texture
// reference) is written by the caller between frames, matching every other
// MetalForge filter.
public final class BlendModeFilter: @unchecked Sendable, MetalForgeFilter {

    /// The compositing mode. Default `.normal`.
    public var mode: BlendMode = .normal

    /// Blend strength. For `.dissolve` this is the dither coverage.
    /// Clamped to `[0, 1]`. Default `1`.
    public var intensity: Float = 1.0

    /// Dither pattern offset for `.dissolve`; advance per frame for an
    /// animated dissolve. Default `0`.
    public var seed: Float = 0.0

    /// The texture composited over the frame. `nil` = pass-through.
    public var overlayTexture: MTLTexture?

    private let sdrPSO: MTLComputePipelineState
    private let hdrPSO: MTLComputePipelineState

    public init(engine: MetalForgeEngine) throws {
        sdrPSO = try makeColorAdjustmentPSO(engine: engine, kernel: "blendModeKernel", isHDR: false)
        hdrPSO = try makeColorAdjustmentPSO(engine: engine, kernel: "blendModeKernel", isHDR: true)
    }

    public func encode(
        source: MTLTexture,
        destination: MTLTexture,
        commandBuffer: MTLCommandBuffer
    ) {
        guard let overlay = overlayTexture else {
            // No overlay configured — copy the frame through unchanged.
            if let blit = commandBuffer.makeBlitCommandEncoder() {
                blit.label = "BlendModeFilter.passthrough"
                blit.copy(from: source, to: destination)
                blit.endEncoding()
            }
            return
        }

        let pso = (source.pixelFormat == .rgba16Float) ? hdrPSO : sdrPSO
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "BlendModeFilter"
        encoder.setComputePipelineState(pso)
        encoder.setTexture(source,      index: 0)
        encoder.setTexture(destination, index: 1)
        encoder.setTexture(overlay,     index: 2)

        var u = BlendModeUniforms(
            mode:      mode.rawValue,
            intensity: min(max(intensity, 0), 1),
            seed:      seed
        )
        encoder.setBytes(&u, length: MemoryLayout<BlendModeUniforms>.stride, index: 0)

        dispatchColorAdjustment(
            encoder: encoder, pso: pso,
            width: destination.width, height: destination.height
        )
        encoder.endEncoding()
    }
}

// MARK: - AlphaBlendFilter

/// Plain alpha compositing of an overlay over the frame — `BlendModeFilter`
/// locked to `.normal`, kept as its own type for API discoverability.
public final class AlphaBlendFilter: @unchecked Sendable, MetalForgeFilter {

    /// Blend strength on top of the overlay's own alpha. Clamped to `[0, 1]`.
    /// Default `1`.
    public var intensity: Float {
        get { inner.intensity }
        set { inner.intensity = newValue }
    }

    /// The texture composited over the frame. `nil` = pass-through.
    public var overlayTexture: MTLTexture? {
        get { inner.overlayTexture }
        set { inner.overlayTexture = newValue }
    }

    private let inner: BlendModeFilter

    public init(engine: MetalForgeEngine) throws {
        inner = try BlendModeFilter(engine: engine)
        inner.mode = .normal
    }

    public func encode(
        source: MTLTexture,
        destination: MTLTexture,
        commandBuffer: MTLCommandBuffer
    ) {
        inner.encode(source: source, destination: destination, commandBuffer: commandBuffer)
    }
}

// MARK: - ChromaKeyFilter

private struct ChromaKeyUniforms {
    var keyColor:             SIMD3<Float>
    var thresholdSensitivity: Float
    var smoothing:            Float
    var hasBackground:        Float
    var pad0: Float = 0
    var pad1: Float = 0
}

/// Green-screen keying. Pixels whose chroma matches `keyColor` are replaced
/// by `backgroundTexture` (or made transparent when no background is set —
/// the output alpha carries the matte).
// @unchecked Sendable: see BlendModeFilter.
public final class ChromaKeyFilter: @unchecked Sendable, MetalForgeFilter {

    /// The colour to key out. Default pure green.
    public var keyColor: SIMD3<Float> = SIMD3(0, 1, 0)

    /// Chroma distance below which a pixel is fully keyed.
    /// Clamped to `[0, 1]`. Default `0.4`.
    public var thresholdSensitivity: Float = 0.4

    /// Width of the soft transition band above the threshold.
    /// Clamped to `[0.001, 1]`. Default `0.1`.
    public var smoothing: Float = 0.1

    /// Replacement background. `nil` → keyed pixels become transparent black.
    public var backgroundTexture: MTLTexture?

    private let device: MTLDevice
    private let sdrPSO: MTLComputePipelineState
    private let hdrPSO: MTLComputePipelineState

    // 1×1 dummy bound when no background is set (Metal requires every
    // declared texture argument to be bound).
    private let dummyBackground: MTLTexture?

    public init(engine: MetalForgeEngine) throws {
        device = engine.device
        sdrPSO = try makeColorAdjustmentPSO(engine: engine, kernel: "chromaKeyKernel", isHDR: false)
        hdrPSO = try makeColorAdjustmentPSO(engine: engine, kernel: "chromaKeyKernel", isHDR: true)

        let desc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .bgra8Unorm, width: 1, height: 1, mipmapped: false
        )
        desc.usage = .shaderRead
        #if os(macOS)
        desc.storageMode = .managed
        #else
        desc.storageMode = .shared
        #endif
        dummyBackground = engine.device.makeTexture(descriptor: desc)
    }

    public func encode(
        source: MTLTexture,
        destination: MTLTexture,
        commandBuffer: MTLCommandBuffer
    ) {
        let pso = (source.pixelFormat == .rgba16Float) ? hdrPSO : sdrPSO
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "ChromaKeyFilter"
        encoder.setComputePipelineState(pso)
        encoder.setTexture(source,      index: 0)
        encoder.setTexture(destination, index: 1)
        encoder.setTexture(backgroundTexture ?? dummyBackground, index: 2)

        var u = ChromaKeyUniforms(
            keyColor:             keyColor,
            thresholdSensitivity: min(max(thresholdSensitivity, 0), 1),
            smoothing:            min(max(smoothing, 0.001), 1),
            hasBackground:        backgroundTexture != nil ? 1 : 0
        )
        encoder.setBytes(&u, length: MemoryLayout<ChromaKeyUniforms>.stride, index: 0)

        dispatchColorAdjustment(
            encoder: encoder, pso: pso,
            width: destination.width, height: destination.height
        )
        encoder.endEncoding()
    }
}

// MARK: - MaskBlendFilter

/// Mixes the frame with `overlayTexture` using the luminance of
/// `maskTexture` as the per-pixel weight (white = overlay, black = frame).
/// Missing overlay or mask → pass-through.
// @unchecked Sendable: see BlendModeFilter.
public final class MaskBlendFilter: @unchecked Sendable, MetalForgeFilter {

    /// The texture shown where the mask is white. `nil` = pass-through.
    public var overlayTexture: MTLTexture?

    /// The mask; its luminance is the blend weight. `nil` = pass-through.
    public var maskTexture: MTLTexture?

    private let sdrPSO: MTLComputePipelineState
    private let hdrPSO: MTLComputePipelineState

    public init(engine: MetalForgeEngine) throws {
        sdrPSO = try makeColorAdjustmentPSO(engine: engine, kernel: "maskBlendKernel", isHDR: false)
        hdrPSO = try makeColorAdjustmentPSO(engine: engine, kernel: "maskBlendKernel", isHDR: true)
    }

    public func encode(
        source: MTLTexture,
        destination: MTLTexture,
        commandBuffer: MTLCommandBuffer
    ) {
        guard let overlay = overlayTexture, let mask = maskTexture else {
            if let blit = commandBuffer.makeBlitCommandEncoder() {
                blit.label = "MaskBlendFilter.passthrough"
                blit.copy(from: source, to: destination)
                blit.endEncoding()
            }
            return
        }

        let pso = (source.pixelFormat == .rgba16Float) ? hdrPSO : sdrPSO
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "MaskBlendFilter"
        encoder.setComputePipelineState(pso)
        encoder.setTexture(source,      index: 0)
        encoder.setTexture(destination, index: 1)
        encoder.setTexture(overlay,     index: 2)
        encoder.setTexture(mask,        index: 3)

        dispatchColorAdjustment(
            encoder: encoder, pso: pso,
            width: destination.width, height: destination.height
        )
        encoder.endEncoding()
    }
}
