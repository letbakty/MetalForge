import Metal
import simd
import Foundation

// ===========================================================================
// ToneFilters — v0.2.0 Color Pack (tone / grayscale / threshold half).
//
// Eight single-pass filters sharing ToneKernels.metal. All reuse the
// `SinglePassColorFilter` base (SDR/HDR PSO pair, dispatch, uniforms hook)
// from ColorAdjustmentFilters.swift.
// ===========================================================================

// MARK: - GrayscaleFilter

/// Luminance collapse using the colour-space-appropriate weights
/// (BT.709 for SDR, BT.2020 for HDR). No parameters.
public final class GrayscaleFilter: SinglePassColorFilter, @unchecked Sendable {

    public init(engine: MetalForgeEngine) throws {
        try super.init(engine: engine, kernel: "grayscaleKernel", label: "GrayscaleFilter")
    }
}

// MARK: - SepiaFilter

/// Classic sepia tone (same matrix as `CISepiaTone`), blended with the
/// original by `intensity`.
public final class SepiaFilter: SinglePassColorFilter, @unchecked Sendable {

    /// `0` = original … `1` = full sepia. Clamped. Default `1`.
    public var intensity: Float = 1.0

    public init(engine: MetalForgeEngine) throws {
        try super.init(engine: engine, kernel: "sepiaKernel", label: "SepiaFilter")
    }

    override func setUniforms(on encoder: MTLComputeCommandEncoder) {
        var i = min(max(intensity, 0), 1)
        encoder.setBytes(&i, length: MemoryLayout<Float>.stride, index: 0)
    }
}

// MARK: - HazeFilter

private struct HazeUniforms {
    var distance: Float
    var slope:    Float
}

/// Adds or removes a white atmospheric veil with a vertical gradient —
/// positive `distance` clears haze, negative adds it.
public final class HazeFilter: SinglePassColorFilter, @unchecked Sendable {

    /// Haze strength. Clamped to `[-0.3, 0.3]`; `0` = identity. Default `0.2`.
    public var distance: Float = 0.2

    /// Vertical gradient of the effect. Clamped to `[-0.3, 0.3]`. Default `0`.
    public var slope: Float = 0.0

    public init(engine: MetalForgeEngine) throws {
        try super.init(engine: engine, kernel: "hazeKernel", label: "HazeFilter")
    }

    override func setUniforms(on encoder: MTLComputeCommandEncoder) {
        var u = HazeUniforms(
            distance: min(max(distance, -0.3), 0.3),
            slope:    min(max(slope, -0.3), 0.3)
        )
        encoder.setBytes(&u, length: MemoryLayout<HazeUniforms>.stride, index: 0)
    }
}

// MARK: - SkinToneFilter

private struct SkinToneUniforms {
    var adjust:            Float
    var skinHue:           Float
    var skinHueThreshold:  Float
    var upperTone:         Float
}

/// Adjusts colours inside the skin-hue band only, pushing them toward pink
/// or green while leaving the rest of the image untouched.
public final class SkinToneFilter: SinglePassColorFilter, @unchecked Sendable {

    /// Adjustment direction/strength. Clamped to `[-1, 1]`; `0` = identity.
    /// Default `0.3`.
    public var adjust: Float = 0.3

    /// Centre of the skin band in hue turns. Default `0.05` (orange-ish).
    public var skinHue: Float = 0.05

    /// Band tightness — larger values narrow the affected hue range.
    /// Clamped to `[1, 80]`. Default `40`.
    public var skinHueThreshold: Float = 40.0

    /// Tone target: `.pink` warms skin, `.green` shifts it olive.
    public enum ToneTarget: Sendable { case pink, green }
    public var toneTarget: ToneTarget = .pink

    public init(engine: MetalForgeEngine) throws {
        try super.init(engine: engine, kernel: "skinToneKernel", label: "SkinToneFilter")
    }

    override func setUniforms(on encoder: MTLComputeCommandEncoder) {
        var u = SkinToneUniforms(
            adjust:           min(max(adjust, -1), 1),
            skinHue:          skinHue,
            skinHueThreshold: min(max(skinHueThreshold, 1), 80),
            upperTone:        toneTarget == .green ? 1 : 0
        )
        encoder.setBytes(&u, length: MemoryLayout<SkinToneUniforms>.stride, index: 0)
    }
}

// MARK: - LuminanceThresholdFilter

/// Hard black/white cut at a fixed luminance threshold.
public final class LuminanceThresholdFilter: SinglePassColorFilter, @unchecked Sendable {

    /// Pixels with luminance above this become white, below become black.
    /// Clamped to `[0, 1]`. Default `0.5`.
    public var threshold: Float = 0.5

    public init(engine: MetalForgeEngine) throws {
        try super.init(engine: engine, kernel: "luminanceThresholdKernel", label: "LuminanceThresholdFilter")
    }

    override func setUniforms(on encoder: MTLComputeCommandEncoder) {
        var t = min(max(threshold, 0), 1)
        encoder.setBytes(&t, length: MemoryLayout<Float>.stride, index: 0)
    }
}

// MARK: - AdaptiveThresholdFilter

private struct AdaptiveThresholdUniforms {
    var radius: Float
    var bias:   Float
}

/// Binarises against the *local* average luminance instead of a global
/// threshold — robust to uneven lighting (document-scanner look).
public final class AdaptiveThresholdFilter: SinglePassColorFilter, @unchecked Sendable {

    /// Radius of the local-average window in pixels. Clamped to `[1, 64]`.
    /// Default `8`.
    public var radius: Float = 8.0

    /// Subtracted from the local mean before comparison; larger values push
    /// more pixels to white. Clamped to `[-0.5, 0.5]`. Default `0.05`.
    public var bias: Float = 0.05

    public init(engine: MetalForgeEngine) throws {
        try super.init(engine: engine, kernel: "adaptiveThresholdKernel", label: "AdaptiveThresholdFilter")
    }

    override func setUniforms(on encoder: MTLComputeCommandEncoder) {
        var u = AdaptiveThresholdUniforms(
            radius: min(max(radius, 1), 64),
            bias:   min(max(bias, -0.5), 0.5)
        )
        encoder.setBytes(&u, length: MemoryLayout<AdaptiveThresholdUniforms>.stride, index: 0)
    }
}

// MARK: - PosterizeFilter

/// Quantises each channel to a fixed number of tonal levels.
public final class PosterizeFilter: SinglePassColorFilter, @unchecked Sendable {

    /// Number of levels per channel. Clamped to `[2, 256]`. Default `8`.
    public var levels: Float = 8.0

    public init(engine: MetalForgeEngine) throws {
        try super.init(engine: engine, kernel: "posterizeKernel", label: "PosterizeFilter")
    }

    override func setUniforms(on encoder: MTLComputeCommandEncoder) {
        var l = min(max(levels, 2), 256)
        encoder.setBytes(&l, length: MemoryLayout<Float>.stride, index: 0)
    }
}

// MARK: - ColorHalftoneFilter

/// CMYK-print-style halftoning: each RGB channel becomes a rotated grid of
/// dots whose size tracks the channel's local intensity.
public final class ColorHalftoneFilter: SinglePassColorFilter, @unchecked Sendable {

    /// Dot-grid period in pixels. Clamped to `[2, 64]`. Default `8`.
    public var dotSize: Float = 8.0

    public init(engine: MetalForgeEngine) throws {
        try super.init(engine: engine, kernel: "colorHalftoneKernel", label: "ColorHalftoneFilter")
    }

    override func setUniforms(on encoder: MTLComputeCommandEncoder) {
        var d = min(max(dotSize, 2), 64)
        encoder.setBytes(&d, length: MemoryLayout<Float>.stride, index: 0)
    }
}
