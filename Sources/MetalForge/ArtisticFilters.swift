import Metal
import simd
import Foundation

// ===========================================================================
// ArtisticFilters — v0.6.0 Artistic Effects Pack.
//
// Twelve single-pass stylisation filters sharing ArtisticKernels.metal.
// All reuse the `SinglePassColorFilter` base.
// ===========================================================================

// MARK: - Toon / SmoothToon

private struct ToonUniforms {
    var threshold:          Float
    var quantizationLevels: Float
}

/// Cartoon shading: posterised colours with dark Sobel outlines.
public final class ToonFilter: SinglePassColorFilter, @unchecked Sendable {

    /// Edge-darkness cut-off. Clamped to `[0, 1]`. Default `0.2`.
    public var threshold: Float = 0.2

    /// Colour steps per channel. Clamped to `[2, 64]`. Default `10`.
    public var quantizationLevels: Float = 10.0

    public init(engine: MetalForgeEngine) throws {
        try super.init(engine: engine, kernel: "toonKernel", label: "ToonFilter")
    }

    override func setUniforms(on encoder: MTLComputeCommandEncoder) {
        var u = ToonUniforms(
            threshold:          min(max(threshold, 0), 1),
            quantizationLevels: min(max(quantizationLevels, 2), 64)
        )
        encoder.setBytes(&u, length: MemoryLayout<ToonUniforms>.stride, index: 0)
    }
}

/// Toon with built-in 3×3 smoothing — cleaner cartoon regions on noisy video.
public final class SmoothToonFilter: SinglePassColorFilter, @unchecked Sendable {

    /// Edge-darkness cut-off. Clamped to `[0, 1]`. Default `0.2`.
    public var threshold: Float = 0.2

    /// Colour steps per channel. Clamped to `[2, 64]`. Default `10`.
    public var quantizationLevels: Float = 10.0

    public init(engine: MetalForgeEngine) throws {
        try super.init(engine: engine, kernel: "smoothToonKernel", label: "SmoothToonFilter")
    }

    override func setUniforms(on encoder: MTLComputeCommandEncoder) {
        var u = ToonUniforms(
            threshold:          min(max(threshold, 0), 1),
            quantizationLevels: min(max(quantizationLevels, 2), 64)
        )
        encoder.setBytes(&u, length: MemoryLayout<ToonUniforms>.stride, index: 0)
    }
}

// MARK: - Sketch / ThresholdSketch

/// Pencil sketch: inverted gradient magnitude — dark strokes on white paper.
public final class SketchFilter: SinglePassColorFilter, @unchecked Sendable {

    /// Stroke gain. Clamped to `[0, 8]`. Default `1`.
    public var edgeStrength: Float = 1.0

    public init(engine: MetalForgeEngine) throws {
        try super.init(engine: engine, kernel: "sketchKernel", label: "SketchFilter")
    }

    override func setUniforms(on encoder: MTLComputeCommandEncoder) {
        var s = min(max(edgeStrength, 0), 8)
        encoder.setBytes(&s, length: MemoryLayout<Float>.stride, index: 0)
    }
}

private struct ThresholdSketchUniforms {
    var threshold:    Float
    var edgeStrength: Float
}

/// Binary ink sketch: pure black lines on pure white.
public final class ThresholdSketchFilter: SinglePassColorFilter, @unchecked Sendable {

    /// Edge magnitude above this becomes ink. Clamped to `[0, 1]`. Default `0.25`.
    public var threshold: Float = 0.25

    /// Stroke gain. Clamped to `[0, 8]`. Default `1`.
    public var edgeStrength: Float = 1.0

    public init(engine: MetalForgeEngine) throws {
        try super.init(engine: engine, kernel: "thresholdSketchKernel", label: "ThresholdSketchFilter")
    }

    override func setUniforms(on encoder: MTLComputeCommandEncoder) {
        var u = ThresholdSketchUniforms(
            threshold:    min(max(threshold, 0), 1),
            edgeStrength: min(max(edgeStrength, 0), 8)
        )
        encoder.setBytes(&u, length: MemoryLayout<ThresholdSketchUniforms>.stride, index: 0)
    }
}

// MARK: - Crosshatch

private struct CrosshatchUniforms {
    var spacing:   Float
    var lineWidth: Float
}

/// Engraving-style diagonal hatching; darker areas accumulate more stroke
/// families.
public final class CrosshatchFilter: SinglePassColorFilter, @unchecked Sendable {

    /// Hatch spacing in pixels. Clamped to `[2, 64]`. Default `10`.
    public var spacing: Float = 10.0

    /// Stroke width in pixels. Clamped to `[0.5, 8]`. Default `1.5`.
    public var lineWidth: Float = 1.5

    public init(engine: MetalForgeEngine) throws {
        try super.init(engine: engine, kernel: "crosshatchKernel", label: "CrosshatchFilter")
    }

    override func setUniforms(on encoder: MTLComputeCommandEncoder) {
        var u = CrosshatchUniforms(
            spacing:   min(max(spacing, 2), 64),
            lineWidth: min(max(lineWidth, 0.5), 8)
        )
        encoder.setBytes(&u, length: MemoryLayout<CrosshatchUniforms>.stride, index: 0)
    }
}

// MARK: - Halftone / PolkaDot

private struct DotUniforms {
    var dotSizePx:  Float
    var dotScaling: Float
}

/// Newspaper halftone: black-and-white luminance dots — dark areas get big
/// ink dots.
public final class HalftoneFilter: SinglePassColorFilter, @unchecked Sendable {

    /// Dot-grid period in pixels. Clamped to `[2, 64]`. Default `8`.
    public var dotSize: Float = 8.0

    /// Maximum dot radius as a fraction of the half-cell. Clamped to `[0.1, 1.5]`.
    /// Default `1`.
    public var dotScaling: Float = 1.0

    public init(engine: MetalForgeEngine) throws {
        try super.init(engine: engine, kernel: "halftoneKernel", label: "HalftoneFilter")
    }

    override func setUniforms(on encoder: MTLComputeCommandEncoder) {
        var u = DotUniforms(
            dotSizePx:  min(max(dotSize, 2), 64),
            dotScaling: min(max(dotScaling, 0.1), 1.5)
        )
        encoder.setBytes(&u, length: MemoryLayout<DotUniforms>.stride, index: 0)
    }
}

/// Pop-art polka dots: constant-size coloured dots (cell colour) on black.
public final class PolkaDotFilter: SinglePassColorFilter, @unchecked Sendable {

    /// Dot-grid period in pixels. Clamped to `[2, 64]`. Default `10`.
    public var dotSize: Float = 10.0

    /// Dot radius as a fraction of the half-cell. Clamped to `[0.1, 1]`.
    /// Default `0.9`.
    public var dotScaling: Float = 0.9

    public init(engine: MetalForgeEngine) throws {
        try super.init(engine: engine, kernel: "polkaDotKernel", label: "PolkaDotFilter")
    }

    override func setUniforms(on encoder: MTLComputeCommandEncoder) {
        var u = DotUniforms(
            dotSizePx:  min(max(dotSize, 2), 64),
            dotScaling: min(max(dotScaling, 0.1), 1)
        )
        encoder.setBytes(&u, length: MemoryLayout<DotUniforms>.stride, index: 0)
    }
}

// MARK: - Kuwahara

/// Oil-painting smoothing: each pixel takes the mean of the least-variant of
/// four overlapping quadrants — flattens texture, keeps edges.
public final class KuwaharaFilter: SinglePassColorFilter, @unchecked Sendable {

    /// Quadrant radius in pixels. Clamped to `[1, 6]` (cost grows as r²).
    /// Default `4`.
    public var radius: Float = 4.0

    public init(engine: MetalForgeEngine) throws {
        try super.init(engine: engine, kernel: "kuwaharaKernel", label: "KuwaharaFilter")
    }

    override func setUniforms(on encoder: MTLComputeCommandEncoder) {
        var r = min(max(radius, 1), 6)
        encoder.setBytes(&r, length: MemoryLayout<Float>.stride, index: 0)
    }
}

// MARK: - Pixellate / PolarPixellate

/// Square pixel mosaic.
public final class PixellateFilter: SinglePassColorFilter, @unchecked Sendable {

    /// Cell edge in pixels. Clamped to `[1, 256]`. Default `10`.
    public var cellSize: Float = 10.0

    public init(engine: MetalForgeEngine) throws {
        try super.init(engine: engine, kernel: "pixellateKernel", label: "PixellateFilter")
    }

    override func setUniforms(on encoder: MTLComputeCommandEncoder) {
        var s = min(max(cellSize, 1), 256)
        encoder.setBytes(&s, length: MemoryLayout<Float>.stride, index: 0)
    }
}

private struct PolarPixellateUniforms {
    var center:      SIMD2<Float>
    var radialSize:  Float
    var angularSize: Float
}

/// Pixellation in polar coordinates — concentric wedge tiles around a centre.
public final class PolarPixellateFilter: SinglePassColorFilter, @unchecked Sendable {

    /// Normalised centre. Default frame centre.
    public var center: SIMD2<Float> = SIMD2(0.5, 0.5)

    /// Radial tile step (normalised). Clamped to `[0.005, 0.5]`. Default `0.05`.
    public var radialSize: Float = 0.05

    /// Angular tile step in radians. Clamped to `[0.01, 1]`. Default `0.1`.
    public var angularSize: Float = 0.1

    public init(engine: MetalForgeEngine) throws {
        try super.init(engine: engine, kernel: "polarPixellateKernel", label: "PolarPixellateFilter")
    }

    override func setUniforms(on encoder: MTLComputeCommandEncoder) {
        var u = PolarPixellateUniforms(
            center:      center,
            radialSize:  min(max(radialSize, 0.005), 0.5),
            angularSize: min(max(angularSize, 0.01), 1)
        )
        encoder.setBytes(&u, length: MemoryLayout<PolarPixellateUniforms>.stride, index: 0)
    }
}

// MARK: - Mosaic

private struct MosaicUniforms {
    var tileSizePx: Float
    var groutWidth: Float
}

/// Ceramic-tile mosaic: averaged square tiles separated by dark grout seams.
public final class MosaicFilter: SinglePassColorFilter, @unchecked Sendable {

    /// Tile edge in pixels. Clamped to `[4, 256]`. Default `16`.
    public var tileSize: Float = 16.0

    /// Grout seam width in pixels (`0` disables). Clamped to `[0, 8]`. Default `1`.
    public var groutWidth: Float = 1.0

    public init(engine: MetalForgeEngine) throws {
        try super.init(engine: engine, kernel: "mosaicKernel", label: "MosaicFilter")
    }

    override func setUniforms(on encoder: MTLComputeCommandEncoder) {
        var u = MosaicUniforms(
            tileSizePx: min(max(tileSize, 4), 256),
            groutWidth: min(max(groutWidth, 0), 8)
        )
        encoder.setBytes(&u, length: MemoryLayout<MosaicUniforms>.stride, index: 0)
    }
}

// MARK: - CGAColorspace

/// 1981 IBM CGA look: chunky 2×2 pixels snapped to the black/cyan/magenta/
/// white palette. No parameters.
public final class CGAColorspaceFilter: SinglePassColorFilter, @unchecked Sendable {

    public init(engine: MetalForgeEngine) throws {
        try super.init(engine: engine, kernel: "cgaColorspaceKernel", label: "CGAColorspaceFilter")
    }
}
