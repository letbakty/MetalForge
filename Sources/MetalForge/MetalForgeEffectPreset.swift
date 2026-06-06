import Foundation

/// A ready-to-use, named look assembled from MetalForge's existing
/// `MetalForgeFilter` building blocks — no preset-specific shaders involved.
///
/// Each case maps to a curated chain of stock filters — colour correction,
/// the analog-distortion and temporal families, and the GPU shader effects
/// (blur, sharpen, vignette, scanlines, RGB split) — with hand-tuned parameters.
/// Call `makeFilters(engine:)` to materialise the chain, or use
/// `MetalForgePipeline.applyPreset(_:)` to swap the pipeline over to a preset in
/// one call (which replaces the current filter chain).
///
/// The "Original" / no-effect state is intentionally **not** a member of this enum:
/// an empty filter chain is the absence of a preset, not a preset itself. UIs that
/// offer an "Original" option should model it separately (e.g. an optional preset).
public enum MetalForgeEffectPreset: String, CaseIterable, Sendable {

    /// Warm, slightly lifted film look with gentle contrast.
    case cinematicWarm
    /// Cool teal-leaning cinematic grade.
    case cinematicCool
    /// Retro VHS tape: chroma fringing, grain, and tracking jitter.
    case vhs
    /// High-contrast black & white.
    case noir
    /// Saturated, cool, neon-edged futuristic look.
    case cyberpunk
    /// Soft, bright, low-contrast ethereal blur.
    case dreamy
    /// Punchy, crushed-contrast high-impact grade.
    case highContrast
    /// Faded warm vintage film stock with fine grain.
    case vintageFilm
    /// Glowing motion trails behind moving subjects.
    case neonTrails

    // MARK: - Metadata

    /// Human-readable name suitable for a picker or button label.
    public var displayName: String {
        switch self {
        case .cinematicWarm: return "Cinematic Warm"
        case .cinematicCool: return "Cinematic Cool"
        case .vhs:           return "VHS"
        case .noir:          return "Noir"
        case .cyberpunk:     return "Cyberpunk"
        case .dreamy:        return "Dreamy"
        case .highContrast:  return "High Contrast"
        case .vintageFilm:   return "Vintage Film"
        case .neonTrails:    return "Neon Trails"
        }
    }

    /// One-line description of the look, for tooltips or detail rows.
    public var description: String {
        switch self {
        case .cinematicWarm:
            return "Warm, gently lifted film grade with smooth contrast."
        case .cinematicCool:
            return "Cool, teal-leaning cinematic colour grade."
        case .vhs:
            return "Retro VHS tape look: chroma fringing, grain, and tracking jitter."
        case .noir:
            return "High-contrast monochrome black & white."
        case .cyberpunk:
            return "Saturated, cool, neon-edged futuristic look."
        case .dreamy:
            return "Soft, bright, low-contrast ethereal motion blur."
        case .highContrast:
            return "Punchy high-impact grade with crushed contrast."
        case .vintageFilm:
            return "Faded warm vintage film stock with fine grain."
        case .neonTrails:
            return "Glowing neon trails left behind moving subjects."
        }
    }

    // MARK: - Filter construction

    /// Build the ordered filter chain for this preset.
    ///
    /// Filters are returned in pipeline order (first element runs first). The
    /// returned instances are freshly allocated and owned by the caller; append
    /// them to a `MetalForgePipeline` or drive them directly.
    ///
    /// - Parameter engine: The shared Metal context used to compile each filter's
    ///   pipeline state.
    /// - Throws: Any error thrown while compiling an underlying filter's PSO
    ///   (see `MetalForgeError`).
    public func makeFilters(engine: MetalForgeEngine) throws -> [any MetalForgeFilter] {
        switch self {

        case .cinematicWarm:
            // Warm grade + soft corner darkening for a filmic frame.
            return [
                try colorCorrection(engine, exposure: 0.08, contrast: 1.15,
                                    saturation: 1.12, temperatureShift: 0.30),
                try vignette(engine, intensity: 0.30, radius: 0.80, softness: 0.50),
            ]

        case .cinematicCool:
            // Cool grade + soft vignette; saturation kept neutral.
            return [
                try colorCorrection(engine, exposure: 0.05, contrast: 1.15,
                                    saturation: 1.00, temperatureShift: -0.30),
                try vignette(engine, intensity: 0.30, radius: 0.80, softness: 0.50),
            ]

        case .vhs:
            // Tape look: scanlines, channel split, grain, tracking jitter, and a
            // touch of lens fringing over a slightly desaturated warm grade.
            return [
                try colorCorrection(engine, contrast: 1.05, saturation: 0.85,
                                    temperatureShift: 0.10),
                try rgbSplit(engine, red: SIMD2(0.004, 0.0), blue: SIMD2(-0.004, 0.0),
                             intensity: 0.6),
                try chromaticAberration(engine, redShift: SIMD2(0.004, 0.0),
                                        greenShift: SIMD2(-0.004, 0.0)),
                try horizontalJitter(engine, jitterIntensity: 0.012),
                try scanline(engine, intensity: 0.25, lineWidth: 2.0),
                try analogNoise(engine, noiseIntensity: 0.10),
            ]

        case .noir:
            // Fully desaturated, punchy, slightly darkened, edge-crisped, with a
            // strong vignette.
            return [
                try colorCorrection(engine, contrast: 1.40, saturation: 0.0),
                try adjustment(engine, brightness: -0.05, contrast: 1.10),
                try sharpen(engine, amount: 0.40),
                try vignette(engine, intensity: 0.50, radius: 0.70, softness: 0.45),
            ]

        case .cyberpunk:
            // Saturated cool grade, light edge crisp, channel split, neon trails.
            return [
                try colorCorrection(engine, exposure: 0.10, contrast: 1.25,
                                    saturation: 1.45, temperatureShift: -0.20),
                try sharpen(engine, amount: 0.40),
                try rgbSplit(engine, red: SIMD2(0.005, 0.0), blue: SIMD2(-0.005, 0.0),
                             intensity: 0.7),
                try neonTrails(engine, intensity: 1.0, decay: 0.85,
                               neonColor: SIMD3(1.0, 0.0, 0.8)),
            ]

        case .dreamy:
            // Soft warm grade, gentle spatial blur, and light temporal smoothing.
            return [
                try colorCorrection(engine, exposure: 0.15, contrast: 0.92,
                                    saturation: 1.08, temperatureShift: 0.12),
                try gaussianBlur(engine, radius: 3.0, intensity: 0.35),
                try motionBlur(engine, accumulationAlpha: 0.70),
            ]

        case .highContrast:
            // Crushed contrast, lifted saturation, crisp edges.
            return [
                try colorCorrection(engine, contrast: 1.50, saturation: 1.20),
                try adjustment(engine, contrast: 1.20),
                try sharpen(engine, amount: 0.80),
            ]

        case .vintageFilm:
            // Faded warm stock: gentle vignette, very faint scanlines, fine grain.
            return [
                try colorCorrection(engine, exposure: -0.05, contrast: 1.08,
                                    saturation: 0.85, temperatureShift: 0.25),
                try vignette(engine, intensity: 0.40, radius: 0.70, softness: 0.50),
                try scanline(engine, intensity: 0.08, lineWidth: 3.0),
                try analogNoise(engine, noiseIntensity: 0.07),
            ]

        case .neonTrails:
            // Boosted grade, subtle channel split, glowing motion trails.
            return [
                try colorCorrection(engine, contrast: 1.12, saturation: 1.30),
                try rgbSplit(engine, red: SIMD2(0.003, 0.0), blue: SIMD2(-0.003, 0.0),
                             intensity: 0.5),
                try neonTrails(engine, intensity: 1.20, decay: 0.90),
            ]
        }
    }

    // MARK: - Filter factory helpers
    //
    // Each helper builds one configured filter. Keeping the per-filter setup out
    // of the switch keeps each preset a short, readable list of intent.

    private func colorCorrection(
        _ engine: MetalForgeEngine,
        exposure: Float = 0.0,
        contrast: Float = 1.0,
        saturation: Float = 1.0,
        temperatureShift: Float = 0.0
    ) throws -> ColorCorrectionFilter {
        let f = try ColorCorrectionFilter(engine: engine)
        f.exposure         = exposure
        f.contrast         = contrast
        f.saturation       = saturation
        f.temperatureShift = temperatureShift
        return f
    }

    private func adjustment(
        _ engine: MetalForgeEngine,
        brightness: Float = 0.0,
        contrast: Float = 1.0
    ) throws -> AdjustmentFilter {
        let f = try AdjustmentFilter(engine: engine)
        f.brightness = brightness
        f.contrast   = contrast
        return f
    }

    private func vignette(
        _ engine: MetalForgeEngine,
        intensity: Float,
        radius: Float,
        softness: Float
    ) throws -> VignetteFilter {
        let f = try VignetteFilter(engine: engine)
        f.intensity = intensity
        f.radius    = radius
        f.softness  = softness
        return f
    }

    private func sharpen(
        _ engine: MetalForgeEngine,
        amount: Float
    ) throws -> SharpenFilter {
        let f = try SharpenFilter(engine: engine)
        f.amount = amount
        return f
    }

    private func gaussianBlur(
        _ engine: MetalForgeEngine,
        radius: Float,
        intensity: Float
    ) throws -> GaussianBlurFilter {
        let f = try GaussianBlurFilter(engine: engine)
        f.radius    = radius
        f.intensity = intensity
        return f
    }

    private func scanline(
        _ engine: MetalForgeEngine,
        intensity: Float,
        lineWidth: Float,
        timeSeed: Float = 0.0
    ) throws -> ScanlineFilter {
        let f = try ScanlineFilter(engine: engine)
        f.intensity = intensity
        f.lineWidth = lineWidth
        f.timeSeed  = timeSeed
        return f
    }

    private func rgbSplit(
        _ engine: MetalForgeEngine,
        red: SIMD2<Float>,
        green: SIMD2<Float> = SIMD2(0.0, 0.0),
        blue: SIMD2<Float>,
        intensity: Float
    ) throws -> RGBSplitFilter {
        let f = try RGBSplitFilter(engine: engine)
        f.redOffset   = red
        f.greenOffset = green
        f.blueOffset  = blue
        f.intensity   = intensity
        return f
    }

    private func motionBlur(
        _ engine: MetalForgeEngine,
        accumulationAlpha: Float
    ) throws -> MotionBlurFilter {
        let f = try MotionBlurFilter(engine: engine)
        f.accumulationAlpha = accumulationAlpha
        return f
    }

    private func neonTrails(
        _ engine: MetalForgeEngine,
        intensity: Float,
        decay: Float,
        neonColor: SIMD3<Float> = SIMD3(0.0, 0.8, 1.0)
    ) throws -> NeonTrailsFilter {
        let f = try NeonTrailsFilter(engine: engine)
        f.intensity = intensity
        f.decay     = decay
        f.neonColor = neonColor
        return f
    }

    private func chromaticAberration(
        _ engine: MetalForgeEngine,
        redShift: SIMD2<Float>,
        greenShift: SIMD2<Float>
    ) throws -> ChromaticAberrationFilter {
        let f = try ChromaticAberrationFilter(engine: engine)
        f.redShift   = redShift
        f.greenShift = greenShift
        return f
    }

    private func analogNoise(
        _ engine: MetalForgeEngine,
        noiseIntensity: Float
    ) throws -> AnalogNoiseFilter {
        let f = try AnalogNoiseFilter(engine: engine)
        f.noiseIntensity = noiseIntensity
        return f
    }

    private func horizontalJitter(
        _ engine: MetalForgeEngine,
        jitterIntensity: Float
    ) throws -> HorizontalJitterFilter {
        let f = try HorizontalJitterFilter(engine: engine)
        f.jitterIntensity = jitterIntensity
        return f
    }
}
