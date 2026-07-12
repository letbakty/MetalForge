# File Map

A file-by-file guide to the repository, grouped by area. See
[`ARCHITECTURE.md`](ARCHITECTURE.md) for how these pieces relate.

## Core

- `Sources/MetalForge/MetalForge.swift` — top-level namespace enum and usage doc.
- `Sources/MetalForge/MetalForgeEngine.swift` — `MTLDevice`, command queue,
  texture caches, centralized shader-library loading, function resolvers, and
  `CVPixelBuffer` → `MTLTexture` wrapping.
- `Sources/MetalForge/TexturePool.swift` — pooled reuse of intermediate
  textures.
- `Sources/MetalForge/MetalForgeError.swift` — error type.
- `Sources/MetalForge/MetalForgeColorSpace.swift` — SDR / HDR color-space enum.

## Pipeline

- `Sources/MetalForge/MetalForgePipeline.swift` — ordered filter chain,
  per-frame processing, automatic conversion/HDR stage insertion, recycling.
- `Sources/MetalForge/MetalForgeFilter.swift` — `MetalForgeFilter` and
  `MetalForgeSourceFilter` protocols and their lifecycle contract.

## Conversion and HDR stages

- `Sources/MetalForge/YUVToRGBConverter.swift` — bi-planar YUV → RGB source
  filter.
- `Sources/MetalForge/RGBToYUVConverter.swift` — RGB → YUV for recording.
- `Sources/MetalForge/YUVColorMatrices.swift` — color matrices by standard.
- `Sources/MetalForge/HDRDecodeFilter.swift` — PQ/HLG → linear.
- `Sources/MetalForge/HDREncodeFilter.swift` — linear → PQ/HLG.

## Filters

- `Sources/MetalForge/ColorGradingFilters.swift` — `ColorCorrectionFilter` and
  `MetalForgeLUTFilter` (3D LUT with built-in presets).
- `Sources/MetalForge/AdjustmentFilter.swift` — `AdjustmentFilter`
  (brightness / contrast).
- `Sources/MetalForge/GlitchFilter.swift` — `GlitchFilter`.
- `Sources/MetalForge/AnalogDistortionFilters.swift` —
  `ChromaticAberrationFilter`, `AnalogNoiseFilter`, `HorizontalJitterFilter`.
- `Sources/MetalForge/TemporalEffectsFilters.swift` — `MotionBlurFilter`,
  `NeonTrailsFilter`.
- `Sources/MetalForge/BlurFilters.swift` — `GaussianBlurFilter` (separable,
  two-pass), `SharpenFilter`.
- `Sources/MetalForge/StylizationFilters.swift` — `VignetteFilter`,
  `ScanlineFilter`, `RGBSplitFilter`.
- `Sources/MetalForge/ColorAdjustmentFilters.swift` — gamma, levels, hue rotate,
  vibrance, white balance, tone curve, highlight/shadow (+ tint), colour matrix,
  invert, monochrome, false colour.
- `Sources/MetalForge/ToneFilters.swift` — grayscale, sepia, haze, skin tone,
  luminance/adaptive threshold, posterize, colour halftone.
- `Sources/MetalForge/BlurMorphologyFilters.swift` — box / directional-motion /
  zoom / tilt-shift / bilateral / median / lens / surface / iOS blurs and
  dilation / erosion / opening / closing morphology.
- `Sources/MetalForge/DistortionFilters.swift` — bulge, pinch, stretch, swirl,
  sphere / glass refraction, transform, crop.
- `Sources/MetalForge/EdgeDetectionFilters.swift` — Sobel, Prewitt, Laplacian,
  Canny, Harris corners, non-max suppression, 3x3 convolution / emboss.
- `Sources/MetalForge/ArtisticFilters.swift` — toon, smooth toon, sketch,
  threshold sketch, crosshatch, halftone, polka dot, Kuwahara, pixellate, polar
  pixellate, mosaic, CGA colourspace.
- `Sources/MetalForge/BlendFilters.swift` — `BlendModeFilter` (19 modes),
  `AlphaBlendFilter`, `ChromaKeyFilter`, `MaskBlendFilter` (all take a second
  input texture).
- `Sources/MetalForge/TemporalPackFilters.swift` — frame blend, low / high pass,
  motion detector, optical-flow warp, frame interpolation over a history buffer.

## Effect presets

- `Sources/MetalForge/MetalForgeEffectPreset.swift` — `MetalForgeEffectPreset`
  enum (nine curated looks) with `displayName`, `description`, and
  `makeFilters(engine:)`.
- `Sources/MetalForge/MetalForgePipeline+Presets.swift` —
  `MetalForgePipeline.applyPreset(_:)`, which replaces the current filter chain
  with a preset's chain.

## Shaders

`Sources/MetalForge/Shaders/` (compiled to `default.metallib` under Xcode,
compiled at runtime under SwiftPM):

- `DisplayRenderShader.metal` — preview render.
- `YUVConverterShader.metal` — YUV → RGB.
- `RGBToYUVShader.metal` — RGB → YUV.
- `HDRTransferShader.metal` — PQ/HLG transfer functions.
- `ColorGradingKernels.metal` — color correction and LUT.
- `AdjustmentShader.metal` — brightness / contrast.
- `GlitchShader.metal` — glitch effect.
- `AnalogKernels.metal` — chromatic aberration, noise, jitter.
- `TemporalKernels.metal` — motion blur, neon trails.
- `BlurKernels.metal` — separable Gaussian blur passes and sharpen.
- `StylizationKernels.metal` — vignette, scanlines, RGB split.
- `ColorAdjustmentKernels.metal` — colour-adjustment pack kernels.
- `ToneKernels.metal` — tone pack kernels.
- `BlurMorphologyKernels.metal` — extended blur and morphology kernels.
- `DistortionKernels.metal` — geometric distortion kernels.
- `EdgeKernels.metal` — edge-detection and convolution kernels.
- `ArtisticKernels.metal` — artistic and halftone kernels.
- `BlendKernels.metal` — blend-mode, alpha, chroma-key, and mask kernels.
- `TemporalPackKernels.metal` — temporal history-buffer kernels.

## Capture

- `Sources/MetalForge/MetalForgeCaptureManager.swift` — adapts
  `AVCaptureSession` video/audio output into frame callbacks.

## Preview

- `Sources/MetalForge/MetalForgeView.swift` — `MTKView` subclass that presents a
  processed texture.
- `Sources/MetalForge/MetalForgeViewRepresentable.swift` — SwiftUI / UIKit /
  AppKit wrapper.

## Recording

- `Sources/MetalForge/MetalForgeRecorder.swift` — `AVAssetWriter`-backed
  recorder with optional best-effort passthrough audio.

## Demo (in-library reference)

- `Sources/MetalForge/MetalForgeDemoView.swift` — `MetalForgeDemoController` and
  `MetalForgeDemoView`, a capture → display → record reference implementation.

## Tests

- `Tests/MetalForgeTests/MetalForgeTests.swift` — build, unit, preset, and
  GPU pixel-behavior tests.
- `Tests/MetalForgeTests/MetalForgePixelTestHelpers.swift` — helpers for
  pixel-behavior tests: known-data textures, GPU readback, tolerance compares.
- `Tests/MetalForgeTests/ColorPackTests.swift` — colour & tone pack tests.
- `Tests/MetalForgeTests/DistortionPackTests.swift` — distortion pack tests.
- `Tests/MetalForgeTests/BlendPackTests.swift` — blend-mode pack tests.
- `Tests/MetalForgeTests/EdgePackTests.swift` — edge-detection pack tests.
- `Tests/MetalForgeTests/ArtisticPackTests.swift` — artistic pack tests.
- `Tests/MetalForgeTests/BlurMorphologyPackTests.swift` — blur & morphology pack tests.
- `Tests/MetalForgeTests/TemporalPackTests.swift` — temporal pack tests.

## Example app

`Examples/MetalForgeCamera/` — minimal SwiftUI iOS camera demo:

- `MetalForgeCameraApp.swift` — `@main` app entry.
- `ContentView.swift` — root layout (preview + HUD + controls).
- `CameraPreviewView.swift` — `MetalForgeViewRepresentable` wrapper.
- `CameraViewModel.swift` — owns engine + pipeline + capture; UI state.
- `FilterControlPanel.swift` — bottom control panel.
- `Info.plist` — camera/microphone usage descriptions.
- `Assets.xcassets/` — app icon and accent color.
- `MetalForgeCamera.xcodeproj` — Xcode project (local SwiftPM dependency
  via relative `../..` path).

## CI

- `.github/workflows/ci.yml` — on push and pull requests targeting `main` and
  `develop`: builds and tests the Swift package, and builds the example app via
  `xcodebuild`.
