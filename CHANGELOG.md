# Changelog

## [0.2.0] - 2026-07-12

### Added
- Colour & tone filter pack: gamma, levels, hue rotate, vibrance, white balance,
  tone curve, highlight/shadow (+ tint), colour matrix, invert, monochrome, false
  colour, grayscale, sepia, haze, skin tone, posterize, luminance/adaptive
  threshold, colour halftone.
- Geometric distortion pack: bulge, pinch, stretch, swirl, sphere/glass refraction,
  transform, crop.
- Blend-mode pack: 19 blend modes plus alpha, chroma-key, and mask blends over a
  second input texture.
- Edge-detection pack: Sobel, Prewitt, Laplacian, Canny, Harris corners, non-max
  suppression, 3x3 convolution/emboss.
- Artistic pack: toon, smooth toon, sketch, threshold sketch, crosshatch, halftone,
  polka dot, Kuwahara, pixellate, polar pixellate, mosaic, CGA colourspace.
- Blur & morphology pack: box, directional-motion, zoom, tilt-shift, bilateral,
  median, lens, surface, and iOS blurs plus dilation/erosion/opening/closing.
- Temporal pack: frame blend, low/high pass, motion detector, optical-flow warp,
  frame interpolation over a temporal history buffer.
- GPU pixel-behavior tests for every new pack (173 tests total).

## [0.1.0] - 2026-06-06

### Added
- Initial MetalForge engine and filter pipeline.
- Real-time GPU processing with Metal compute shaders.
- CVPixelBuffer to MTLTexture pipeline.
- SDR and HDR-oriented processing stages.
- Color correction, LUT, glitch, analog, temporal, blur, sharpen, and stylization filters.
- Ready-to-use effect presets.
- SwiftUI/UIKit/AppKit preview support.
- AVFoundation capture and recording building blocks.
- Example camera app.
- Unit tests and pixel-behavior GPU tests.
- GitHub Actions CI.

### Notes
- This is an early 0.1.x release.
- APIs may change before 1.0.
