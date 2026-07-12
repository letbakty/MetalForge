# MetalForge v0.2.0

Seven new filter packs bring the catalogue to 90+ GPU filters, each covered by
pixel-behavior tests running on a real GPU.

## Highlights

- **Colour & tone pack** — gamma, levels, hue rotate, vibrance, white balance,
  tone curve, highlight/shadow, colour matrix, posterize, thresholds, and more
- **Geometric distortion pack** — bulge, pinch, stretch, swirl, sphere/glass
  refraction, transform, crop
- **Blend-mode pack** — 19 blend modes plus alpha, chroma-key, and mask blends
  over a second input texture
- **Edge-detection pack** — Sobel, Prewitt, Laplacian, Canny, Harris corners,
  non-max suppression, 3x3 convolution/emboss
- **Artistic pack** — toon, sketch, crosshatch, halftone, Kuwahara, pixellate,
  mosaic, CGA colourspace, and more
- **Blur & morphology pack** — box, motion, zoom, tilt-shift, bilateral, median,
  lens, surface, and iOS blurs plus dilation/erosion/opening/closing
- **Temporal pack** — frame blend, low/high pass, motion detector, optical-flow
  warp, frame interpolation over a temporal history buffer
- **173 GPU pixel-behavior tests** covering every new pack

## Requirements

- iOS 17+
- macOS 14+
- Swift 6
- Xcode 16+
- Metal-capable Apple device

## Notes

This is an early 0.2.x release. APIs may change before 1.0.
