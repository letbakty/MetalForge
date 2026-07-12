#include <metal_stdlib>
using namespace metal;

// ===========================================================================
// BlurMorphologyKernels — v0.7.0 Advanced Blur & Morphology Pack.
//
//   boxBlurHorizontalKernel / boxBlurVerticalKernel   separable box blur
//   directionalBlurKernel                             line blur along an angle
//   zoomBlurKernel                                    radial streaks toward a centre
//   tiltShiftKernel                                   focus band + disc blur
//   bilateralBlurKernel                               edge-preserving smoothing
//   medianBlurKernel                                  3×3 luminance-median denoise
//   lensBlurKernel                                    hexagonal bokeh
//   surfaceBlurKernel                                 threshold-gated box smoothing
//   frostedGlassKernel                                iOS-style frosted finish pass
//   morphologyHorizontalKernel / morphologyVerticalKernel  separable min/max
//
// Blurs run on the working values directly (HDR highlights spread naturally);
// the isHDR constant gates only the final clamp.
// ===========================================================================

constant bool isHDR [[function_constant(0)]];

constant float3 kLumaW = float3(0.2126f, 0.7152f, 0.0722f);

static inline float4 readClamped(texture2d<float, access::read> t, int x, int y) {
    const int w = int(t.get_width());
    const int h = int(t.get_height());
    return t.read(uint2(clamp(x, 0, w - 1), clamp(y, 0, h - 1)));
}

static inline float4 blurFinish(float4 c) {
    return isHDR ? max(c, 0.0f) : saturate(c);
}

// ---------------------------------------------------------------------------
// 1. Box blur (separable)
// ---------------------------------------------------------------------------

struct BoxBlurUniforms {
    float radius;   // taps each side
};

kernel void boxBlurHorizontalKernel(
    texture2d<float, access::read>  src [[texture(0)]],
    texture2d<float, access::write> dst [[texture(1)]],
    constant BoxBlurUniforms&       u   [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    const int r = clamp(int(u.radius), 1, 32);
    float4 acc = float4(0.0f);
    for (int i = -r; i <= r; ++i) {
        acc += readClamped(src, int(gid.x) + i, int(gid.y));
    }
    dst.write(acc / float(2 * r + 1), gid);
}

kernel void boxBlurVerticalKernel(
    texture2d<float, access::read>  src [[texture(0)]],
    texture2d<float, access::write> dst [[texture(1)]],
    constant BoxBlurUniforms&       u   [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    const int r = clamp(int(u.radius), 1, 32);
    float4 acc = float4(0.0f);
    for (int j = -r; j <= r; ++j) {
        acc += readClamped(src, int(gid.x), int(gid.y) + j);
    }
    dst.write(blurFinish(acc / float(2 * r + 1)), gid);
}

// ---------------------------------------------------------------------------
// 2. Directional motion blur
// ---------------------------------------------------------------------------

struct DirectionalBlurUniforms {
    float angle;    // radians
    float radius;   // streak half-length in pixels
};

kernel void directionalBlurKernel(
    texture2d<float, access::read>  src [[texture(0)]],
    texture2d<float, access::write> dst [[texture(1)]],
    constant DirectionalBlurUniforms& u [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    const float2 dir = float2(cos(u.angle), sin(u.angle));
    const int taps = clamp(int(u.radius), 1, 32);

    float4 acc = float4(0.0f);
    for (int i = -taps; i <= taps; ++i) {
        const float t = float(i) * u.radius / float(taps);
        acc += readClamped(src,
                           int(round(float(gid.x) + dir.x * t)),
                           int(round(float(gid.y) + dir.y * t)));
    }
    dst.write(blurFinish(acc / float(2 * taps + 1)), gid);
}

// ---------------------------------------------------------------------------
// 3. Zoom blur
// ---------------------------------------------------------------------------

struct ZoomBlurUniforms {
    float2 center;   // normalised
    float  strength; // 0 = identity … 1 = strong streaks
};

kernel void zoomBlurKernel(
    texture2d<float, access::read>  src [[texture(0)]],
    texture2d<float, access::write> dst [[texture(1)]],
    constant ZoomBlurUniforms&      u   [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    const float2 size = float2(dst.get_width(), dst.get_height());
    const float2 pos = float2(gid);
    const float2 centrePx = u.center * size;

    // March from the pixel toward the centre; equal weights give the classic
    // radial streak.
    const int taps = 16;
    float4 acc = float4(0.0f);
    for (int i = 0; i < taps; ++i) {
        const float t = u.strength * float(i) / float(taps - 1) * 0.3f;
        const float2 p = mix(pos, centrePx, t);
        acc += readClamped(src, int(round(p.x)), int(round(p.y)));
    }
    dst.write(blurFinish(acc / float(taps)), gid);
}

// ---------------------------------------------------------------------------
// 4. Tilt-shift
// ---------------------------------------------------------------------------

struct TiltShiftUniforms {
    float focusCenter;   // normalised position of the sharp band (along axis)
    float focusWidth;    // half-width of the fully sharp band
    float falloff;       // width of the sharp→blurred transition
    float blurRadius;    // disc blur radius in pixels at full blur
    float vertical;      // 0 = horizontal band (y axis), 1 = vertical band
};

kernel void tiltShiftKernel(
    texture2d<float, access::read>  src [[texture(0)]],
    texture2d<float, access::write> dst [[texture(1)]],
    constant TiltShiftUniforms&     u   [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    const float2 size = float2(dst.get_width(), dst.get_height());
    const float coord = (u.vertical > 0.5f)
        ? (float(gid.x) + 0.5f) / size.x
        : (float(gid.y) + 0.5f) / size.y;

    // Blur weight: 0 inside the focus band, ramping to 1 over `falloff`.
    const float d = abs(coord - u.focusCenter);
    const float w = smoothstep(u.focusWidth, u.focusWidth + max(u.falloff, 1e-3f), d);

    const float4 sharp = src.read(gid);
    if (w < 1e-3f) { dst.write(blurFinish(sharp), gid); return; }

    // 12-tap Poisson-ish disc blur scaled by the blur weight.
    constexpr float2 taps[12] = {
        float2(-0.326f, -0.406f), float2(-0.840f, -0.074f),
        float2(-0.696f,  0.457f), float2(-0.203f,  0.621f),
        float2( 0.962f, -0.195f), float2( 0.473f, -0.480f),
        float2( 0.519f,  0.767f), float2( 0.185f, -0.893f),
        float2( 0.507f,  0.064f), float2( 0.896f,  0.412f),
        float2(-0.322f, -0.933f), float2(-0.792f, -0.598f),
    };
    const float r = u.blurRadius * w;
    float4 acc = sharp;
    for (int i = 0; i < 12; ++i) {
        acc += readClamped(src,
                           int(round(float(gid.x) + taps[i].x * r)),
                           int(round(float(gid.y) + taps[i].y * r)));
    }
    const float4 blurred = acc / 13.0f;
    dst.write(blurFinish(mix(sharp, blurred, w)), gid);
}

// ---------------------------------------------------------------------------
// 5. Bilateral blur
// ---------------------------------------------------------------------------

struct BilateralUniforms {
    float radius;            // spatial radius in pixels
    float rangeSigma;        // colour-distance falloff (smaller = edge-stricter)
};

kernel void bilateralBlurKernel(
    texture2d<float, access::read>  src [[texture(0)]],
    texture2d<float, access::write> dst [[texture(1)]],
    constant BilateralUniforms&     u   [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    const int r = clamp(int(u.radius), 1, 6);
    const float4 centre = src.read(gid);
    const float spatialSigma = max(u.radius * 0.5f, 0.5f);
    const float rs2 = 2.0f * u.rangeSigma * u.rangeSigma;
    const float ss2 = 2.0f * spatialSigma * spatialSigma;

    float4 acc = float4(0.0f);
    float wsum = 0.0f;
    for (int j = -r; j <= r; ++j) {
        for (int i = -r; i <= r; ++i) {
            const float4 s = readClamped(src, int(gid.x) + i, int(gid.y) + j);
            const float colorDist = distance_squared(s.rgb, centre.rgb);
            const float w = exp(-float(i * i + j * j) / ss2 - colorDist / rs2);
            acc += s * w;
            wsum += w;
        }
    }
    dst.write(blurFinish(acc / max(wsum, 1e-5f)), gid);
}

// ---------------------------------------------------------------------------
// 6. Median blur (3×3, luminance-ranked)
// ---------------------------------------------------------------------------

kernel void medianBlurKernel(
    texture2d<float, access::read>  src [[texture(0)]],
    texture2d<float, access::write> dst [[texture(1)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;

    // Pick the neighbourhood pixel with the median *luminance* and output its
    // full colour — avoids per-channel medians inventing colours that never
    // existed in the input.
    float4 colors[9];
    float lumas[9];
    int n = 0;
    for (int j = -1; j <= 1; ++j) {
        for (int i = -1; i <= 1; ++i) {
            colors[n] = readClamped(src, int(gid.x) + i, int(gid.y) + j);
            lumas[n] = dot(colors[n].rgb, kLumaW);
            ++n;
        }
    }
    // Insertion sort of 9 — branch-light enough on GPU for a 3×3 window.
    for (int a = 1; a < 9; ++a) {
        const float lv = lumas[a];
        const float4 cv = colors[a];
        int b = a - 1;
        while (b >= 0 && lumas[b] > lv) {
            lumas[b + 1] = lumas[b];
            colors[b + 1] = colors[b];
            --b;
        }
        lumas[b + 1] = lv;
        colors[b + 1] = cv;
    }
    dst.write(blurFinish(colors[4]), gid);
}

// ---------------------------------------------------------------------------
// 7. Lens blur (hexagonal bokeh)
// ---------------------------------------------------------------------------

struct LensBlurUniforms {
    float radius;       // bokeh radius in pixels
    float brightness;   // highlight boost exponent driver (0 = energy-true)
};

kernel void lensBlurKernel(
    texture2d<float, access::read>  src [[texture(0)]],
    texture2d<float, access::write> dst [[texture(1)]],
    constant LensBlurUniforms&      u   [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;

    // Sample 3 concentric hexagon rings (1 + 6 + 12 + 18 = 37 taps). Weighting
    // each sample by a power of its luminance makes highlights "win", which is
    // what reads as bokeh rather than mush.
    const float gamma = 1.0f + u.brightness * 3.0f;
    float4 acc = float4(0.0f);
    float wsum = 0.0f;

    for (int ring = 0; ring <= 3; ++ring) {
        const int count = (ring == 0) ? 1 : 6 * ring;
        const float rr = u.radius * float(ring) / 3.0f;
        for (int k = 0; k < count; ++k) {
            // Hexagonal ring: vertices + edge subdivisions of a hexagon.
            const float a = 2.0f * M_PI_F * float(k) / float(max(count, 1));
            // Map the circle to a hexagon: scale by hex radius at this angle.
            const float sectorA = fmod(a, M_PI_F / 3.0f) - M_PI_F / 6.0f;
            const float hexScale = (M_PI_F / 6.0f) / max(cos(sectorA), 1e-3f) * (3.0f / M_PI_F) * 0.5f + 0.5f;
            const float2 offset = float2(cos(a), sin(a)) * rr * hexScale;
            const float4 s = readClamped(src,
                                         int(round(float(gid.x) + offset.x)),
                                         int(round(float(gid.y) + offset.y)));
            const float w = pow(max(dot(s.rgb, kLumaW), 1e-3f), gamma);
            acc += s * w;
            wsum += w;
        }
    }
    dst.write(blurFinish(acc / max(wsum, 1e-5f)), gid);
}

// ---------------------------------------------------------------------------
// 8. Surface blur
// ---------------------------------------------------------------------------

struct SurfaceBlurUniforms {
    float radius;      // box radius in pixels
    float threshold;   // max luminance difference to participate
};

kernel void surfaceBlurKernel(
    texture2d<float, access::read>  src [[texture(0)]],
    texture2d<float, access::write> dst [[texture(1)]],
    constant SurfaceBlurUniforms&   u   [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    const int r = clamp(int(u.radius), 1, 6);
    const float4 centre = src.read(gid);
    const float centreLuma = dot(saturate(centre.rgb), kLumaW);

    // Photoshop-style surface blur: only neighbours within the luminance
    // threshold contribute — texture smooths, edges stay.
    float4 acc = centre;
    float count = 1.0f;
    for (int j = -r; j <= r; ++j) {
        for (int i = -r; i <= r; ++i) {
            if (i == 0 && j == 0) continue;
            const float4 s = readClamped(src, int(gid.x) + i, int(gid.y) + j);
            if (abs(dot(saturate(s.rgb), kLumaW) - centreLuma) < u.threshold) {
                acc += s;
                count += 1.0f;
            }
        }
    }
    dst.write(blurFinish(acc / count), gid);
}

// ---------------------------------------------------------------------------
// 9. Frosted-glass finish (iOSBlur final pass)
// ---------------------------------------------------------------------------

struct FrostedUniforms {
    float saturationBoost;   // UIKit blurs re-saturate (≈1.8)
    float whiteMix;          // light material whitening (≈0.2)
};

kernel void frostedGlassKernel(
    texture2d<float, access::read>  src [[texture(0)]],
    texture2d<float, access::write> dst [[texture(1)]],
    constant FrostedUniforms&       u   [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    const float4 c = src.read(gid);

    // Saturation boost compensates the desaturation a strong blur causes,
    // then a touch of white gives the "light material" frosting.
    const float luma = dot(saturate(c.rgb), kLumaW);
    float3 rgb = mix(float3(luma), saturate(c.rgb), u.saturationBoost);
    rgb = mix(rgb, float3(1.0f), u.whiteMix);

    dst.write(blurFinish(float4(rgb, c.a)), gid);
}

// ---------------------------------------------------------------------------
// 10. Morphology (separable square structuring element)
// ---------------------------------------------------------------------------

struct MorphologyUniforms {
    float radius;     // structuring element half-size
    float isDilate;   // 1 = max (dilate), 0 = min (erode)
};

kernel void morphologyHorizontalKernel(
    texture2d<float, access::read>  src [[texture(0)]],
    texture2d<float, access::write> dst [[texture(1)]],
    constant MorphologyUniforms&    u   [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    const int r = clamp(int(u.radius), 1, 16);
    float4 best = src.read(gid);
    for (int i = -r; i <= r; ++i) {
        const float4 s = readClamped(src, int(gid.x) + i, int(gid.y));
        best = (u.isDilate > 0.5f) ? max(best, s) : min(best, s);
    }
    dst.write(best, gid);
}

kernel void morphologyVerticalKernel(
    texture2d<float, access::read>  src [[texture(0)]],
    texture2d<float, access::write> dst [[texture(1)]],
    constant MorphologyUniforms&    u   [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    const int r = clamp(int(u.radius), 1, 16);
    float4 best = src.read(gid);
    for (int j = -r; j <= r; ++j) {
        const float4 s = readClamped(src, int(gid.x), int(gid.y) + j);
        best = (u.isDilate > 0.5f) ? max(best, s) : min(best, s);
    }
    dst.write(blurFinish(best), gid);
}
