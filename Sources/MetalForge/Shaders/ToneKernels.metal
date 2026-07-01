#include <metal_stdlib>
using namespace metal;

// ===========================================================================
// ToneKernels — v0.2.0 Color Pack (tone / grayscale / threshold half).
//
//   grayscaleKernel           luminance collapse
//   sepiaKernel               warm-brown matrix with intensity blend
//   hazeKernel                vertical haze/fog add-remove
//   skinToneKernel            hue-banded skin adjustment (pink ↔ green)
//   luminanceThresholdKernel  hard black/white cut
//   adaptiveThresholdKernel   threshold against a local box average
//   posterizeKernel           per-channel quantisation
//   colorHalftoneKernel       rotated-grid CMYK-style dots per RGB channel
//
// Same isHDR specialisation contract as the rest of MetalForge.
// ===========================================================================

constant bool isHDR [[function_constant(0)]];

constant float3 kBT709Luma  = float3(0.2126f, 0.7152f, 0.0722f);
constant float3 kBT2020Luma = float3(0.2627f, 0.6780f, 0.0593f);

static inline float toneLuma(float3 rgb) {
    return dot(saturate(rgb), isHDR ? kBT2020Luma : kBT709Luma);
}

static inline float3 toneFinish(float3 rgb) {
    return isHDR ? max(rgb, 0.0f) : saturate(rgb);
}

// ---------------------------------------------------------------------------
// 1. Grayscale
// ---------------------------------------------------------------------------

kernel void grayscaleKernel(
    texture2d<float, access::read>  src [[texture(0)]],
    texture2d<float, access::write> dst [[texture(1)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    const float4 c = src.read(gid);
    // Use the un-clamped luma so HDR highlights stay above 1.0.
    const float luma = dot(max(c.rgb, 0.0f), isHDR ? kBT2020Luma : kBT709Luma);
    dst.write(float4(toneFinish(float3(luma)), c.a), gid);
}

// ---------------------------------------------------------------------------
// 2. Sepia
// ---------------------------------------------------------------------------

struct SepiaUniforms {
    float intensity;   // 0 = identity … 1 = full sepia
};

kernel void sepiaKernel(
    texture2d<float, access::read>  src [[texture(0)]],
    texture2d<float, access::write> dst [[texture(1)]],
    constant SepiaUniforms&         u   [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    const float4 c = src.read(gid);
    const float3 rgb = saturate(c.rgb);

    // Classic sepia matrix (same coefficients as CISepiaTone).
    float3 sepia;
    sepia.r = dot(rgb, float3(0.393f, 0.769f, 0.189f));
    sepia.g = dot(rgb, float3(0.349f, 0.686f, 0.168f));
    sepia.b = dot(rgb, float3(0.272f, 0.534f, 0.131f));

    dst.write(float4(saturate(mix(rgb, sepia, u.intensity)), c.a), gid);
}

// ---------------------------------------------------------------------------
// 3. Haze
// ---------------------------------------------------------------------------

struct HazeUniforms {
    float distance;   // -0.3 … 0.3; >0 adds white haze, <0 removes it
    float slope;      // -0.3 … 0.3; vertical gradient of the haze amount
};

kernel void hazeKernel(
    texture2d<float, access::read>  src [[texture(0)]],
    texture2d<float, access::write> dst [[texture(1)]],
    constant HazeUniforms&          u   [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    const float4 c = src.read(gid);

    // Haze amount varies linearly down the frame (sky → ground).
    const float v = float(gid.y) / float(max(dst.get_height() - 1u, 1u));
    const float d = v * u.slope + u.distance;

    // Remove `d` worth of white veil and re-normalise the remaining contrast.
    const float3 white = float3(1.0f);
    float3 rgb = (c.rgb - d * white) / max(1.0f - d, 1e-4f);

    dst.write(float4(toneFinish(rgb), c.a), gid);
}

// ---------------------------------------------------------------------------
// 4. Skin tone
// ---------------------------------------------------------------------------

struct SkinToneUniforms {
    float adjust;            // -1 … +1; effect direction set by upperTone
    float skinHue;           // centre of the skin band (≈0.05 in hue turns)
    float skinHueThreshold;  // band width multiplier (≈40)
    float upperTone;         // 0 = shift toward pink, 1 = toward green
};

kernel void skinToneKernel(
    texture2d<float, access::read>  src [[texture(0)]],
    texture2d<float, access::write> dst [[texture(1)]],
    constant SkinToneUniforms&      u   [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    const float4 c = src.read(gid);
    const float3 rgb = saturate(c.rgb);

    // --- RGB → HSV hue (in turns, [0,1)) ---
    const float mx = max3(rgb.r, rgb.g, rgb.b);
    const float mn = min3(rgb.r, rgb.g, rgb.b);
    const float delta = mx - mn;
    float hue = 0.0f;
    if (delta > 1e-5f) {
        if (mx == rgb.r)      hue = fmod((rgb.g - rgb.b) / delta, 6.0f);
        else if (mx == rgb.g) hue = (rgb.b - rgb.r) / delta + 2.0f;
        else                  hue = (rgb.r - rgb.g) / delta + 4.0f;
        hue /= 6.0f;
        if (hue < 0.0f) hue += 1.0f;
    }

    // Gaussian-ish weight centred on the skin hue band.
    const float dist = min(abs(hue - u.skinHue), 1.0f - abs(hue - u.skinHue));
    const float w = exp(-dist * dist * u.skinHueThreshold * u.skinHueThreshold * 0.5f) ;

    // Pink: push red & blue up slightly; green: push green up. Weighted by
    // band membership and saturation (greys are untouched).
    const float satW = delta > 1e-5f ? saturate(delta * 4.0f) : 0.0f;
    const float amount = u.adjust * w * satW * 0.25f;

    float3 result = rgb;
    if (u.upperTone < 0.5f) {
        result.r += amount;
        result.b += amount * 0.5f;
    } else {
        result.g += amount;
    }

    dst.write(float4(saturate(result), c.a), gid);
}

// ---------------------------------------------------------------------------
// 5. Luminance threshold
// ---------------------------------------------------------------------------

struct LuminanceThresholdUniforms {
    float threshold;   // 0 … 1
};

kernel void luminanceThresholdKernel(
    texture2d<float, access::read>       src [[texture(0)]],
    texture2d<float, access::write>      dst [[texture(1)]],
    constant LuminanceThresholdUniforms& u   [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    const float4 c = src.read(gid);
    const float v = step(u.threshold, toneLuma(c.rgb));
    dst.write(float4(v, v, v, c.a), gid);
}

// ---------------------------------------------------------------------------
// 6. Adaptive threshold
// ---------------------------------------------------------------------------

struct AdaptiveThresholdUniforms {
    float radius;   // local-average box radius in pixels (sampled sparsely)
    float bias;     // subtracted from the local mean before comparing
};

kernel void adaptiveThresholdKernel(
    texture2d<float, access::read>      src [[texture(0)]],
    texture2d<float, access::write>     dst [[texture(1)]],
    constant AdaptiveThresholdUniforms& u   [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;

    const int w = int(dst.get_width());
    const int h = int(dst.get_height());

    // 9×9 sparse box sample of local luminance. `step` spreads the taps over
    // the requested radius so large radii don't cost more.
    const float stepPx = max(u.radius / 4.0f, 1.0f);
    float sum = 0.0f;
    for (int j = -4; j <= 4; ++j) {
        for (int i = -4; i <= 4; ++i) {
            const int2 p = int2(
                clamp(int(gid.x) + int(round(float(i) * stepPx)), 0, w - 1),
                clamp(int(gid.y) + int(round(float(j) * stepPx)), 0, h - 1));
            sum += toneLuma(src.read(uint2(p)).rgb);
        }
    }
    const float mean = sum / 81.0f;

    const float4 c = src.read(gid);
    const float v = step(mean - u.bias, toneLuma(c.rgb));
    dst.write(float4(v, v, v, c.a), gid);
}

// ---------------------------------------------------------------------------
// 7. Posterize
// ---------------------------------------------------------------------------

struct PosterizeUniforms {
    float levels;   // 2 … 256; number of tonal steps per channel
};

kernel void posterizeKernel(
    texture2d<float, access::read>  src [[texture(0)]],
    texture2d<float, access::write> dst [[texture(1)]],
    constant PosterizeUniforms&     u   [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    const float4 c = src.read(gid);
    const float3 base      = saturate(c.rgb);
    const float3 highlight = isHDR ? max(c.rgb - 1.0f, 0.0f) : float3(0.0f);
    const float3 rgb = floor(base * u.levels + 0.5f) / u.levels;
    dst.write(float4(toneFinish(rgb + highlight), c.a), gid);
}

// ---------------------------------------------------------------------------
// 8. Colour halftone (rotated dot grids per channel)
// ---------------------------------------------------------------------------

struct ColorHalftoneUniforms {
    float dotSizePx;   // grid period in pixels
};

/// Dot coverage of one channel: rotate the pixel grid by `angle`, find the
/// nearest dot centre, read the channel there, and compare the distance to
/// the radius that would reproduce that intensity as ink coverage.
static inline float halftoneChannel(
    texture2d<float, access::read> src,
    float2 pos, float angle, float dotSize, int channel, int w, int h)
{
    const float cosA = cos(angle);
    const float sinA = sin(angle);

    // Rotate into grid space, snap to the nearest cell centre, rotate back.
    const float2 rotated = float2( pos.x * cosA + pos.y * sinA,
                                  -pos.x * sinA + pos.y * cosA);
    const float2 cellCentre = (floor(rotated / dotSize) + 0.5f) * dotSize;
    const float2 centrePx = float2(cellCentre.x * cosA - cellCentre.y * sinA,
                                   cellCentre.x * sinA + cellCentre.y * cosA);

    const int2 p = int2(clamp(int(round(centrePx.x)), 0, w - 1),
                        clamp(int(round(centrePx.y)), 0, h - 1));
    const float4 sample = src.read(uint2(p));
    const float value = saturate(channel == 0 ? sample.r : (channel == 1 ? sample.g : sample.b));

    // Brighter value → larger dot of the channel colour. Radius covers the
    // cell diagonal at value 1 so full intensity yields full coverage.
    const float radius = dotSize * 0.7071f * sqrt(value);
    const float distToCentre = length(rotated - cellCentre);
    return 1.0f - smoothstep(radius - 1.0f, radius + 1.0f, distToCentre);
}

kernel void colorHalftoneKernel(
    texture2d<float, access::read>  src [[texture(0)]],
    texture2d<float, access::write> dst [[texture(1)]],
    constant ColorHalftoneUniforms& u   [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;

    const int w = int(dst.get_width());
    const int h = int(dst.get_height());
    const float2 pos = float2(gid);
    const float dotSize = max(u.dotSizePx, 2.0f);

    // Print-style screen angles (degrees → radians): R 15°, G 75°, B 0°.
    const float3 rgb = float3(
        halftoneChannel(src, pos, 0.2618f, dotSize, 0, w, h),
        halftoneChannel(src, pos, 1.3090f, dotSize, 1, w, h),
        halftoneChannel(src, pos, 0.0f,    dotSize, 2, w, h));

    const float4 c = src.read(gid);
    dst.write(float4(rgb, c.a), gid);
}
