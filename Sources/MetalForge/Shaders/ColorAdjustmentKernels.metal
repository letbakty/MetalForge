#include <metal_stdlib>
using namespace metal;

// ===========================================================================
// ColorAdjustmentKernels — v0.2.0 Color Pack (correction half).
//
//   gammaKernel               per-channel power curve
//   levelsKernel              input/output black-white points + mid gamma
//   hueRotateKernel           hue rotation via YIQ chroma-plane rotation
//   vibranceKernel            smart saturation (protects already-vivid pixels)
//   whiteBalanceKernel        temperature / tint push
//   toneCurveKernel           256-entry per-channel 1D LUT (CPU-baked spline)
//   highlightShadowKernel     luminance-banded lift / recovery
//   highlightShadowTintKernel luminance-banded colour tinting
//   colorMatrixKernel         arbitrary 4×4 colour matrix with blend
//   colorInvertKernel         1 - rgb
//   monochromeKernel          luma re-tinted by a filter colour
//   falseColorKernel          two-colour luminance gradient map
//
// All kernels are `isHDR`-specialised: the SDR variant saturates the result,
// the HDR variant only guards against negative values so above-white light
// can propagate (matching ColorGradingKernels.metal).
// ===========================================================================

constant bool isHDR [[function_constant(0)]];

constant float3 kBT709Luma  = float3(0.2126f, 0.7152f, 0.0722f);
constant float3 kBT2020Luma = float3(0.2627f, 0.6780f, 0.0593f);

static inline float3 caLuma3() { return isHDR ? kBT2020Luma : kBT709Luma; }

/// Common output clamp: SDR saturates, HDR only forbids negative light.
static inline float3 caFinish(float3 rgb) {
    return isHDR ? max(rgb, 0.0f) : saturate(rgb);
}

// ---------------------------------------------------------------------------
// 1. Gamma
// ---------------------------------------------------------------------------

struct GammaUniforms {
    float gamma;     // 1 = identity; <1 brightens mids, >1 darkens mids
};

kernel void gammaKernel(
    texture2d<float, access::read>  src [[texture(0)]],
    texture2d<float, access::write> dst [[texture(1)]],
    constant GammaUniforms&         u   [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    const float4 c = src.read(gid);
    // pow() is undefined for negative bases — clamp the working value first.
    float3 rgb = pow(max(c.rgb, 0.0f), float3(u.gamma));
    dst.write(float4(caFinish(rgb), c.a), gid);
}

// ---------------------------------------------------------------------------
// 2. Levels
// ---------------------------------------------------------------------------

struct LevelsUniforms {
    float3 inMin;    float pad0;
    float3 inMax;    float pad1;
    float3 outMin;   float pad2;
    float3 outMax;   float pad3;
    float3 gamma;    float pad4;
};

kernel void levelsKernel(
    texture2d<float, access::read>  src [[texture(0)]],
    texture2d<float, access::write> dst [[texture(1)]],
    constant LevelsUniforms&        u   [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    const float4 c = src.read(gid);
    // Classic Photoshop levels: normalise input range, apply mid gamma,
    // remap into the output range.
    const float3 span = max(u.inMax - u.inMin, float3(1e-4f));
    float3 norm = saturate((c.rgb - u.inMin) / span);
    norm = pow(norm, 1.0f / max(u.gamma, float3(1e-4f)));
    float3 rgb = u.outMin + norm * (u.outMax - u.outMin);
    // HDR: levels is display-referred by definition; still let outMax > 1 pass.
    dst.write(float4(isHDR ? max(rgb, 0.0f) : saturate(rgb), c.a), gid);
}

// ---------------------------------------------------------------------------
// 3. Hue rotate (YIQ chroma-plane rotation)
// ---------------------------------------------------------------------------

struct HueRotateUniforms {
    float angle;     // radians; 0 = identity
};

kernel void hueRotateKernel(
    texture2d<float, access::read>  src [[texture(0)]],
    texture2d<float, access::write> dst [[texture(1)]],
    constant HueRotateUniforms&     u   [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    const float4 c = src.read(gid);
    const float3 rgbIn = c.rgb;

    // RGB → YIQ via dot products (avoids matrix column-order pitfalls).
    const float y = dot(rgbIn, float3(0.299f,  0.587f,  0.114f));
    const float i = dot(rgbIn, float3(0.596f, -0.274f, -0.322f));
    const float q = dot(rgbIn, float3(0.211f, -0.523f,  0.312f));

    // Rotate the IQ chroma plane.
    const float cosA = cos(u.angle);
    const float sinA = sin(u.angle);
    const float i2 = i * cosA - q * sinA;
    const float q2 = i * sinA + q * cosA;

    // YIQ → RGB.
    float3 rgb;
    rgb.r = y + 0.956f * i2 + 0.621f * q2;
    rgb.g = y - 0.272f * i2 - 0.647f * q2;
    rgb.b = y - 1.106f * i2 + 1.703f * q2;

    dst.write(float4(caFinish(rgb), c.a), gid);
}

// ---------------------------------------------------------------------------
// 4. Vibrance
// ---------------------------------------------------------------------------

struct VibranceUniforms {
    float vibrance;  // 0 = identity; >0 boosts muted colours, <0 mutes
};

kernel void vibranceKernel(
    texture2d<float, access::read>  src [[texture(0)]],
    texture2d<float, access::write> dst [[texture(1)]],
    constant VibranceUniforms&      u   [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    const float4 c = src.read(gid);
    float3 rgb = c.rgb;

    // The classic vibrance trick: the boost is proportional to how *unsaturated*
    // the pixel already is (mx - average), so vivid pixels are left alone.
    const float average = (rgb.r + rgb.g + rgb.b) / 3.0f;
    const float mx      = max3(rgb.r, rgb.g, rgb.b);
    const float amt     = (mx - average) * (-u.vibrance * 3.0f);
    rgb = mix(rgb, float3(mx), amt);

    dst.write(float4(caFinish(rgb), c.a), gid);
}

// ---------------------------------------------------------------------------
// 5. White balance
// ---------------------------------------------------------------------------

struct WhiteBalanceUniforms {
    float temperature;  // -1 cool … +1 warm; 0 = identity
    float tint;         // -1 green … +1 magenta; 0 = identity
};

kernel void whiteBalanceKernel(
    texture2d<float, access::read>  src [[texture(0)]],
    texture2d<float, access::write> dst [[texture(1)]],
    constant WhiteBalanceUniforms&  u   [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    const float4 c = src.read(gid);
    float3 rgb = c.rgb;

    // Temperature: push R against B (Planckian-ish slope, capped for sanity).
    rgb.r *= 1.0f + u.temperature * 0.25f;
    rgb.b *= 1.0f - u.temperature * 0.25f;

    // Tint: push G against magenta (R+B together).
    rgb.g *= 1.0f - u.tint * 0.25f;
    rgb.r *= 1.0f + u.tint * 0.10f;
    rgb.b *= 1.0f + u.tint * 0.10f;

    dst.write(float4(caFinish(rgb), c.a), gid);
}

// ---------------------------------------------------------------------------
// 6. Tone curve (256×1 per-channel LUT, baked on the CPU)
// ---------------------------------------------------------------------------

constexpr sampler curveSampler(coord::normalized, filter::linear, address::clamp_to_edge);

kernel void toneCurveKernel(
    texture2d<float, access::read>   src   [[texture(0)]],
    texture2d<float, access::write>  dst   [[texture(1)]],
    texture2d<float, access::sample> curve [[texture(2)]],   // 256×1, rgba8
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    const float4 c = src.read(gid);

    // HDR-safe: split into base [0,1] (curved) + above-white residual (kept).
    const float3 base      = saturate(c.rgb);
    const float3 highlight = isHDR ? max(c.rgb - 1.0f, 0.0f) : float3(0.0f);

    // Texel-centre mapping for a 256-wide LUT: v → (v*255 + 0.5)/256.
    float3 mapped;
    mapped.r = curve.sample(curveSampler, float2((base.r * 255.0f + 0.5f) / 256.0f, 0.5f)).r;
    mapped.g = curve.sample(curveSampler, float2((base.g * 255.0f + 0.5f) / 256.0f, 0.5f)).g;
    mapped.b = curve.sample(curveSampler, float2((base.b * 255.0f + 0.5f) / 256.0f, 0.5f)).b;

    dst.write(float4(caFinish(mapped + highlight), c.a), gid);
}

// ---------------------------------------------------------------------------
// 7. Highlight / shadow recovery
// ---------------------------------------------------------------------------

struct HighlightShadowUniforms {
    float shadows;     // 0 = identity … 1 = full shadow lift
    float highlights;  // 1 = identity … 0 = full highlight recovery
};

kernel void highlightShadowKernel(
    texture2d<float, access::read>    src [[texture(0)]],
    texture2d<float, access::write>   dst [[texture(1)]],
    constant HighlightShadowUniforms& u   [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    const float4 c = src.read(gid);
    const float3 rgb  = max(c.rgb, 0.0f);
    const float  luma = dot(saturate(rgb), caLuma3());

    // GPUImage-style banded gains: shadows act below ~⅓ luminance,
    // highlight recovery acts above ~⅔.
    const float shadow = clamp(
        (pow(luma, 1.0f / (u.shadows + 1.0f)) +
         (-0.76f) * pow(luma, 2.0f / (u.shadows + 1.0f))) - luma,
        0.0f, 1.0f);
    const float highlight = clamp(
        (1.0f - (pow(1.0f - luma, 1.0f / (2.0f - u.highlights)) +
                 (-0.8f) * pow(1.0f - luma, 2.0f / (2.0f - u.highlights)))) - luma,
        -1.0f, 0.0f);

    // Re-scale the pixel so its luminance moves by (shadow + highlight).
    const float newLuma = luma + shadow + highlight;
    float3 result = rgb * (newLuma / max(luma, 1e-4f));

    dst.write(float4(caFinish(result), c.a), gid);
}

// ---------------------------------------------------------------------------
// 8. Highlight / shadow tint
// ---------------------------------------------------------------------------

struct HighlightShadowTintUniforms {
    float3 shadowTintColor;    float shadowTintIntensity;
    float3 highlightTintColor; float highlightTintIntensity;
};

kernel void highlightShadowTintKernel(
    texture2d<float, access::read>        src [[texture(0)]],
    texture2d<float, access::write>       dst [[texture(1)]],
    constant HighlightShadowTintUniforms& u   [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    const float4 c = src.read(gid);
    const float3 rgb  = c.rgb;
    const float  luma = dot(saturate(rgb), caLuma3());

    // Shadow tint is strongest at black, highlight tint strongest at white.
    const float3 shadowResult = mix(
        rgb, max(rgb, u.shadowTintColor * luma),
        u.shadowTintIntensity * (1.0f - luma));
    const float3 result = mix(
        shadowResult, max(shadowResult, u.highlightTintColor * luma),
        u.highlightTintIntensity * luma);

    dst.write(float4(caFinish(result), c.a), gid);
}

// ---------------------------------------------------------------------------
// 9. Colour matrix
// ---------------------------------------------------------------------------

struct ColorMatrixUniforms {
    float4x4 matrix;     // applied to (r, g, b, a)
    float    intensity;  // 0 = identity … 1 = full matrix
};

kernel void colorMatrixKernel(
    texture2d<float, access::read>  src [[texture(0)]],
    texture2d<float, access::write> dst [[texture(1)]],
    constant ColorMatrixUniforms&   u   [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    const float4 c = src.read(gid);
    const float4 transformed = u.matrix * c;
    const float4 blended = mix(c, transformed, u.intensity);
    dst.write(float4(caFinish(blended.rgb), saturate(blended.a)), gid);
}

// ---------------------------------------------------------------------------
// 10. Colour invert
// ---------------------------------------------------------------------------

kernel void colorInvertKernel(
    texture2d<float, access::read>  src [[texture(0)]],
    texture2d<float, access::write> dst [[texture(1)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    const float4 c = src.read(gid);
    // Inversion is only defined on [0,1]; HDR highlights are clamped first.
    const float3 rgb = 1.0f - saturate(c.rgb);
    dst.write(float4(rgb, c.a), gid);
}

// ---------------------------------------------------------------------------
// 11. Monochrome
// ---------------------------------------------------------------------------

struct MonochromeUniforms {
    float3 filterColor;  // tint applied to the luma image
    float  intensity;    // 0 = identity … 1 = fully monochrome
};

kernel void monochromeKernel(
    texture2d<float, access::read>  src [[texture(0)]],
    texture2d<float, access::write> dst [[texture(1)]],
    constant MonochromeUniforms&    u   [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    const float4 c = src.read(gid);
    const float luma = dot(saturate(c.rgb), caLuma3());

    // GPUImage's monochrome curve: darker pixels lean toward the filter
    // colour's shadow response, brighter toward its highlight response.
    const float3 mono = float3(
        luma < 0.5f ? (2.0f * luma * u.filterColor.r) : (1.0f - 2.0f * (1.0f - luma) * (1.0f - u.filterColor.r)),
        luma < 0.5f ? (2.0f * luma * u.filterColor.g) : (1.0f - 2.0f * (1.0f - luma) * (1.0f - u.filterColor.g)),
        luma < 0.5f ? (2.0f * luma * u.filterColor.b) : (1.0f - 2.0f * (1.0f - luma) * (1.0f - u.filterColor.b)));

    const float3 rgb = mix(c.rgb, mono, u.intensity);
    dst.write(float4(caFinish(rgb), c.a), gid);
}

// ---------------------------------------------------------------------------
// 12. False colour
// ---------------------------------------------------------------------------

struct FalseColorUniforms {
    float3 firstColor;  float pad0;   // mapped to luma 0
    float3 secondColor; float pad1;   // mapped to luma 1
};

kernel void falseColorKernel(
    texture2d<float, access::read>  src [[texture(0)]],
    texture2d<float, access::write> dst [[texture(1)]],
    constant FalseColorUniforms&    u   [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    const float4 c = src.read(gid);
    const float luma = dot(saturate(c.rgb), caLuma3());
    const float3 rgb = mix(u.firstColor, u.secondColor, luma);
    dst.write(float4(saturate(rgb), c.a), gid);
}
