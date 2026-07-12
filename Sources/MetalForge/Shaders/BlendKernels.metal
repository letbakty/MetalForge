#include <metal_stdlib>
using namespace metal;

// ===========================================================================
// BlendKernels — v0.4.0 Blend Modes API.
//
//   blendModeKernel   one kernel, 23 blend modes selected by uniform (the
//                     mode set is fixed, so a runtime switch beats 23 PSOs)
//   chromaKeyKernel   green-screen keying against a configurable colour
//   maskBlendKernel   per-pixel mask chooses between source and overlay
//
// Terminology: `base` is the incoming pipeline frame, `blend` is the overlay
// texture. All blend math happens on values clamped to [0,1] (blend modes are
// display-referred by definition); the HDR residual above 1.0 is preserved
// and added back, matching the LUT filter's strategy.
// ===========================================================================

constant bool isHDR [[function_constant(0)]];

constexpr sampler overlaySampler(coord::normalized, filter::linear, address::clamp_to_edge);

// ---------------------------------------------------------------------------
// Separable blend-mode primitives (per channel)
// ---------------------------------------------------------------------------

static inline float blendChannel(float b, float s, int mode) {
    switch (mode) {
        case 1:  return b * s;                                           // multiply
        case 2:  return 1.0f - (1.0f - b) * (1.0f - s);                  // screen
        case 3:  return b < 0.5f ? 2.0f * b * s                          // overlay
                                 : 1.0f - 2.0f * (1.0f - b) * (1.0f - s);
        case 4:  return s < 0.5f ? 2.0f * s * b                          // hard light
                                 : 1.0f - 2.0f * (1.0f - s) * (1.0f - b);
        case 5: {                                                        // soft light (W3C)
            if (s <= 0.5f) return b - (1.0f - 2.0f * s) * b * (1.0f - b);
            const float d = (b <= 0.25f) ? ((16.0f * b - 12.0f) * b + 4.0f) * b : sqrt(b);
            return b + (2.0f * s - 1.0f) * (d - b);
        }
        case 6:  return min(b, s);                                       // darken
        case 7:  return max(b, s);                                       // lighten
        case 8:  return s >= 1.0f ? 1.0f : min(b / (1.0f - s), 1.0f);    // color dodge
        case 9:  return s <= 0.0f ? 0.0f : 1.0f - min((1.0f - b) / s, 1.0f); // color burn
        case 10: return max(b + s - 1.0f, 0.0f);                         // linear burn
        case 11: return abs(b - s);                                      // difference
        case 12: return b + s - 2.0f * b * s;                            // exclusion
        case 13: return max(b - s, 0.0f);                                // subtract
        case 14: return s <= 0.0f ? 1.0f : min(b / s, 1.0f);             // divide
        case 19: {                                                       // vivid light
            if (s < 0.5f) {
                const float s2 = 2.0f * s;
                return s2 <= 0.0f ? 0.0f : 1.0f - min((1.0f - b) / s2, 1.0f);
            }
            const float s2 = 2.0f * (s - 0.5f);
            return s2 >= 1.0f ? 1.0f : min(b / (1.0f - s2), 1.0f);
        }
        case 20: return s < 0.5f ? min(b, 2.0f * s)                      // pin light
                                 : max(b, 2.0f * s - 1.0f);
        case 21: return (b + s) >= 1.0f ? 1.0f : 0.0f;                   // hard mix
        default: return s;                                               // normal
    }
}

// ---------------------------------------------------------------------------
// Non-separable (HSL) modes — Photoshop SetLum / SetSat machinery
// ---------------------------------------------------------------------------

static inline float psLum(float3 c) {
    return dot(c, float3(0.3f, 0.59f, 0.11f));
}

static inline float3 psClipColor(float3 c) {
    const float l = psLum(c);
    const float n = min3(c.r, c.g, c.b);
    const float x = max3(c.r, c.g, c.b);
    if (n < 0.0f) c = l + (c - l) * l / max(l - n, 1e-5f);
    if (x > 1.0f) c = l + (c - l) * (1.0f - l) / max(x - l, 1e-5f);
    return c;
}

static inline float3 psSetLum(float3 c, float l) {
    return psClipColor(c + (l - psLum(c)));
}

static inline float psSat(float3 c) {
    return max3(c.r, c.g, c.b) - min3(c.r, c.g, c.b);
}

static inline float3 psSetSat(float3 c, float s) {
    const float mn = min3(c.r, c.g, c.b);
    const float mx = max3(c.r, c.g, c.b);
    float3 result = float3(0.0f);
    if (mx > mn) {
        result = (c - mn) * s / (mx - mn);
    }
    return result;
}

static inline float3 blendNonSeparable(float3 b, float3 s, int mode) {
    switch (mode) {
        case 15: return psSetLum(psSetSat(s, psSat(b)), psLum(b));  // hue
        case 16: return psSetLum(psSetSat(b, psSat(s)), psLum(b));  // saturation
        case 17: return psSetLum(s, psLum(b));                      // color
        case 18: return psSetLum(b, psLum(s));                      // luminosity
        default: return s;
    }
}

/// Cheap per-pixel hash for the dissolve dither.
static inline float hash21(float2 p) {
    return fract(sin(dot(p, float2(12.9898f, 78.233f))) * 43758.5453f);
}

// ---------------------------------------------------------------------------
// blendModeKernel
// ---------------------------------------------------------------------------

struct BlendModeUniforms {
    int   mode;        // see BlendMode raw values in Swift
    float intensity;   // 0 = base only … 1 = full blend result
    float seed;        // dissolve dither offset
};

kernel void blendModeKernel(
    texture2d<float, access::read>   src     [[texture(0)]],
    texture2d<float, access::write>  dst     [[texture(1)]],
    texture2d<float, access::sample> overlay [[texture(2)]],
    constant BlendModeUniforms&      u       [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    const float2 size = float2(dst.get_width(), dst.get_height());
    const float2 uv = (float2(gid) + 0.5f) / size;

    const float4 baseC  = src.read(gid);
    const float4 blendC = overlay.sample(overlaySampler, uv);

    // Display-referred split: blend modes operate on [0,1]; HDR residual
    // rides through unchanged.
    const float3 b = saturate(baseC.rgb);
    const float3 s = saturate(blendC.rgb);
    const float3 highlight = isHDR ? max(baseC.rgb - 1.0f, 0.0f) : float3(0.0f);

    float3 blended;
    if (u.mode >= 15 && u.mode <= 18) {
        blended = blendNonSeparable(b, s, u.mode);
    } else if (u.mode == 22) {
        // Dissolve: intensity is the dither coverage probability.
        const float r = hash21(float2(gid) + u.seed);
        blended = (r < u.intensity) ? s : b;
    } else {
        blended = float3(
            blendChannel(b.r, s.r, u.mode),
            blendChannel(b.g, s.g, u.mode),
            blendChannel(b.b, s.b, u.mode));
    }

    // Honour the overlay's own alpha, then the global intensity.
    // (Dissolve already consumed intensity as coverage.)
    float3 composed = mix(b, blended, blendC.a);
    if (u.mode != 22) {
        composed = mix(b, composed, u.intensity);
    }

    float3 result = composed + highlight;
    if (!isHDR) result = saturate(result);
    dst.write(float4(result, baseC.a), gid);
}

// ---------------------------------------------------------------------------
// chromaKeyKernel
// ---------------------------------------------------------------------------

struct ChromaKeyUniforms {
    float3 keyColor;          float thresholdSensitivity;
    float  smoothing;         float hasBackground;
    float  pad0, pad1;
};

kernel void chromaKeyKernel(
    texture2d<float, access::read>   src        [[texture(0)]],
    texture2d<float, access::write>  dst        [[texture(1)]],
    texture2d<float, access::sample> background [[texture(2)]],
    constant ChromaKeyUniforms&      u          [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    const float2 size = float2(dst.get_width(), dst.get_height());
    const float2 uv = (float2(gid) + 0.5f) / size;

    const float4 c = src.read(gid);
    const float3 rgb = saturate(c.rgb);

    // Compare in Cb/Cr space (GPUImage's approach): chroma distance is far
    // more robust to lighting variation than RGB distance.
    const float maskY  = 0.2989f * u.keyColor.r + 0.5866f * u.keyColor.g + 0.1145f * u.keyColor.b;
    const float maskCr = 0.7132f * (u.keyColor.r - maskY);
    const float maskCb = 0.5647f * (u.keyColor.b - maskY);

    const float y  = 0.2989f * rgb.r + 0.5866f * rgb.g + 0.1145f * rgb.b;
    const float cr = 0.7132f * (rgb.r - y);
    const float cb = 0.5647f * (rgb.b - y);

    const float dist = distance(float2(cr, cb), float2(maskCr, maskCb));
    // alpha 0 = keyed out, 1 = kept.
    const float alpha = smoothstep(u.thresholdSensitivity,
                                   u.thresholdSensitivity + u.smoothing, dist);

    float4 result;
    if (u.hasBackground > 0.5f) {
        const float3 bg = background.sample(overlaySampler, uv).rgb;
        result = float4(mix(bg, c.rgb, alpha), c.a);
    } else {
        // No background: write premultiplied-style keyed output with alpha.
        result = float4(c.rgb * alpha, alpha);
    }

    if (!isHDR) result = saturate(result);
    else        result = max(result, 0.0f);
    dst.write(result, gid);
}

// ---------------------------------------------------------------------------
// maskBlendKernel
// ---------------------------------------------------------------------------

kernel void maskBlendKernel(
    texture2d<float, access::read>   src     [[texture(0)]],
    texture2d<float, access::write>  dst     [[texture(1)]],
    texture2d<float, access::sample> overlay [[texture(2)]],
    texture2d<float, access::sample> mask    [[texture(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    const float2 size = float2(dst.get_width(), dst.get_height());
    const float2 uv = (float2(gid) + 0.5f) / size;

    const float4 baseC  = src.read(gid);
    const float4 overC  = overlay.sample(overlaySampler, uv);
    // Mask weight = luminance of the mask texture (white shows the overlay).
    const float w = dot(saturate(mask.sample(overlaySampler, uv).rgb),
                        float3(0.2126f, 0.7152f, 0.0722f));

    float4 result = float4(mix(baseC.rgb, overC.rgb, w), baseC.a);
    if (!isHDR) result = saturate(result);
    else        result = max(result, 0.0f);
    dst.write(result, gid);
}
