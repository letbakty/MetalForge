#include <metal_stdlib>
using namespace metal;

// ===========================================================================
// StylizationKernels — vignette + scanlines + RGB split.
//
// Same conventions as the rest of MetalForge: an `isHDR` function constant
// gates the final clamp, non-uniform dispatch requires the bounds guard, and
// sampling kernels use a bilinear clamp-to-edge sampler.
// ===========================================================================

constant bool isHDR [[function_constant(0)]];

constexpr sampler stylizeSampler(
    coord::normalized,
    filter::linear,
    address::clamp_to_edge
);

// ===========================================================================
// 1. Vignette — radial darkening towards the frame edges.
//
// Distance is measured from the centre in aspect-corrected UV space (so the
// falloff is circular, not elliptical). `radius` is where darkening begins and
// `softness` is the width of the falloff band; `intensity` is how dark the
// corners get (1 = fully black at the outer edge of the band).
// ===========================================================================
struct VignetteUniforms {
    float intensity;   // 0 = bypass, 1 = full darkening
    float radius;      // normalised distance where falloff starts
    float softness;    // width of the falloff band
};

kernel void vignetteKernel(
    texture2d<float, access::read>  source [[texture(0)]],
    texture2d<float, access::write> dest   [[texture(1)]],
    constant VignetteUniforms&      u      [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint W = dest.get_width();
    const uint H = dest.get_height();
    if (gid.x >= W || gid.y >= H) return;

    const float2 uv = (float2(gid) + 0.5f) / float2(float(W), float(H));

    // Aspect-correct the X axis so the vignette is a circle on non-square frames.
    float2 d = uv - 0.5f;
    d.x *= float(W) / float(H);
    const float dist = length(d);

    // 0 inside `radius`, ramping to 1 across the softness band.
    const float falloff = smoothstep(u.radius, u.radius + u.softness, dist);
    const float factor  = 1.0f - u.intensity * falloff;

    float4 color = source.read(gid);
    color.rgb *= factor;

    if (!isHDR) {
        color = clamp(color, 0.0f, 1.0f);
    }
    dest.write(color, gid);
}

// ===========================================================================
// 2. Scanlines — horizontal CRT-style darkening bands.
//
// A squared sine over the row index produces smooth alternating dark/light
// bands. `lineWidth` is the band period in pixels; `timeSeed` scrolls the bands
// vertically (drive it from a frame clock for a rolling-CRT look); `intensity`
// scales how dark the troughs get.
// ===========================================================================
struct ScanlineUniforms {
    float intensity;   // 0 = bypass, 1 = strong scanlines
    float lineWidth;   // band period in pixels (>= 1)
    float timeSeed;    // vertical scroll offset
};

kernel void scanlineKernel(
    texture2d<float, access::read>  source [[texture(0)]],
    texture2d<float, access::write> dest   [[texture(1)]],
    constant ScanlineUniforms&      u      [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint W = dest.get_width();
    const uint H = dest.get_height();
    if (gid.x >= W || gid.y >= H) return;

    // sin² gives a smooth 0..1 band; lineWidth is the period in rows.
    const float phase = M_PI_F * (float(gid.y) + u.timeSeed) / max(u.lineWidth, 1.0f);
    const float s     = sin(phase);
    const float mask  = s * s;                 // 0..1
    const float factor = 1.0f - u.intensity * mask;

    float4 color = source.read(gid);
    color.rgb *= factor;

    if (!isHDR) {
        color = clamp(color, 0.0f, 1.0f);
    }
    dest.write(color, gid);
}

// ===========================================================================
// 3. RGB Split — independent per-channel UV displacement.
//
// Samples R, G and B at three independently-offset coordinates, scaled by a
// global `intensity` so the effect can be dialled 0→full. Alpha is taken from
// the un-shifted centre tap. This generalises chromatic aberration to arbitrary
// per-channel directions (e.g. diagonal glitch splits).
// ===========================================================================
struct RGBSplitUniforms {
    float2 redOffset;     // normalised UV offset for the R channel
    float2 greenOffset;   // normalised UV offset for the G channel
    float2 blueOffset;    // normalised UV offset for the B channel
    float  intensity;     // global multiplier on all three offsets
};

kernel void rgbSplitKernel(
    texture2d<float, access::sample> source [[texture(0)]],
    texture2d<float, access::write>  dest   [[texture(1)]],
    constant RGBSplitUniforms&       u      [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint W = dest.get_width();
    const uint H = dest.get_height();
    if (gid.x >= W || gid.y >= H) return;

    const float2 uv = (float2(gid) + 0.5f) / float2(float(W), float(H));
    const float  t  = clamp(u.intensity, 0.0f, 1.0f);

    const float r = source.sample(stylizeSampler, uv + u.redOffset   * t).r;
    const float g = source.sample(stylizeSampler, uv + u.greenOffset * t).g;
    const float b = source.sample(stylizeSampler, uv + u.blueOffset  * t).b;
    const float a = source.sample(stylizeSampler, uv).a;

    float4 color = float4(r, g, b, a);
    if (!isHDR) {
        color = clamp(color, 0.0f, 1.0f);
    }
    dest.write(color, gid);
}
