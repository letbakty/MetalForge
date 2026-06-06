#include <metal_stdlib>
using namespace metal;

// ===========================================================================
// BlurKernels — separable Gaussian blur (two passes) + single-pass sharpen.
//
// Shared design (same conventions as the rest of MetalForge):
//   • `isHDR` function constant gates the final [0,1] clamp. Two specialised
//     PSO variants are emitted per kernel that reference it; the dead branch is
//     eliminated, so there is zero per-thread cost.
//   • Non-uniform `dispatchThreads`, so the in-shader bounds guard is mandatory.
//   • Sampling kernels use a bilinear clamp-to-edge sampler so out-of-bounds
//     taps read the edge texel rather than wrapping or returning black.
//
// The Gaussian blur is separable: a horizontal pass writes an intermediate,
// then a vertical pass reads that intermediate (and the original source, for the
// intensity blend) and writes the destination. This is O(2R) taps per pixel
// instead of O(R²) for a naive 2D kernel.
// ===========================================================================

constant bool isHDR [[function_constant(0)]];

// Upper bound on the blur radius. Caps the dynamic loop length so a runaway
// `radius` can never stall the GPU. Mirrored by the Swift-side clamp.
constant int kMaxBlurRadius = 64;

constexpr sampler blurSampler(
    coord::normalized,
    filter::linear,
    address::clamp_to_edge
);

// Must mirror `GaussianBlurUniforms` in BlurFilters.swift.
struct GaussianBlurUniforms {
    float radius;      // Gaussian radius in pixels (0 = no blur)
    float intensity;   // 0 = original, 1 = fully blurred (vertical pass only)
};

// ---------------------------------------------------------------------------
// Pass 1 — horizontal Gaussian. Reads the source, writes the intermediate.
// No clamp: the intermediate shares the source's pixel format, and an
// .rgba16Float intermediate must keep highlights above 1.0 intact. (A
// .bgra8Unorm intermediate auto-clamps on write.)
// ---------------------------------------------------------------------------
kernel void gaussianBlurHorizontalKernel(
    texture2d<float, access::sample> source [[texture(0)]],
    texture2d<float, access::write>  dest   [[texture(1)]],
    constant GaussianBlurUniforms&   u      [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint W = dest.get_width();
    const uint H = dest.get_height();
    if (gid.x >= W || gid.y >= H) return;

    const float2 uv     = (float2(gid) + 0.5f) / float2(float(W), float(H));
    const float  texelX = 1.0f / float(W);

    const int   R     = clamp(int(ceil(u.radius)), 0, kMaxBlurRadius);
    const float sigma = max(u.radius * 0.5f, 1e-4f);
    const float denom = 2.0f * sigma * sigma;

    float4 acc  = float4(0.0f);
    float  wsum = 0.0f;
    for (int i = -R; i <= R; ++i) {
        const float w  = exp(-float(i * i) / denom);
        const float2 s = float2(uv.x + float(i) * texelX, uv.y);
        acc  += source.sample(blurSampler, s) * w;
        wsum += w;
    }

    dest.write(acc / max(wsum, 1e-4f), gid);
}

// ---------------------------------------------------------------------------
// Pass 2 — vertical Gaussian + intensity blend. Reads the horizontally-blurred
// intermediate (texture 0) and the ORIGINAL source (texture 1), then writes
// dest = mix(original, fullyBlurred, intensity).
// ---------------------------------------------------------------------------
kernel void gaussianBlurVerticalKernel(
    texture2d<float, access::sample> blurredH [[texture(0)]],
    texture2d<float, access::sample> original [[texture(1)]],
    texture2d<float, access::write>  dest     [[texture(2)]],
    constant GaussianBlurUniforms&   u        [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint W = dest.get_width();
    const uint H = dest.get_height();
    if (gid.x >= W || gid.y >= H) return;

    const float2 uv     = (float2(gid) + 0.5f) / float2(float(W), float(H));
    const float  texelY = 1.0f / float(H);

    const int   R     = clamp(int(ceil(u.radius)), 0, kMaxBlurRadius);
    const float sigma = max(u.radius * 0.5f, 1e-4f);
    const float denom = 2.0f * sigma * sigma;

    float4 acc  = float4(0.0f);
    float  wsum = 0.0f;
    for (int i = -R; i <= R; ++i) {
        const float w  = exp(-float(i * i) / denom);
        const float2 s = float2(uv.x, uv.y + float(i) * texelY);
        acc  += blurredH.sample(blurSampler, s) * w;
        wsum += w;
    }

    const float4 blurred = acc / max(wsum, 1e-4f);
    const float4 orig    = original.sample(blurSampler, uv);
    float4 color = mix(orig, blurred, clamp(u.intensity, 0.0f, 1.0f));

    if (!isHDR) {
        color = clamp(color, 0.0f, 1.0f);
    }
    dest.write(color, gid);
}

// ===========================================================================
// Sharpen — single-pass unsharp mask via a 4-neighbour Laplacian.
//
//   out = center·(1 + 4·amount) − amount·(up + down + left + right)
//
// At amount = 0 this is the identity. Larger amounts boost local contrast at
// edges. Alpha is passed through from the centre tap unchanged.
// ===========================================================================
struct SharpenUniforms {
    float amount;   // 0 = identity; sensible range up to ~2
};

kernel void sharpenKernel(
    texture2d<float, access::sample> source [[texture(0)]],
    texture2d<float, access::write>  dest   [[texture(1)]],
    constant SharpenUniforms&        u      [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint W = dest.get_width();
    const uint H = dest.get_height();
    if (gid.x >= W || gid.y >= H) return;

    const float2 size = float2(float(W), float(H));
    const float2 uv   = (float2(gid) + 0.5f) / size;
    const float2 tx   = float2(1.0f, 0.0f) / size;
    const float2 ty   = float2(0.0f, 1.0f) / size;

    const float4 c = source.sample(blurSampler, uv);
    const float3 up    = source.sample(blurSampler, uv + ty).rgb;
    const float3 down  = source.sample(blurSampler, uv - ty).rgb;
    const float3 left  = source.sample(blurSampler, uv - tx).rgb;
    const float3 right = source.sample(blurSampler, uv + tx).rgb;

    const float a = u.amount;   // already clamped to >= 0 on the Swift side
    float3 rgb = c.rgb * (1.0f + 4.0f * a) - a * (up + down + left + right);

    float4 color = float4(rgb, c.a);
    if (!isHDR) {
        color = clamp(color, 0.0f, 1.0f);
    }
    dest.write(color, gid);
}
