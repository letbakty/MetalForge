#include <metal_stdlib>
using namespace metal;

// ===========================================================================
// TemporalPackKernels — v0.8.0 Temporal Video Effects Pack.
//
//   frameBlendKernel          decaying max-blend light trails
//   lowPassKernel             exponential moving average of the stream
//   highPassKernel            |current − EMA| with EMA side-output
//   motionDetectorKernel      thresholded frame difference visualisation
//   opticalFlowWarpKernel     gradient-based flow estimate + UV warp
//   frameInterpolationKernel  phase crossfade between consecutive frames
//
// All kernels read the current frame at texture(0) and the filter-owned
// history at texture(1); the history update policy (what gets blitted back)
// lives in the Swift wrappers.
// ===========================================================================

constant bool isHDR [[function_constant(0)]];

constant float3 kLumaW = float3(0.2126f, 0.7152f, 0.0722f);

static inline float4 temporalFinish(float4 c) {
    return isHDR ? max(c, 0.0f) : saturate(c);
}

static inline float4 readClampedT(texture2d<float, access::read> t, int x, int y) {
    const int w = int(t.get_width());
    const int h = int(t.get_height());
    return t.read(uint2(clamp(x, 0, w - 1), clamp(y, 0, h - 1)));
}

// ---------------------------------------------------------------------------
// 1. Frame blend (light trails)
// ---------------------------------------------------------------------------

struct FrameBlendUniforms {
    float decay;   // per-frame trail multiplier (0.8…0.98)
};

kernel void frameBlendKernel(
    texture2d<float, access::read>  src  [[texture(0)]],
    texture2d<float, access::read>  prev [[texture(1)]],
    texture2d<float, access::write> dst  [[texture(2)]],
    constant FrameBlendUniforms&    u    [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    const float4 c = src.read(gid);
    const float4 trail = prev.read(gid) * u.decay;
    // max() keeps bright streaks alive while dark content refreshes instantly
    // — the classic long-exposure light-trail response.
    dst.write(temporalFinish(max(c, trail)), gid);
}

// ---------------------------------------------------------------------------
// 2. Low pass (EMA)
// ---------------------------------------------------------------------------

struct LowPassUniforms {
    float strength;   // 0 = pass-through, →1 = heavy temporal smoothing
};

kernel void lowPassKernel(
    texture2d<float, access::read>  src  [[texture(0)]],
    texture2d<float, access::read>  prev [[texture(1)]],
    texture2d<float, access::write> dst  [[texture(2)]],
    constant LowPassUniforms&       u    [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    const float4 c = src.read(gid);
    const float4 ema = mix(c, prev.read(gid), u.strength);
    dst.write(temporalFinish(ema), gid);
}

// ---------------------------------------------------------------------------
// 3. High pass (current − EMA)
// ---------------------------------------------------------------------------

struct HighPassUniforms {
    float strength;   // EMA persistence, as in lowPassKernel
};

kernel void highPassKernel(
    texture2d<float, access::read>  src   [[texture(0)]],
    texture2d<float, access::read>  prev  [[texture(1)]],   // previous EMA
    texture2d<float, access::write> dst   [[texture(2)]],   // |src − EMA|
    texture2d<float, access::write> emaOut [[texture(3)]],  // new EMA (saved by wrapper)
    constant HighPassUniforms&      u     [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    const float4 c = src.read(gid);
    const float4 ema = mix(c, prev.read(gid), u.strength);
    emaOut.write(temporalFinish(ema), gid);
    // The residual: only what *changed* against the slow average survives.
    dst.write(float4(abs(c.rgb - ema.rgb), c.a), gid);
}

// ---------------------------------------------------------------------------
// 4. Motion detector
// ---------------------------------------------------------------------------

struct MotionDetectorUniforms {
    float3 highlightColor;   float threshold;
    float  dimming;          float pad0, pad1, pad2;
};

kernel void motionDetectorKernel(
    texture2d<float, access::read>  src  [[texture(0)]],
    texture2d<float, access::read>  prev [[texture(1)]],
    texture2d<float, access::write> dst  [[texture(2)]],
    constant MotionDetectorUniforms& u   [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    const float4 c = src.read(gid);
    const float3 d = abs(saturate(c.rgb) - saturate(prev.read(gid).rgb));
    const float motion = dot(d, kLumaW) * 3.0f;   // amplified luma difference

    // Moving pixels get painted with the highlight colour over a dimmed scene.
    const float m = smoothstep(u.threshold, u.threshold + 0.1f, motion);
    const float3 rgb = mix(saturate(c.rgb) * u.dimming, u.highlightColor, m);
    dst.write(temporalFinish(float4(rgb, c.a)), gid);
}

// ---------------------------------------------------------------------------
// 5. Optical-flow warp
// ---------------------------------------------------------------------------

struct FlowWarpUniforms {
    float strength;    // warp distance multiplier in pixels
    float smoothing;   // flow magnitude clamp (px)
};

kernel void opticalFlowWarpKernel(
    texture2d<float, access::read>  src  [[texture(0)]],
    texture2d<float, access::read>  prev [[texture(1)]],
    texture2d<float, access::write> dst  [[texture(2)]],
    constant FlowWarpUniforms&      u    [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    const int x = int(gid.x);
    const int y = int(gid.y);

    // Single-iteration Lucas–Kanade-style estimate: spatial gradient from the
    // current frame, temporal gradient against the previous frame.
    const float ix = (dot(readClampedT(src, x + 1, y).rgb, kLumaW)
                    - dot(readClampedT(src, x - 1, y).rgb, kLumaW)) * 0.5f;
    const float iy = (dot(readClampedT(src, x, y + 1).rgb, kLumaW)
                    - dot(readClampedT(src, x, y - 1).rgb, kLumaW)) * 0.5f;
    const float it = dot(src.read(gid).rgb, kLumaW)
                   - dot(prev.read(gid).rgb, kLumaW);

    const float g2 = ix * ix + iy * iy;
    float2 flow = (g2 > 1e-5f) ? (-it / (g2 + 1e-3f)) * float2(ix, iy) : float2(0.0f);
    flow = clamp(flow, -u.smoothing, u.smoothing);

    // Warp the current frame backwards along the estimated motion — moving
    // regions liquefy, static regions stay put.
    const float2 p = float2(gid) + flow * u.strength;
    dst.write(temporalFinish(readClampedT(src, int(round(p.x)), int(round(p.y)))), gid);
}

// ---------------------------------------------------------------------------
// 6. Frame interpolation (phase crossfade)
// ---------------------------------------------------------------------------

struct FrameInterpolationUniforms {
    float phase;   // 0 = previous frame … 1 = current frame
};

kernel void frameInterpolationKernel(
    texture2d<float, access::read>  src  [[texture(0)]],
    texture2d<float, access::read>  prev [[texture(1)]],
    texture2d<float, access::write> dst  [[texture(2)]],
    constant FrameInterpolationUniforms& u [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    const float4 c = mix(prev.read(gid), src.read(gid), saturate(u.phase));
    dst.write(temporalFinish(c), gid);
}
