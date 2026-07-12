#include <metal_stdlib>
using namespace metal;

// ===========================================================================
// EdgeKernels — v0.5.0 Edge Detection & Convolution Pack.
//
//   sobelKernel              Sobel gradient magnitude on luminance
//   prewittKernel            Prewitt gradient magnitude on luminance
//   convolution3x3Kernel     arbitrary 3×3 kernel + bias (also drives Emboss)
//   laplacianKernel          8-neighbour Laplacian, mid-grey biased
//   cannyGradientKernel      pass 1 of Canny: blurred Sobel magnitude + direction
//   cannySuppressKernel      pass 2 of Canny: NMS along gradient + double threshold
//   harrisCornerKernel       Harris corner response (windowed structure tensor)
//   nonMaximumSuppressionKernel  keep only local-maximum luminance pixels
//
// Edge detectors output grayscale maps and are display-referred: inputs are
// clamped to [0,1]; the isHDR constant only controls the final clamp.
// ===========================================================================

constant bool isHDR [[function_constant(0)]];

constant float3 kLumaW = float3(0.2126f, 0.7152f, 0.0722f);

/// Clamped-coordinate luminance read.
static inline float lumaAt(texture2d<float, access::read> t, int x, int y) {
    const int w = int(t.get_width());
    const int h = int(t.get_height());
    const uint2 p = uint2(clamp(x, 0, w - 1), clamp(y, 0, h - 1));
    return dot(saturate(t.read(p).rgb), kLumaW);
}

static inline float4 edgeFinish(float3 rgb, float a) {
    return isHDR ? float4(max(rgb, 0.0f), a) : float4(saturate(rgb), a);
}

// ---------------------------------------------------------------------------
// 1/2. Sobel & Prewitt
// ---------------------------------------------------------------------------

struct EdgeStrengthUniforms {
    float edgeStrength;   // output gain
};

/// Shared 3×3 gradient: `cWeight` is the centre-row/column weight
/// (2 for Sobel, 1 for Prewitt).
static inline float gradientMagnitude(
    texture2d<float, access::read> src, int2 p, float cWeight)
{
    const float tl = lumaAt(src, p.x - 1, p.y - 1);
    const float tc = lumaAt(src, p.x,     p.y - 1);
    const float tr = lumaAt(src, p.x + 1, p.y - 1);
    const float ml = lumaAt(src, p.x - 1, p.y);
    const float mr = lumaAt(src, p.x + 1, p.y);
    const float bl = lumaAt(src, p.x - 1, p.y + 1);
    const float bc = lumaAt(src, p.x,     p.y + 1);
    const float br = lumaAt(src, p.x + 1, p.y + 1);

    const float gx = (tr + cWeight * mr + br) - (tl + cWeight * ml + bl);
    const float gy = (bl + cWeight * bc + br) - (tl + cWeight * tc + tr);
    return length(float2(gx, gy));
}

kernel void sobelKernel(
    texture2d<float, access::read>  src [[texture(0)]],
    texture2d<float, access::write> dst [[texture(1)]],
    constant EdgeStrengthUniforms&  u   [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    const float mag = gradientMagnitude(src, int2(gid), 2.0f) * u.edgeStrength;
    dst.write(edgeFinish(float3(mag), src.read(gid).a), gid);
}

kernel void prewittKernel(
    texture2d<float, access::read>  src [[texture(0)]],
    texture2d<float, access::write> dst [[texture(1)]],
    constant EdgeStrengthUniforms&  u   [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    const float mag = gradientMagnitude(src, int2(gid), 1.0f) * u.edgeStrength;
    dst.write(edgeFinish(float3(mag), src.read(gid).a), gid);
}

// ---------------------------------------------------------------------------
// 3. Generic 3×3 convolution (+ Emboss)
// ---------------------------------------------------------------------------

struct Convolution3x3Uniforms {
    float3x3 kernelMatrix;   // column c = kernel column c, rows top→bottom
    float    bias;           // added to the result (0.5 for emboss-style)
};

kernel void convolution3x3Kernel(
    texture2d<float, access::read>   src [[texture(0)]],
    texture2d<float, access::write>  dst [[texture(1)]],
    constant Convolution3x3Uniforms& u   [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    const int w = int(dst.get_width());
    const int h = int(dst.get_height());

    float3 sum = float3(0.0f);
    for (int j = -1; j <= 1; ++j) {
        for (int i = -1; i <= 1; ++i) {
            const uint2 p = uint2(clamp(int(gid.x) + i, 0, w - 1),
                                  clamp(int(gid.y) + j, 0, h - 1));
            // kernelMatrix[column][row]; column = i+1, row = j+1.
            sum += saturate(src.read(p).rgb) * u.kernelMatrix[i + 1][j + 1];
        }
    }
    dst.write(edgeFinish(sum + u.bias, src.read(gid).a), gid);
}

// ---------------------------------------------------------------------------
// 4. Laplacian
// ---------------------------------------------------------------------------

struct LaplacianUniforms {
    float strength;   // output gain
};

kernel void laplacianKernel(
    texture2d<float, access::read>  src [[texture(0)]],
    texture2d<float, access::write> dst [[texture(1)]],
    constant LaplacianUniforms&     u   [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    const int2 p = int2(gid);

    // 8-neighbour Laplacian on luminance, biased to mid-grey so both edge
    // polarities are visible.
    float sum = -8.0f * lumaAt(src, p.x, p.y);
    for (int j = -1; j <= 1; ++j) {
        for (int i = -1; i <= 1; ++i) {
            if (i == 0 && j == 0) continue;
            sum += lumaAt(src, p.x + i, p.y + j);
        }
    }
    const float v = sum * u.strength + 0.5f;
    dst.write(edgeFinish(float3(v), src.read(gid).a), gid);
}

// ---------------------------------------------------------------------------
// 5. Canny — pass 1: smoothed gradient magnitude + direction
// ---------------------------------------------------------------------------

kernel void cannyGradientKernel(
    texture2d<float, access::read>  src [[texture(0)]],
    texture2d<float, access::write> dst [[texture(1)]],   // r = mag, gb = dir
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    const int2 p = int2(gid);

    // Light 3×3 Gaussian smoothing folded into the Sobel taps (the σ≈1 blur
    // of textbook Canny, collapsed into one pass for speed).
    float sm[3][3];
    for (int j = -1; j <= 1; ++j) {
        for (int i = -1; i <= 1; ++i) {
            float acc = 0.0f;
            const float wsum = 16.0f;
            acc += 4.0f * lumaAt(src, p.x + i,     p.y + j);
            acc += 2.0f * (lumaAt(src, p.x + i - 1, p.y + j) + lumaAt(src, p.x + i + 1, p.y + j)
                         + lumaAt(src, p.x + i, p.y + j - 1) + lumaAt(src, p.x + i, p.y + j + 1));
            acc += 1.0f * (lumaAt(src, p.x + i - 1, p.y + j - 1) + lumaAt(src, p.x + i + 1, p.y + j - 1)
                         + lumaAt(src, p.x + i - 1, p.y + j + 1) + lumaAt(src, p.x + i + 1, p.y + j + 1));
            sm[j + 1][i + 1] = acc / wsum;
        }
    }

    const float gx = (sm[0][2] + 2.0f * sm[1][2] + sm[2][2])
                   - (sm[0][0] + 2.0f * sm[1][0] + sm[2][0]);
    const float gy = (sm[2][0] + 2.0f * sm[2][1] + sm[2][2])
                   - (sm[0][0] + 2.0f * sm[0][1] + sm[0][2]);

    const float mag = length(float2(gx, gy));
    const float2 dir = (mag > 1e-5f) ? float2(gx, gy) / mag : float2(0.0f);
    // Pack direction into [0,1] for the unorm intermediate texture.
    dst.write(float4(saturate(mag), dir * 0.5f + 0.5f, 1.0f), gid);
}

// ---------------------------------------------------------------------------
// 5b. Canny — pass 2: NMS along the gradient + double threshold
// ---------------------------------------------------------------------------

struct CannyUniforms {
    float lowerThreshold;
    float upperThreshold;
};

kernel void cannySuppressKernel(
    texture2d<float, access::read>  grad [[texture(0)]],   // pass-1 output
    texture2d<float, access::write> dst  [[texture(1)]],
    constant CannyUniforms&         u    [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    const int w = int(dst.get_width());
    const int h = int(dst.get_height());

    const float4 g = grad.read(gid);
    const float mag = g.r;
    const float2 dir = g.gb * 2.0f - 1.0f;

    // Sample the two neighbours along the gradient direction.
    const int2 step1 = int2(round(dir.x), round(dir.y));
    const uint2 pa = uint2(clamp(int(gid.x) + step1.x, 0, w - 1),
                           clamp(int(gid.y) + step1.y, 0, h - 1));
    const uint2 pb = uint2(clamp(int(gid.x) - step1.x, 0, w - 1),
                           clamp(int(gid.y) - step1.y, 0, h - 1));

    const bool isMax = mag >= grad.read(pa).r && mag >= grad.read(pb).r;

    // Double threshold (no hysteresis walk — strong edges keep full white,
    // weak-but-plausible edges get half intensity).
    float v = 0.0f;
    if (isMax && mag >= u.lowerThreshold) {
        v = (mag >= u.upperThreshold) ? 1.0f : 0.5f;
    }
    dst.write(float4(v, v, v, 1.0f), gid);
}

// ---------------------------------------------------------------------------
// 6. Harris corner response
// ---------------------------------------------------------------------------

struct HarrisUniforms {
    float sensitivity;   // Harris k (≈0.04…0.15)
    float threshold;     // response cut-off
};

kernel void harrisCornerKernel(
    texture2d<float, access::read>  src [[texture(0)]],
    texture2d<float, access::write> dst [[texture(1)]],
    constant HarrisUniforms&        u   [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    const int2 p = int2(gid);

    // Structure tensor accumulated over a 5×5 window of central-difference
    // derivatives. O(25) reads per pixel — fine at video resolutions.
    float sxx = 0.0f, syy = 0.0f, sxy = 0.0f;
    for (int j = -2; j <= 2; ++j) {
        for (int i = -2; i <= 2; ++i) {
            const float ix = lumaAt(src, p.x + i + 1, p.y + j) - lumaAt(src, p.x + i - 1, p.y + j);
            const float iy = lumaAt(src, p.x + i, p.y + j + 1) - lumaAt(src, p.x + i, p.y + j - 1);
            sxx += ix * ix;
            syy += iy * iy;
            sxy += ix * iy;
        }
    }
    sxx /= 25.0f; syy /= 25.0f; sxy /= 25.0f;

    // Harris response: det(M) − k·trace(M)².
    const float det = sxx * syy - sxy * sxy;
    const float trace = sxx + syy;
    const float response = det - u.sensitivity * trace * trace;

    const float v = response > u.threshold ? 1.0f : 0.0f;
    dst.write(float4(v, v, v, 1.0f), gid);
}

// ---------------------------------------------------------------------------
// 7. Non-maximum suppression (luminance, 3×3)
// ---------------------------------------------------------------------------

struct NMSUniforms {
    float threshold;   // minimum luminance to survive at all
};

kernel void nonMaximumSuppressionKernel(
    texture2d<float, access::read>  src [[texture(0)]],
    texture2d<float, access::write> dst [[texture(1)]],
    constant NMSUniforms&           u   [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    const int2 p = int2(gid);
    const float centre = lumaAt(src, p.x, p.y);

    bool isMax = centre >= u.threshold;
    for (int j = -1; j <= 1 && isMax; ++j) {
        for (int i = -1; i <= 1; ++i) {
            if (i == 0 && j == 0) continue;
            if (lumaAt(src, p.x + i, p.y + j) > centre) { isMax = false; break; }
        }
    }

    const float4 c = src.read(gid);
    dst.write(isMax ? edgeFinish(c.rgb, c.a) : float4(0, 0, 0, c.a), gid);
}
