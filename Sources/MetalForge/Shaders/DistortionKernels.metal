#include <metal_stdlib>
using namespace metal;

// ===========================================================================
// DistortionKernels — v0.3.0 Distortion Pack.
//
//   bulgeDistortionKernel     radial magnify (fisheye)
//   pinchDistortionKernel     radial squeeze
//   stretchDistortionKernel   centre-anchored stretch
//   swirlDistortionKernel     angular twist falling off with radius
//   sphereRefractionKernel    refraction through a glass ball (black outside)
//   glassSphereKernel         refraction with the scene kept outside
//   cropKernel                normalised crop rect, zoom-to-fill
//   transformKernel           affine UV transform around the centre
//
// All kernels are UV-remap effects: they sample the source with a bilinear
// sampler at a computed coordinate. Colours are untouched, so the isHDR
// constant only gates the final safety clamp.
// ===========================================================================

constant bool isHDR [[function_constant(0)]];

constexpr sampler bilinearEdge(coord::normalized, filter::linear, address::clamp_to_edge);
constexpr sampler bilinearZero(coord::normalized, filter::linear, address::clamp_to_zero);

static inline float4 distortFinish(float4 c) {
    return isHDR ? max(c, 0.0f) : saturate(c);
}

/// Aspect-corrected distance helpers: circular effects must operate in a
/// square space or circles become ellipses on non-square frames.
static inline float2 toCircleSpace(float2 uv, float2 center, float aspect) {
    return float2(uv.x - center.x, (uv.y - center.y) * aspect);
}
static inline float2 fromCircleSpace(float2 p, float2 center, float aspect) {
    return float2(p.x + center.x, p.y / aspect + center.y);
}

// ---------------------------------------------------------------------------
// 1. Bulge
// ---------------------------------------------------------------------------

struct BulgeUniforms {
    float2 center;   // normalised effect centre
    float  radius;   // normalised radius of the affected disc
    float  scale;    // -1 … 1; >0 magnifies (bulge), <0 shrinks
};

kernel void bulgeDistortionKernel(
    texture2d<float, access::sample> src [[texture(0)]],
    texture2d<float, access::write>  dst [[texture(1)]],
    constant BulgeUniforms&          u   [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    const float2 size = float2(dst.get_width(), dst.get_height());
    const float aspect = size.y / size.x;
    const float2 uv = (float2(gid) + 0.5f) / size;

    float2 p = toCircleSpace(uv, u.center, aspect);
    const float dist = length(p);
    if (dist < u.radius) {
        // Pull samples toward the centre near the middle of the disc —
        // magnification — with a smooth quadratic falloff to the rim.
        float percent = 1.0f - ((u.radius - dist) / u.radius) * u.scale;
        percent = percent * percent;
        p *= percent;
    }
    const float2 sampleUV = fromCircleSpace(p, u.center, aspect);
    dst.write(distortFinish(src.sample(bilinearEdge, sampleUV)), gid);
}

// ---------------------------------------------------------------------------
// 2. Pinch
// ---------------------------------------------------------------------------

struct PinchUniforms {
    float2 center;
    float  radius;
    float  scale;    // 0 … 2; >0 pinches inward
};

kernel void pinchDistortionKernel(
    texture2d<float, access::sample> src [[texture(0)]],
    texture2d<float, access::write>  dst [[texture(1)]],
    constant PinchUniforms&          u   [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    const float2 size = float2(dst.get_width(), dst.get_height());
    const float aspect = size.y / size.x;
    const float2 uv = (float2(gid) + 0.5f) / size;

    float2 p = toCircleSpace(uv, u.center, aspect);
    const float dist = length(p);
    if (dist < u.radius) {
        // Push samples away from the centre → the image appears sucked in.
        const float percent = 1.0f + ((u.radius - dist) / u.radius) * u.scale;
        p *= percent;
    }
    const float2 sampleUV = fromCircleSpace(p, u.center, aspect);
    dst.write(distortFinish(src.sample(bilinearEdge, sampleUV)), gid);
}

// ---------------------------------------------------------------------------
// 3. Stretch
// ---------------------------------------------------------------------------

struct StretchUniforms {
    float2 center;
};

kernel void stretchDistortionKernel(
    texture2d<float, access::sample> src [[texture(0)]],
    texture2d<float, access::write>  dst [[texture(1)]],
    constant StretchUniforms&        u   [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    const float2 size = float2(dst.get_width(), dst.get_height());
    const float2 uv = (float2(gid) + 0.5f) / size;

    // GPUImage-style stretch: the centre region is magnified, the outer band
    // compressed, with a smoothstep transition between the two.
    float2 normCoord  = 2.0f * uv - 1.0f;
    const float2 normCenter = 2.0f * u.center - 1.0f;
    normCoord -= normCenter;
    const float2 s = sign(normCoord);
    float2 a = abs(normCoord);
    a = 0.5f * a + 0.5f * smoothstep(0.25f, 0.5f, a) * a;
    normCoord = s * a + normCenter;

    const float2 sampleUV = normCoord * 0.5f + 0.5f;
    dst.write(distortFinish(src.sample(bilinearEdge, sampleUV)), gid);
}

// ---------------------------------------------------------------------------
// 4. Swirl
// ---------------------------------------------------------------------------

struct SwirlUniforms {
    float2 center;
    float  radius;
    float  angle;    // radians of twist at the centre
};

kernel void swirlDistortionKernel(
    texture2d<float, access::sample> src [[texture(0)]],
    texture2d<float, access::write>  dst [[texture(1)]],
    constant SwirlUniforms&          u   [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    const float2 size = float2(dst.get_width(), dst.get_height());
    const float aspect = size.y / size.x;
    const float2 uv = (float2(gid) + 0.5f) / size;

    float2 p = toCircleSpace(uv, u.center, aspect);
    const float dist = length(p);
    if (dist < u.radius) {
        // Twist angle decays linearly to zero at the rim.
        const float percent = (u.radius - dist) / u.radius;
        const float theta = percent * percent * u.angle;
        const float sinT = sin(theta);
        const float cosT = cos(theta);
        p = float2(p.x * cosT - p.y * sinT,
                   p.x * sinT + p.y * cosT);
    }
    const float2 sampleUV = fromCircleSpace(p, u.center, aspect);
    dst.write(distortFinish(src.sample(bilinearEdge, sampleUV)), gid);
}

// ---------------------------------------------------------------------------
// 5/6. Sphere refraction & glass sphere
// ---------------------------------------------------------------------------

struct SphereUniforms {
    float2 center;
    float  radius;
    float  refractiveIndex;   // ≈0.71 glass
};

/// Shared refraction maths: returns the refracted sample UV, and whether the
/// pixel lies inside the sphere via `inside`.
static inline float2 sphereRefractUV(
    float2 uv, constant SphereUniforms& u, float aspect, thread bool& inside)
{
    const float2 p = toCircleSpace(uv, u.center, aspect);
    const float dist = length(p);
    inside = dist < u.radius;
    if (!inside) return uv;

    // Build the sphere surface normal at this point and refract a view ray
    // (0,0,-1) through it.
    const float2 pn = p / u.radius;
    const float depth = sqrt(max(1.0f - dot(pn, pn), 0.0f));
    const float3 normal = normalize(float3(pn, depth));
    const float3 refracted = refract(float3(0.0f, 0.0f, -1.0f), normal, u.refractiveIndex);
    return (refracted.xy + 1.0f) * 0.5f;
}

kernel void sphereRefractionKernel(
    texture2d<float, access::sample> src [[texture(0)]],
    texture2d<float, access::write>  dst [[texture(1)]],
    constant SphereUniforms&         u   [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    const float2 size = float2(dst.get_width(), dst.get_height());
    const float aspect = size.y / size.x;
    const float2 uv = (float2(gid) + 0.5f) / size;

    bool inside = false;
    const float2 sampleUV = sphereRefractUV(uv, u, aspect, inside);
    // Classic GPUImage behaviour: the world outside the ball is black.
    const float4 c = inside ? src.sample(bilinearEdge, sampleUV) : float4(0, 0, 0, 1);
    dst.write(distortFinish(c), gid);
}

kernel void glassSphereKernel(
    texture2d<float, access::sample> src [[texture(0)]],
    texture2d<float, access::write>  dst [[texture(1)]],
    constant SphereUniforms&         u   [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    const float2 size = float2(dst.get_width(), dst.get_height());
    const float aspect = size.y / size.x;
    const float2 uv = (float2(gid) + 0.5f) / size;

    bool inside = false;
    // Glass ball flips the refracted image; mirror the sample around centre.
    float2 sampleUV = sphereRefractUV(uv, u, aspect, inside);
    if (inside) {
        sampleUV = 1.0f - sampleUV;

        // Cheap rim lighting: brighten toward the silhouette edge.
        const float2 p = toCircleSpace(uv, u.center, aspect);
        const float rim = smoothstep(u.radius * 0.7f, u.radius, length(p));
        float4 c = src.sample(bilinearEdge, sampleUV);
        c.rgb += rim * 0.25f;
        dst.write(distortFinish(c), gid);
        return;
    }
    // Outside the ball the scene is untouched.
    dst.write(distortFinish(src.sample(bilinearEdge, uv)), gid);
}

// ---------------------------------------------------------------------------
// 7. Crop (zoom-to-fill)
// ---------------------------------------------------------------------------

struct CropUniforms {
    float2 origin;   // normalised top-left of the crop rect
    float2 cropSize; // normalised width/height of the crop rect
};

kernel void cropKernel(
    texture2d<float, access::sample> src [[texture(0)]],
    texture2d<float, access::write>  dst [[texture(1)]],
    constant CropUniforms&           u   [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    const float2 size = float2(dst.get_width(), dst.get_height());
    const float2 uv = (float2(gid) + 0.5f) / size;

    // The pipeline's destination is fixed-size, so the crop region is scaled
    // back up to fill the frame (crop-and-zoom).
    const float2 sampleUV = u.origin + uv * u.cropSize;
    dst.write(distortFinish(src.sample(bilinearEdge, sampleUV)), gid);
}

// ---------------------------------------------------------------------------
// 8. Affine transform
// ---------------------------------------------------------------------------

struct TransformUniforms {
    float3x3 matrix;   // maps centred destination UV → centred source UV
};

kernel void transformKernel(
    texture2d<float, access::sample> src [[texture(0)]],
    texture2d<float, access::write>  dst [[texture(1)]],
    constant TransformUniforms&      u   [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    const float2 size = float2(dst.get_width(), dst.get_height());
    const float2 uv = (float2(gid) + 0.5f) / size;

    // Work in centred coordinates so rotation/scale pivot on the frame centre.
    const float3 centred = float3(uv - 0.5f, 1.0f);
    const float3 mapped  = u.matrix * centred;
    const float2 sampleUV = mapped.xy + 0.5f;

    // clamp_to_zero → pixels mapped from outside the source are transparent black.
    dst.write(distortFinish(src.sample(bilinearZero, sampleUV)), gid);
}
