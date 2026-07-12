#include <metal_stdlib>
using namespace metal;

// ===========================================================================
// ArtisticKernels — v0.6.0 Artistic Effects Pack.
//
//   toonKernel             colour posterisation + dark Sobel outlines
//   smoothToonKernel       toon with inline 3×3 pre-smoothing
//   sketchKernel           inverted gradient — pencil on white paper
//   thresholdSketchKernel  binary ink lines on white
//   crosshatchKernel       luma-driven diagonal hatching
//   halftoneKernel         newspaper-style B/W luminance dots
//   polkaDotKernel         coloured dots on black, sized by cell luminance
//   kuwaharaKernel         oil-painting smoothing (min-variance quadrant)
//   pixellateKernel        square mosaic snap
//   polarPixellateKernel   radial/angular mosaic around a centre
//   mosaicKernel           averaged tiles with grout lines
//   cgaColorspaceKernel    chunky pixels + 4-colour CGA palette
//
// Stylisation is display-referred: inputs are clamped to [0,1]; isHDR only
// gates the final clamp so the formats round-trip.
// ===========================================================================

constant bool isHDR [[function_constant(0)]];

constant float3 kLumaW709 = float3(0.2126f, 0.7152f, 0.0722f);

static inline float lumaRead(texture2d<float, access::read> t, int x, int y) {
    const int w = int(t.get_width());
    const int h = int(t.get_height());
    return dot(saturate(t.read(uint2(clamp(x, 0, w - 1), clamp(y, 0, h - 1))).rgb), kLumaW709);
}

static inline float3 rgbRead(texture2d<float, access::read> t, int x, int y) {
    const int w = int(t.get_width());
    const int h = int(t.get_height());
    return saturate(t.read(uint2(clamp(x, 0, w - 1), clamp(y, 0, h - 1))).rgb);
}

static inline float4 artFinish(float3 rgb, float a) {
    return isHDR ? float4(max(rgb, 0.0f), a) : float4(saturate(rgb), a);
}

static inline float sobelMag(texture2d<float, access::read> src, int2 p) {
    const float tl = lumaRead(src, p.x - 1, p.y - 1);
    const float tc = lumaRead(src, p.x,     p.y - 1);
    const float tr = lumaRead(src, p.x + 1, p.y - 1);
    const float ml = lumaRead(src, p.x - 1, p.y);
    const float mr = lumaRead(src, p.x + 1, p.y);
    const float bl = lumaRead(src, p.x - 1, p.y + 1);
    const float bc = lumaRead(src, p.x,     p.y + 1);
    const float br = lumaRead(src, p.x + 1, p.y + 1);
    const float gx = (tr + 2.0f * mr + br) - (tl + 2.0f * ml + bl);
    const float gy = (bl + 2.0f * bc + br) - (tl + 2.0f * tc + tr);
    return length(float2(gx, gy));
}

// ---------------------------------------------------------------------------
// 1/2. Toon & SmoothToon
// ---------------------------------------------------------------------------

struct ToonUniforms {
    float threshold;            // edge blackness cut-off
    float quantizationLevels;   // colour steps per channel
};

static inline float4 toonShade(
    texture2d<float, access::read> src, uint2 gid, float3 rgb,
    float threshold, float levels)
{
    const float mag = sobelMag(src, int2(gid));
    const float3 quantised = floor(rgb * levels + 0.5f) / levels;
    // Edges go black; everything else gets the posterised colour.
    const float edge = smoothstep(threshold, threshold + 0.05f, mag);
    return float4(quantised * (1.0f - edge), 1.0f);
}

kernel void toonKernel(
    texture2d<float, access::read>  src [[texture(0)]],
    texture2d<float, access::write> dst [[texture(1)]],
    constant ToonUniforms&          u   [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    const float4 c = src.read(gid);
    const float4 r = toonShade(src, gid, saturate(c.rgb), u.threshold, u.quantizationLevels);
    dst.write(artFinish(r.rgb, c.a), gid);
}

kernel void smoothToonKernel(
    texture2d<float, access::read>  src [[texture(0)]],
    texture2d<float, access::write> dst [[texture(1)]],
    constant ToonUniforms&          u   [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    // Inline 3×3 box smoothing tames sensor noise before quantisation, which
    // is what separates "smooth toon" from plain toon.
    float3 acc = float3(0.0f);
    for (int j = -1; j <= 1; ++j)
        for (int i = -1; i <= 1; ++i)
            acc += rgbRead(src, int(gid.x) + i, int(gid.y) + j);
    const float3 smoothed = acc / 9.0f;

    const float4 c = src.read(gid);
    const float4 r = toonShade(src, gid, smoothed, u.threshold, u.quantizationLevels);
    dst.write(artFinish(r.rgb, c.a), gid);
}

// ---------------------------------------------------------------------------
// 3/4. Sketch & ThresholdSketch
// ---------------------------------------------------------------------------

struct SketchUniforms {
    float edgeStrength;
};

kernel void sketchKernel(
    texture2d<float, access::read>  src [[texture(0)]],
    texture2d<float, access::write> dst [[texture(1)]],
    constant SketchUniforms&        u   [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    const float mag = sobelMag(src, int2(gid)) * u.edgeStrength;
    const float v = 1.0f - saturate(mag);   // pencil: dark lines on white
    dst.write(artFinish(float3(v), src.read(gid).a), gid);
}

struct ThresholdSketchUniforms {
    float threshold;
    float edgeStrength;
};

kernel void thresholdSketchKernel(
    texture2d<float, access::read>    src [[texture(0)]],
    texture2d<float, access::write>   dst [[texture(1)]],
    constant ThresholdSketchUniforms& u   [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    const float mag = sobelMag(src, int2(gid)) * u.edgeStrength;
    const float v = mag > u.threshold ? 0.0f : 1.0f;   // pure ink on paper
    dst.write(artFinish(float3(v), src.read(gid).a), gid);
}

// ---------------------------------------------------------------------------
// 5. Crosshatch
// ---------------------------------------------------------------------------

struct CrosshatchUniforms {
    float spacing;     // hatch line spacing in pixels
    float lineWidth;   // hatch stroke width in pixels
};

kernel void crosshatchKernel(
    texture2d<float, access::read>  src [[texture(0)]],
    texture2d<float, access::write> dst [[texture(1)]],
    constant CrosshatchUniforms&    u   [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    const float luma = lumaRead(src, int(gid.x), int(gid.y));
    const float x = float(gid.x);
    const float y = float(gid.y);

    // Darker pixels accumulate more diagonal stroke families (GPUImage look).
    float v = 1.0f;
    if (luma < 1.00f && fmod(x + y,        u.spacing) <= u.lineWidth) v = 0.0f;
    if (luma < 0.75f && fmod(x - y,        u.spacing) <= u.lineWidth) v = 0.0f;
    if (luma < 0.50f && fmod(x + y - u.spacing * 0.5f, u.spacing) <= u.lineWidth) v = 0.0f;
    if (luma < 0.25f && fmod(x - y - u.spacing * 0.5f, u.spacing) <= u.lineWidth) v = 0.0f;

    dst.write(artFinish(float3(v), src.read(gid).a), gid);
}

// ---------------------------------------------------------------------------
// 6/7. Halftone & PolkaDot
// ---------------------------------------------------------------------------

struct DotUniforms {
    float dotSizePx;     // grid period in pixels
    float dotScaling;    // max dot radius as a fraction of the half-cell
};

kernel void halftoneKernel(
    texture2d<float, access::read>  src [[texture(0)]],
    texture2d<float, access::write> dst [[texture(1)]],
    constant DotUniforms&           u   [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    const float2 pos = float2(gid);
    const float2 cellCentre = (floor(pos / u.dotSizePx) + 0.5f) * u.dotSizePx;

    const float cellLuma = lumaRead(src, int(cellCentre.x), int(cellCentre.y));
    // Dark cells get big ink dots: radius grows with (1 - luma).
    const float radius = u.dotSizePx * 0.5f * u.dotScaling * sqrt(1.0f - cellLuma);
    const float d = distance(pos, cellCentre);
    const float v = smoothstep(radius - 0.75f, radius + 0.75f, d);

    dst.write(artFinish(float3(v), src.read(gid).a), gid);
}

kernel void polkaDotKernel(
    texture2d<float, access::read>  src [[texture(0)]],
    texture2d<float, access::write> dst [[texture(1)]],
    constant DotUniforms&           u   [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    const float2 pos = float2(gid);
    const float2 cellCentre = (floor(pos / u.dotSizePx) + 0.5f) * u.dotSizePx;

    const float3 cellColor = rgbRead(src, int(cellCentre.x), int(cellCentre.y));
    // Constant-size coloured dots on black (the classic Lichtenstein look).
    const float radius = u.dotSizePx * 0.5f * u.dotScaling;
    const float d = distance(pos, cellCentre);
    const float inDot = 1.0f - smoothstep(radius - 0.75f, radius + 0.75f, d);

    dst.write(artFinish(cellColor * inDot, src.read(gid).a), gid);
}

// ---------------------------------------------------------------------------
// 8. Kuwahara (oil painting)
// ---------------------------------------------------------------------------

struct KuwaharaUniforms {
    float radius;   // quadrant radius in pixels (3…6 sensible)
};

kernel void kuwaharaKernel(
    texture2d<float, access::read>  src [[texture(0)]],
    texture2d<float, access::write> dst [[texture(1)]],
    constant KuwaharaUniforms&      u   [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    const int2 p = int2(gid);
    const int r = clamp(int(u.radius), 1, 6);

    // Four overlapping (r+1)² quadrants; output the mean of the quadrant with
    // the lowest luminance variance. This flattens texture while keeping
    // edges crisp — the classic oil-paint look.
    float3 bestMean = float3(0.0f);
    float bestVar = INFINITY;

    for (int q = 0; q < 4; ++q) {
        const int sx = (q & 1) == 0 ? -r : 0;
        const int sy = (q & 2) == 0 ? -r : 0;

        float3 sum = float3(0.0f);
        float lumSum = 0.0f, lumSq = 0.0f;
        const float n = float((r + 1) * (r + 1));

        for (int j = 0; j <= r; ++j) {
            for (int i = 0; i <= r; ++i) {
                const float3 c = rgbRead(src, p.x + sx + i, p.y + sy + j);
                const float l = dot(c, kLumaW709);
                sum += c;
                lumSum += l;
                lumSq  += l * l;
            }
        }
        const float mean = lumSum / n;
        const float variance = lumSq / n - mean * mean;
        if (variance < bestVar) {
            bestVar = variance;
            bestMean = sum / n;
        }
    }

    dst.write(artFinish(bestMean, src.read(gid).a), gid);
}

// ---------------------------------------------------------------------------
// 9/10. Pixellate & PolarPixellate
// ---------------------------------------------------------------------------

struct PixellateUniforms {
    float fractionalCellSize;   // cell edge in pixels
};

kernel void pixellateKernel(
    texture2d<float, access::read>  src [[texture(0)]],
    texture2d<float, access::write> dst [[texture(1)]],
    constant PixellateUniforms&     u   [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    const float2 cellCentre = (floor(float2(gid) / u.fractionalCellSize) + 0.5f) * u.fractionalCellSize;
    const float3 c = rgbRead(src, int(cellCentre.x), int(cellCentre.y));
    dst.write(artFinish(c, src.read(gid).a), gid);
}

struct PolarPixellateUniforms {
    float2 center;      // normalised
    float  radialSize;  // normalised radial step
    float  angularSize; // angular step in radians
};

kernel void polarPixellateKernel(
    texture2d<float, access::read>  src [[texture(0)]],
    texture2d<float, access::write> dst [[texture(1)]],
    constant PolarPixellateUniforms& u  [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    const float2 size = float2(dst.get_width(), dst.get_height());
    const float2 uv = (float2(gid) + 0.5f) / size;

    // Snap polar coordinates around the centre → concentric wedge tiles.
    const float2 d = uv - u.center;
    const float r = length(d);
    const float a = atan2(d.y, d.x);
    const float rSnap = floor(r / u.radialSize  + 0.5f) * u.radialSize;
    const float aSnap = floor(a / u.angularSize + 0.5f) * u.angularSize;
    const float2 snapped = u.center + rSnap * float2(cos(aSnap), sin(aSnap));

    const int2 p = int2(clamp(snapped, float2(0.0f), float2(1.0f)) * (size - 1.0f));
    dst.write(artFinish(rgbRead(src, p.x, p.y), src.read(gid).a), gid);
}

// ---------------------------------------------------------------------------
// 11. Mosaic (averaged tiles + grout)
// ---------------------------------------------------------------------------

struct MosaicUniforms {
    float tileSizePx;   // tile edge in pixels
    float groutWidth;   // dark seam width in pixels (0 = none)
};

kernel void mosaicKernel(
    texture2d<float, access::read>  src [[texture(0)]],
    texture2d<float, access::write> dst [[texture(1)]],
    constant MosaicUniforms&        u   [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    const float2 pos = float2(gid);
    const float2 tileOrigin = floor(pos / u.tileSizePx) * u.tileSizePx;

    // Average 4 spread samples inside the tile — close enough to a true mean
    // at a fraction of the cost.
    const float q = u.tileSizePx * 0.25f;
    float3 acc = float3(0.0f);
    acc += rgbRead(src, int(tileOrigin.x + q),        int(tileOrigin.y + q));
    acc += rgbRead(src, int(tileOrigin.x + 3.0f * q), int(tileOrigin.y + q));
    acc += rgbRead(src, int(tileOrigin.x + q),        int(tileOrigin.y + 3.0f * q));
    acc += rgbRead(src, int(tileOrigin.x + 3.0f * q), int(tileOrigin.y + 3.0f * q));
    float3 tile = acc / 4.0f;

    // Dark grout seams on the tile borders sell the ceramic look.
    const float2 inTile = pos - tileOrigin;
    const float edgeDist = min(min(inTile.x, inTile.y),
                               min(u.tileSizePx - inTile.x, u.tileSizePx - inTile.y));
    if (edgeDist < u.groutWidth) tile *= 0.35f;

    dst.write(artFinish(tile, src.read(gid).a), gid);
}

// ---------------------------------------------------------------------------
// 12. CGA colorspace
// ---------------------------------------------------------------------------

kernel void cgaColorspaceKernel(
    texture2d<float, access::read>  src [[texture(0)]],
    texture2d<float, access::write> dst [[texture(1)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;

    // Chunky 2×2 sampling then nearest CGA palette-1 colour
    // (black, cyan, magenta, white) — the 1981 IBM look.
    const uint2 p = uint2((gid.x / 2) * 2, (gid.y / 2) * 2);
    const float3 c = saturate(src.read(p).rgb);

    const float3 palette[4] = {
        float3(0.0f, 0.0f, 0.0f),
        float3(0.0f, 1.0f, 1.0f),
        float3(1.0f, 0.0f, 1.0f),
        float3(1.0f, 1.0f, 1.0f),
    };

    float bestDist = INFINITY;
    float3 best = palette[0];
    for (int i = 0; i < 4; ++i) {
        const float d = distance_squared(c, palette[i]);
        if (d < bestDist) { bestDist = d; best = palette[i]; }
    }

    dst.write(float4(best, src.read(gid).a), gid);
}
