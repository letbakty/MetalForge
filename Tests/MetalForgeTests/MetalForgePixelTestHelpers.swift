import XCTest
import Metal
@testable import MetalForge

// ===========================================================================
// Pixel-level test support for the GPU filter behavior tests.
//
// These helpers build small, CPU-readable textures with known contents, run a
// filter end-to-end on the GPU, read the result back, and compare colours with
// a tolerance (GPU float math and 8-bit quantisation make exact equality
// brittle). Everything uses `.shared` storage so the CPU can populate the source
// and read the destination without blits.
// ===========================================================================

enum PixelTest {

    // MARK: - Texture allocation

    /// A CPU-readable/writable texture (shared storage, shader read + write).
    static func makeTexture(
        device: MTLDevice,
        width: Int,
        height: Int,
        pixelFormat: MTLPixelFormat = .rgba8Unorm
    ) -> MTLTexture? {
        let desc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: pixelFormat, width: width, height: height, mipmapped: false
        )
        desc.storageMode = .shared
        desc.usage = [.shaderRead, .shaderWrite]
        return device.makeTexture(descriptor: desc)
    }

    // MARK: - Filling (8-bit)

    /// Fill an `.rgba8Unorm` texture from a per-pixel colour generator
    /// (components in `[0, 1]`).
    static func fill(
        _ texture: MTLTexture,
        generator: (_ x: Int, _ y: Int) -> SIMD4<Float>
    ) {
        let w = texture.width, h = texture.height
        var bytes = [UInt8](repeating: 0, count: w * h * 4)
        for y in 0..<h {
            for x in 0..<w {
                let c = generator(x, y)
                let i = (y * w + x) * 4
                bytes[i + 0] = toByte(c.x)
                bytes[i + 1] = toByte(c.y)
                bytes[i + 2] = toByte(c.z)
                bytes[i + 3] = toByte(c.w)
            }
        }
        texture.replace(
            region: MTLRegionMake2D(0, 0, w, h),
            mipmapLevel: 0,
            withBytes: bytes,
            bytesPerRow: w * 4
        )
    }

    /// Read an `.rgba8Unorm` texture back into a flat byte buffer.
    static func readBytes(_ texture: MTLTexture) -> [UInt8] {
        let w = texture.width, h = texture.height
        var bytes = [UInt8](repeating: 0, count: w * h * 4)
        texture.getBytes(
            &bytes,
            bytesPerRow: w * 4,
            from: MTLRegionMake2D(0, 0, w, h),
            mipmapLevel: 0
        )
        return bytes
    }

    /// Colour of one pixel from an `.rgba8Unorm` readback buffer (`[0, 1]`).
    static func color(_ bytes: [UInt8], x: Int, y: Int, width: Int) -> SIMD4<Float> {
        let i = (y * width + x) * 4
        return SIMD4(
            Float(bytes[i + 0]) / 255.0,
            Float(bytes[i + 1]) / 255.0,
            Float(bytes[i + 2]) / 255.0,
            Float(bytes[i + 3]) / 255.0
        )
    }

    // MARK: - Filling / reading (16-bit float, HDR path)

    /// Fill an `.rgba16Float` texture from a per-pixel colour generator.
    static func fillHDR(
        _ texture: MTLTexture,
        generator: (_ x: Int, _ y: Int) -> SIMD4<Float>
    ) {
        let w = texture.width, h = texture.height
        var halfs = [Float16](repeating: 0, count: w * h * 4)
        for y in 0..<h {
            for x in 0..<w {
                let c = generator(x, y)
                let i = (y * w + x) * 4
                halfs[i + 0] = Float16(c.x)
                halfs[i + 1] = Float16(c.y)
                halfs[i + 2] = Float16(c.z)
                halfs[i + 3] = Float16(c.w)
            }
        }
        halfs.withUnsafeBytes { raw in
            texture.replace(
                region: MTLRegionMake2D(0, 0, w, h),
                mipmapLevel: 0,
                withBytes: raw.baseAddress!,
                bytesPerRow: w * 4 * MemoryLayout<Float16>.size
            )
        }
    }

    /// Colour of one pixel from an `.rgba16Float` texture (`[0, 1+]`).
    static func colorHDR(_ texture: MTLTexture, x: Int, y: Int) -> SIMD4<Float> {
        let w = texture.width, h = texture.height
        var halfs = [Float16](repeating: 0, count: w * h * 4)
        halfs.withUnsafeMutableBytes { raw in
            texture.getBytes(
                raw.baseAddress!,
                bytesPerRow: w * 4 * MemoryLayout<Float16>.size,
                from: MTLRegionMake2D(0, 0, w, h),
                mipmapLevel: 0
            )
        }
        let i = (y * w + x) * 4
        return SIMD4(
            Float(halfs[i + 0]), Float(halfs[i + 1]),
            Float(halfs[i + 2]), Float(halfs[i + 3])
        )
    }

    // MARK: - Known test patterns (all `.rgba8Unorm`)

    /// Uniform colour across the whole texture.
    static func solidColor(
        device: MTLDevice, size: Int, rgba: SIMD4<Float>
    ) -> MTLTexture? {
        guard let tex = makeTexture(device: device, width: size, height: size) else { return nil }
        fill(tex) { _, _ in rgba }
        return tex
    }

    /// Black texture with a single white pixel at the centre.
    static func centerBright(device: MTLDevice, size: Int) -> MTLTexture? {
        guard let tex = makeTexture(device: device, width: size, height: size) else { return nil }
        let cx = size / 2, cy = size / 2
        fill(tex) { x, y in
            (x == cx && y == cy) ? SIMD4(1, 1, 1, 1) : SIMD4(0, 0, 0, 1)
        }
        return tex
    }

    /// Vertical edge: left half dark, right half bright.
    static func verticalEdge(
        device: MTLDevice, size: Int,
        dark: SIMD4<Float> = SIMD4(0, 0, 0, 1),
        bright: SIMD4<Float> = SIMD4(1, 1, 1, 1)
    ) -> MTLTexture? {
        guard let tex = makeTexture(device: device, width: size, height: size) else { return nil }
        fill(tex) { x, _ in x < size / 2 ? dark : bright }
        return tex
    }

    /// Horizontal edge: top half dark, bottom half bright.
    static func horizontalEdge(device: MTLDevice, size: Int) -> MTLTexture? {
        guard let tex = makeTexture(device: device, width: size, height: size) else { return nil }
        fill(tex) { _, y in y < size / 2 ? SIMD4(0, 0, 0, 1) : SIMD4(1, 1, 1, 1) }
        return tex
    }

    /// Multi-channel pattern: left half pure red, right half pure blue. Useful
    /// for RGB-split, where shifting channels in different directions across the
    /// colour boundary produces channel-specific changes.
    static func rgbZones(device: MTLDevice, size: Int) -> MTLTexture? {
        guard let tex = makeTexture(device: device, width: size, height: size) else { return nil }
        fill(tex) { x, _ in
            x < size / 2 ? SIMD4(1, 0, 0, 1) : SIMD4(0, 0, 1, 1)
        }
        return tex
    }

    /// Horizontal luminance gradient from black (left) to white (right).
    static func gradient(device: MTLDevice, size: Int) -> MTLTexture? {
        guard let tex = makeTexture(device: device, width: size, height: size) else { return nil }
        fill(tex) { x, _ in
            let v = size > 1 ? Float(x) / Float(size - 1) : 0
            return SIMD4(v, v, v, 1)
        }
        return tex
    }

    // MARK: - Running a filter

    /// Encode a filter on `source`, run it to completion, and return the
    /// CPU-readable destination texture (same size/format as the source).
    static func run(
        _ filter: any MetalForgeFilter,
        source: MTLTexture,
        device: MTLDevice,
        queue: MTLCommandQueue
    ) -> MTLTexture? {
        guard
            let dest = makeTexture(
                device: device, width: source.width, height: source.height,
                pixelFormat: source.pixelFormat
            ),
            let cb = queue.makeCommandBuffer()
        else { return nil }

        filter.encode(source: source, destination: dest, commandBuffer: cb)
        cb.commit()
        cb.waitUntilCompleted()
        return cb.status == .completed ? dest : nil
    }

    // MARK: - Misc

    private static func toByte(_ v: Float) -> UInt8 {
        UInt8((min(max(v, 0), 1) * 255.0).rounded())
    }

    /// Average of the R, G, B channels — a cheap luminance proxy for brightness
    /// comparisons that don't care about colour.
    static func luma(_ c: SIMD4<Float>) -> Float {
        (c.x + c.y + c.z) / 3.0
    }
}
