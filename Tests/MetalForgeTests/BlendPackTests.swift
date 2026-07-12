import XCTest
import Metal
import simd
@testable import MetalForge

// ===========================================================================
// v0.4.0 Blend Modes API — constructibility + compositing-behaviour tests.
// ===========================================================================

final class BlendPackTests: XCTestCase {

    private var engine: MetalForgeEngine!

    override func setUpWithError() throws {
        engine = try MetalForgeEngine()
    }

    private func runBlend(
        mode: BlendMode,
        base: SIMD4<Float>,
        overlay: SIMD4<Float>,
        intensity: Float = 1.0
    ) throws -> SIMD4<Float> {
        let filter = try BlendModeFilter(engine: engine)
        filter.mode = mode
        filter.intensity = intensity
        filter.overlayTexture = PixelTest.solidColor(device: engine.device, size: 8, rgba: overlay)
        let src = try XCTUnwrap(PixelTest.solidColor(device: engine.device, size: 8, rgba: base))
        let dst = try XCTUnwrap(PixelTest.run(filter, source: src, device: engine.device, queue: engine.commandQueue))
        return PixelTest.color(PixelTest.readBytes(dst), x: 4, y: 4, width: 8)
    }

    // MARK: - Construction

    func testBlendFiltersAreConstructible() throws {
        XCTAssertNoThrow(try BlendModeFilter(engine: engine))
        XCTAssertNoThrow(try AlphaBlendFilter(engine: engine))
        XCTAssertNoThrow(try ChromaKeyFilter(engine: engine))
        XCTAssertNoThrow(try MaskBlendFilter(engine: engine))
    }

    func testBlendModeEnumHasAllModes() {
        XCTAssertEqual(BlendMode.allCases.count, 23)
    }

    // MARK: - Pass-through without overlay

    func testBlendWithoutOverlayIsPassthrough() throws {
        let filter = try BlendModeFilter(engine: engine)
        filter.mode = .multiply
        let src = try XCTUnwrap(PixelTest.solidColor(
            device: engine.device, size: 8, rgba: SIMD4(0.3, 0.6, 0.9, 1)))
        let dst = try XCTUnwrap(PixelTest.run(filter, source: src, device: engine.device, queue: engine.commandQueue))
        let c = PixelTest.color(PixelTest.readBytes(dst), x: 4, y: 4, width: 8)
        XCTAssertEqual(c.x, 0.3, accuracy: 0.01)
        XCTAssertEqual(c.y, 0.6, accuracy: 0.01)
        XCTAssertEqual(c.z, 0.9, accuracy: 0.01)
    }

    // MARK: - Separable blend-mode math

    func testMultiplyBlend() throws {
        let c = try runBlend(mode: .multiply,
                             base: SIMD4(0.5, 0.5, 0.5, 1), overlay: SIMD4(0.5, 1.0, 0.0, 1))
        XCTAssertEqual(c.x, 0.25, accuracy: 0.02)
        XCTAssertEqual(c.y, 0.5,  accuracy: 0.02)
        XCTAssertEqual(c.z, 0.0,  accuracy: 0.02)
    }

    func testScreenBlend() throws {
        let c = try runBlend(mode: .screen,
                             base: SIMD4(0.5, 0.5, 0.5, 1), overlay: SIMD4(0.5, 0.0, 1.0, 1))
        XCTAssertEqual(c.x, 0.75, accuracy: 0.02)
        XCTAssertEqual(c.y, 0.5,  accuracy: 0.02)
        XCTAssertEqual(c.z, 1.0,  accuracy: 0.02)
    }

    func testDarkenAndLighten() throws {
        let dark = try runBlend(mode: .darken,
                                base: SIMD4(0.3, 0.8, 0.5, 1), overlay: SIMD4(0.6, 0.2, 0.5, 1))
        XCTAssertEqual(dark.x, 0.3, accuracy: 0.02)
        XCTAssertEqual(dark.y, 0.2, accuracy: 0.02)

        let light = try runBlend(mode: .lighten,
                                 base: SIMD4(0.3, 0.8, 0.5, 1), overlay: SIMD4(0.6, 0.2, 0.5, 1))
        XCTAssertEqual(light.x, 0.6, accuracy: 0.02)
        XCTAssertEqual(light.y, 0.8, accuracy: 0.02)
    }

    func testDifferenceBlend() throws {
        let c = try runBlend(mode: .difference,
                             base: SIMD4(0.8, 0.2, 0.5, 1), overlay: SIMD4(0.3, 0.6, 0.5, 1))
        XCTAssertEqual(c.x, 0.5, accuracy: 0.02)
        XCTAssertEqual(c.y, 0.4, accuracy: 0.02)
        XCTAssertEqual(c.z, 0.0, accuracy: 0.02)
    }

    func testSubtractBlend() throws {
        let c = try runBlend(mode: .subtract,
                             base: SIMD4(0.8, 0.2, 1.0, 1), overlay: SIMD4(0.3, 0.6, 0.4, 1))
        XCTAssertEqual(c.x, 0.5, accuracy: 0.02)
        XCTAssertEqual(c.y, 0.0, accuracy: 0.02)
        XCTAssertEqual(c.z, 0.6, accuracy: 0.02)
    }

    func testHardMixIsBinary() throws {
        let c = try runBlend(mode: .hardMix,
                             base: SIMD4(0.7, 0.2, 0.5, 1), overlay: SIMD4(0.5, 0.5, 0.6, 1))
        XCTAssertEqual(c.x, 1.0, accuracy: 0.01)   // 0.7+0.5 ≥ 1
        XCTAssertEqual(c.y, 0.0, accuracy: 0.01)   // 0.2+0.5 < 1
        XCTAssertEqual(c.z, 1.0, accuracy: 0.01)   // 0.5+0.6 ≥ 1
    }

    func testIntensityHalvesEffect() throws {
        let c = try runBlend(mode: .multiply,
                             base: SIMD4(1.0, 1.0, 1.0, 1), overlay: SIMD4(0.0, 0.0, 0.0, 1),
                             intensity: 0.5)
        XCTAssertEqual(c.x, 0.5, accuracy: 0.02, "50% intensity multiply with black halves white")
    }

    // MARK: - Non-separable modes

    func testLuminosityKeepsOverlayBrightness() throws {
        let base    = SIMD4<Float>(0.9, 0.1, 0.1, 1)   // bright red
        let overlay = SIMD4<Float>(0.1, 0.1, 0.1, 1)   // dark grey
        let c = try runBlend(mode: .luminosity, base: base, overlay: overlay)
        // Result takes the dark luminosity but keeps reddish hue dominance.
        XCTAssertLessThan(PixelTest.luma(c), 0.35)
        XCTAssertGreaterThanOrEqual(c.x, c.y)
    }

    func testColorModeTransfersHue() throws {
        let base    = SIMD4<Float>(0.5, 0.5, 0.5, 1)   // grey
        let overlay = SIMD4<Float>(1.0, 0.0, 0.0, 1)   // red
        let c = try runBlend(mode: .color, base: base, overlay: overlay)
        XCTAssertGreaterThan(c.x, c.y + 0.1, "colour mode should paint the grey red")
    }

    // MARK: - Dissolve

    func testDissolveFullCoverageShowsOverlay() throws {
        let c = try runBlend(mode: .dissolve,
                             base: SIMD4(0, 0, 0, 1), overlay: SIMD4(1, 1, 1, 1),
                             intensity: 1.0)
        XCTAssertEqual(PixelTest.luma(c), 1.0, accuracy: 0.01)
    }

    func testDissolveHalfCoverageMixesStatistically() throws {
        let filter = try BlendModeFilter(engine: engine)
        filter.mode = .dissolve
        filter.intensity = 0.5
        filter.overlayTexture = PixelTest.solidColor(
            device: engine.device, size: 32, rgba: SIMD4(1, 1, 1, 1))
        let src = try XCTUnwrap(PixelTest.solidColor(
            device: engine.device, size: 32, rgba: SIMD4(0, 0, 0, 1)))
        let dst = try XCTUnwrap(PixelTest.run(filter, source: src, device: engine.device, queue: engine.commandQueue))
        let bytes = PixelTest.readBytes(dst)
        var whites = 0
        for y in 0..<32 {
            for x in 0..<32 where PixelTest.luma(PixelTest.color(bytes, x: x, y: y, width: 32)) > 0.5 {
                whites += 1
            }
        }
        let coverage = Float(whites) / 1024.0
        XCTAssertEqual(coverage, 0.5, accuracy: 0.15, "50% dissolve should dither roughly half the pixels")
    }

    // MARK: - ChromaKey

    func testChromaKeyRemovesKeyColor() throws {
        let filter = try ChromaKeyFilter(engine: engine)
        filter.keyColor = SIMD3(0, 1, 0)
        let src = try XCTUnwrap(PixelTest.solidColor(
            device: engine.device, size: 8, rgba: SIMD4(0, 1, 0, 1)))
        let dst = try XCTUnwrap(PixelTest.run(filter, source: src, device: engine.device, queue: engine.commandQueue))
        let c = PixelTest.color(PixelTest.readBytes(dst), x: 4, y: 4, width: 8)
        XCTAssertEqual(c.w, 0, accuracy: 0.02, "pure green must be keyed to alpha 0")
    }

    func testChromaKeyKeepsForeground() throws {
        let filter = try ChromaKeyFilter(engine: engine)
        filter.keyColor = SIMD3(0, 1, 0)
        let src = try XCTUnwrap(PixelTest.solidColor(
            device: engine.device, size: 8, rgba: SIMD4(0.8, 0.2, 0.3, 1)))
        let dst = try XCTUnwrap(PixelTest.run(filter, source: src, device: engine.device, queue: engine.commandQueue))
        let c = PixelTest.color(PixelTest.readBytes(dst), x: 4, y: 4, width: 8)
        XCTAssertEqual(c.w, 1, accuracy: 0.02, "non-green pixels keep alpha 1")
        XCTAssertEqual(c.x, 0.8, accuracy: 0.03)
    }

    func testChromaKeyCompositesBackground() throws {
        let filter = try ChromaKeyFilter(engine: engine)
        filter.keyColor = SIMD3(0, 1, 0)
        filter.backgroundTexture = PixelTest.solidColor(
            device: engine.device, size: 8, rgba: SIMD4(0, 0, 1, 1))
        let src = try XCTUnwrap(PixelTest.solidColor(
            device: engine.device, size: 8, rgba: SIMD4(0, 1, 0, 1)))
        let dst = try XCTUnwrap(PixelTest.run(filter, source: src, device: engine.device, queue: engine.commandQueue))
        let c = PixelTest.color(PixelTest.readBytes(dst), x: 4, y: 4, width: 8)
        XCTAssertGreaterThan(c.z, 0.9, "keyed green is replaced by the blue background")
        XCTAssertLessThan(c.y, 0.1)
    }

    // MARK: - MaskBlend

    func testMaskBlendUsesMaskLuminance() throws {
        let filter = try MaskBlendFilter(engine: engine)
        filter.overlayTexture = PixelTest.solidColor(
            device: engine.device, size: 16, rgba: SIMD4(1, 0, 0, 1))
        // Mask: left half black (keep frame), right half white (show overlay).
        filter.maskTexture = PixelTest.verticalEdge(device: engine.device, size: 16)
        let src = try XCTUnwrap(PixelTest.solidColor(
            device: engine.device, size: 16, rgba: SIMD4(0, 0, 1, 1)))
        let dst = try XCTUnwrap(PixelTest.run(filter, source: src, device: engine.device, queue: engine.commandQueue))
        let bytes = PixelTest.readBytes(dst)
        let left  = PixelTest.color(bytes, x: 2,  y: 8, width: 16)
        let right = PixelTest.color(bytes, x: 13, y: 8, width: 16)
        XCTAssertGreaterThan(left.z, 0.9,  "black mask keeps the blue frame")
        XCTAssertGreaterThan(right.x, 0.9, "white mask shows the red overlay")
    }

    func testMaskBlendWithoutTexturesIsPassthrough() throws {
        let filter = try MaskBlendFilter(engine: engine)
        let src = try XCTUnwrap(PixelTest.solidColor(
            device: engine.device, size: 8, rgba: SIMD4(0.2, 0.4, 0.6, 1)))
        let dst = try XCTUnwrap(PixelTest.run(filter, source: src, device: engine.device, queue: engine.commandQueue))
        let c = PixelTest.color(PixelTest.readBytes(dst), x: 4, y: 4, width: 8)
        XCTAssertEqual(c.x, 0.2, accuracy: 0.01)
        XCTAssertEqual(c.y, 0.4, accuracy: 0.01)
        XCTAssertEqual(c.z, 0.6, accuracy: 0.01)
    }

    // MARK: - HDR

    func testBlendHandlesHDRTexture() throws {
        let filter = try BlendModeFilter(engine: engine)
        filter.mode = .screen
        filter.overlayTexture = PixelTest.solidColor(
            device: engine.device, size: 8, rgba: SIMD4(0.5, 0.5, 0.5, 1))
        guard let src = PixelTest.makeTexture(
            device: engine.device, width: 8, height: 8, pixelFormat: .rgba16Float
        ) else { return XCTFail("no HDR texture") }
        PixelTest.fillHDR(src) { _, _ in SIMD4(1.5, 0.5, 0.25, 1) }
        let dst = PixelTest.run(filter, source: src, device: engine.device, queue: engine.commandQueue)
        XCTAssertNotNil(dst)
        if let dst {
            let c = PixelTest.colorHDR(dst, x: 4, y: 4)
            XCTAssertGreaterThan(c.x, 1.0, "HDR residual above white must survive the blend")
        }
    }
}
