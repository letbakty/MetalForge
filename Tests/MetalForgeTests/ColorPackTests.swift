import XCTest
import Metal
import simd
@testable import MetalForge

// ===========================================================================
// v0.2.0 Color & Stylization Pack — constructibility + pixel-behaviour tests.
// ===========================================================================

final class ColorPackTests: XCTestCase {

    private var engine: MetalForgeEngine!

    override func setUpWithError() throws {
        engine = try MetalForgeEngine()
    }

    // MARK: - Constructibility (every filter in the pack)

    func testColorPackFiltersAreConstructible() throws {
        XCTAssertNoThrow(try GammaFilter(engine: engine))
        XCTAssertNoThrow(try LevelsFilter(engine: engine))
        XCTAssertNoThrow(try HueRotateFilter(engine: engine))
        XCTAssertNoThrow(try VibranceFilter(engine: engine))
        XCTAssertNoThrow(try WhiteBalanceFilter(engine: engine))
        XCTAssertNoThrow(try ToneCurveFilter(engine: engine))
        XCTAssertNoThrow(try HighlightShadowFilter(engine: engine))
        XCTAssertNoThrow(try HighlightShadowTintFilter(engine: engine))
        XCTAssertNoThrow(try ColorMatrixFilter(engine: engine))
        XCTAssertNoThrow(try ColorInvertFilter(engine: engine))
        XCTAssertNoThrow(try MonochromeFilter(engine: engine))
        XCTAssertNoThrow(try FalseColorFilter(engine: engine))
    }

    func testTonePackFiltersAreConstructible() throws {
        XCTAssertNoThrow(try GrayscaleFilter(engine: engine))
        XCTAssertNoThrow(try SepiaFilter(engine: engine))
        XCTAssertNoThrow(try HazeFilter(engine: engine))
        XCTAssertNoThrow(try SkinToneFilter(engine: engine))
        XCTAssertNoThrow(try LuminanceThresholdFilter(engine: engine))
        XCTAssertNoThrow(try AdaptiveThresholdFilter(engine: engine))
        XCTAssertNoThrow(try PosterizeFilter(engine: engine))
        XCTAssertNoThrow(try ColorHalftoneFilter(engine: engine))
    }

    // MARK: - Identity behaviour

    func testGammaIdentityLeavesPixelsUnchanged() throws {
        let filter = try GammaFilter(engine: engine)
        filter.gamma = 1.0
        let src = try XCTUnwrap(PixelTest.solidColor(
            device: engine.device, size: 8, rgba: SIMD4(0.25, 0.5, 0.75, 1)))
        let dst = try XCTUnwrap(PixelTest.run(filter, source: src, device: engine.device, queue: engine.commandQueue))
        let c = PixelTest.color(PixelTest.readBytes(dst), x: 4, y: 4, width: 8)
        XCTAssertEqual(c.x, 0.25, accuracy: 0.02)
        XCTAssertEqual(c.y, 0.5,  accuracy: 0.02)
        XCTAssertEqual(c.z, 0.75, accuracy: 0.02)
    }

    func testLevelsIdentityLeavesPixelsUnchanged() throws {
        let filter = try LevelsFilter(engine: engine)
        let src = try XCTUnwrap(PixelTest.solidColor(
            device: engine.device, size: 8, rgba: SIMD4(0.3, 0.6, 0.9, 1)))
        let dst = try XCTUnwrap(PixelTest.run(filter, source: src, device: engine.device, queue: engine.commandQueue))
        let c = PixelTest.color(PixelTest.readBytes(dst), x: 4, y: 4, width: 8)
        XCTAssertEqual(c.x, 0.3, accuracy: 0.02)
        XCTAssertEqual(c.y, 0.6, accuracy: 0.02)
        XCTAssertEqual(c.z, 0.9, accuracy: 0.02)
    }

    func testToneCurveIdentityLeavesPixelsUnchanged() throws {
        let filter = try ToneCurveFilter(engine: engine)
        let src = try XCTUnwrap(PixelTest.solidColor(
            device: engine.device, size: 8, rgba: SIMD4(0.2, 0.5, 0.8, 1)))
        let dst = try XCTUnwrap(PixelTest.run(filter, source: src, device: engine.device, queue: engine.commandQueue))
        let c = PixelTest.color(PixelTest.readBytes(dst), x: 4, y: 4, width: 8)
        XCTAssertEqual(c.x, 0.2, accuracy: 0.02)
        XCTAssertEqual(c.y, 0.5, accuracy: 0.02)
        XCTAssertEqual(c.z, 0.8, accuracy: 0.02)
    }

    func testSepiaZeroIntensityIsIdentity() throws {
        let filter = try SepiaFilter(engine: engine)
        filter.intensity = 0
        let src = try XCTUnwrap(PixelTest.solidColor(
            device: engine.device, size: 8, rgba: SIMD4(0.1, 0.7, 0.4, 1)))
        let dst = try XCTUnwrap(PixelTest.run(filter, source: src, device: engine.device, queue: engine.commandQueue))
        let c = PixelTest.color(PixelTest.readBytes(dst), x: 4, y: 4, width: 8)
        XCTAssertEqual(c.x, 0.1, accuracy: 0.02)
        XCTAssertEqual(c.y, 0.7, accuracy: 0.02)
        XCTAssertEqual(c.z, 0.4, accuracy: 0.02)
    }

    // MARK: - Effect behaviour

    func testGammaAboveOneDarkensMidGrey() throws {
        let filter = try GammaFilter(engine: engine)
        filter.gamma = 2.2
        let src = try XCTUnwrap(PixelTest.solidColor(
            device: engine.device, size: 8, rgba: SIMD4(0.5, 0.5, 0.5, 1)))
        let dst = try XCTUnwrap(PixelTest.run(filter, source: src, device: engine.device, queue: engine.commandQueue))
        let c = PixelTest.color(PixelTest.readBytes(dst), x: 4, y: 4, width: 8)
        XCTAssertLessThan(c.x, 0.4, "gamma 2.2 should darken mid grey")
    }

    func testColorInvertInvertsChannels() throws {
        let filter = try ColorInvertFilter(engine: engine)
        let src = try XCTUnwrap(PixelTest.solidColor(
            device: engine.device, size: 8, rgba: SIMD4(0.2, 0.4, 0.8, 1)))
        let dst = try XCTUnwrap(PixelTest.run(filter, source: src, device: engine.device, queue: engine.commandQueue))
        let c = PixelTest.color(PixelTest.readBytes(dst), x: 4, y: 4, width: 8)
        XCTAssertEqual(c.x, 0.8, accuracy: 0.02)
        XCTAssertEqual(c.y, 0.6, accuracy: 0.02)
        XCTAssertEqual(c.z, 0.2, accuracy: 0.02)
    }

    func testGrayscaleEqualisesChannels() throws {
        let filter = try GrayscaleFilter(engine: engine)
        let src = try XCTUnwrap(PixelTest.solidColor(
            device: engine.device, size: 8, rgba: SIMD4(1, 0, 0, 1)))
        let dst = try XCTUnwrap(PixelTest.run(filter, source: src, device: engine.device, queue: engine.commandQueue))
        let c = PixelTest.color(PixelTest.readBytes(dst), x: 4, y: 4, width: 8)
        XCTAssertEqual(c.x, c.y, accuracy: 0.01)
        XCTAssertEqual(c.y, c.z, accuracy: 0.01)
        // BT.709: pure red collapses to ≈0.2126.
        XCTAssertEqual(c.x, 0.2126, accuracy: 0.03)
    }

    func testLuminanceThresholdBinarisesGradient() throws {
        let filter = try LuminanceThresholdFilter(engine: engine)
        filter.threshold = 0.5
        let src = try XCTUnwrap(PixelTest.gradient(device: engine.device, size: 16))
        let dst = try XCTUnwrap(PixelTest.run(filter, source: src, device: engine.device, queue: engine.commandQueue))
        let bytes = PixelTest.readBytes(dst)
        let dark   = PixelTest.color(bytes, x: 1,  y: 8, width: 16)
        let bright = PixelTest.color(bytes, x: 14, y: 8, width: 16)
        XCTAssertEqual(PixelTest.luma(dark),   0, accuracy: 0.01)
        XCTAssertEqual(PixelTest.luma(bright), 1, accuracy: 0.01)
    }

    func testPosterizeTwoLevelsLimitsDistinctValues() throws {
        let filter = try PosterizeFilter(engine: engine)
        filter.levels = 2
        let src = try XCTUnwrap(PixelTest.gradient(device: engine.device, size: 16))
        let dst = try XCTUnwrap(PixelTest.run(filter, source: src, device: engine.device, queue: engine.commandQueue))
        let bytes = PixelTest.readBytes(dst)
        var distinct = Set<UInt8>()
        for x in 0..<16 {
            distinct.insert(bytes[(8 * 16 + x) * 4])
        }
        XCTAssertLessThanOrEqual(distinct.count, 3, "2-level posterize should leave ≤3 distinct red values")
    }

    func testFalseColorMapsExtremes() throws {
        let filter = try FalseColorFilter(engine: engine)
        filter.firstColor  = SIMD3(0, 0, 1)
        filter.secondColor = SIMD3(1, 0, 0)
        let src = try XCTUnwrap(PixelTest.verticalEdge(device: engine.device, size: 8))
        let dst = try XCTUnwrap(PixelTest.run(filter, source: src, device: engine.device, queue: engine.commandQueue))
        let bytes = PixelTest.readBytes(dst)
        let darkSide   = PixelTest.color(bytes, x: 1, y: 4, width: 8)
        let brightSide = PixelTest.color(bytes, x: 6, y: 4, width: 8)
        XCTAssertGreaterThan(darkSide.z, 0.9,   "black maps to firstColor (blue)")
        XCTAssertGreaterThan(brightSide.x, 0.9, "white maps to secondColor (red)")
    }

    func testHueRotate180ChangesRedSignificantly() throws {
        let filter = try HueRotateFilter(engine: engine)
        filter.angleDegrees = 180
        let src = try XCTUnwrap(PixelTest.solidColor(
            device: engine.device, size: 8, rgba: SIMD4(1, 0, 0, 1)))
        let dst = try XCTUnwrap(PixelTest.run(filter, source: src, device: engine.device, queue: engine.commandQueue))
        let c = PixelTest.color(PixelTest.readBytes(dst), x: 4, y: 4, width: 8)
        // 180° rotation must move red away from the red primary.
        XCTAssertLessThan(c.x, 0.6)
        XCTAssertGreaterThan(c.y + c.z, 0.3)
    }

    func testVibranceBoostsMutedColor() throws {
        let filter = try VibranceFilter(engine: engine)
        filter.vibrance = 1.0
        let src = try XCTUnwrap(PixelTest.solidColor(
            device: engine.device, size: 8, rgba: SIMD4(0.6, 0.5, 0.4, 1)))
        let dst = try XCTUnwrap(PixelTest.run(filter, source: src, device: engine.device, queue: engine.commandQueue))
        let c = PixelTest.color(PixelTest.readBytes(dst), x: 4, y: 4, width: 8)
        let satBefore: Float = 0.6 - 0.4
        let satAfter = c.x - c.z
        XCTAssertGreaterThan(satAfter, satBefore + 0.02, "positive vibrance should widen channel spread")
    }

    func testWhiteBalanceWarmsImage() throws {
        let filter = try WhiteBalanceFilter(engine: engine)
        filter.temperature = 1.0
        let src = try XCTUnwrap(PixelTest.solidColor(
            device: engine.device, size: 8, rgba: SIMD4(0.5, 0.5, 0.5, 1)))
        let dst = try XCTUnwrap(PixelTest.run(filter, source: src, device: engine.device, queue: engine.commandQueue))
        let c = PixelTest.color(PixelTest.readBytes(dst), x: 4, y: 4, width: 8)
        XCTAssertGreaterThan(c.x, c.z + 0.05, "warm temperature pushes red above blue")
    }

    func testMonochromeProducesTintedOutput() throws {
        let filter = try MonochromeFilter(engine: engine)
        filter.filterColor = SIMD3(1, 0, 0)
        filter.intensity = 1
        let src = try XCTUnwrap(PixelTest.solidColor(
            device: engine.device, size: 8, rgba: SIMD4(0.2, 0.6, 0.9, 1)))
        let dst = try XCTUnwrap(PixelTest.run(filter, source: src, device: engine.device, queue: engine.commandQueue))
        let c = PixelTest.color(PixelTest.readBytes(dst), x: 4, y: 4, width: 8)
        XCTAssertGreaterThan(c.x, c.y, "red-tinted monochrome keeps red dominant")
        XCTAssertGreaterThan(c.x, c.z)
    }

    func testColorMatrixSwapsChannels() throws {
        let filter = try ColorMatrixFilter(engine: engine)
        // Swap R and B via the matrix (columns map input components).
        filter.matrix = simd_float4x4(
            SIMD4(0, 0, 1, 0),   // input r → output b
            SIMD4(0, 1, 0, 0),
            SIMD4(1, 0, 0, 0),   // input b → output r
            SIMD4(0, 0, 0, 1)
        )
        let src = try XCTUnwrap(PixelTest.solidColor(
            device: engine.device, size: 8, rgba: SIMD4(1, 0, 0, 1)))
        let dst = try XCTUnwrap(PixelTest.run(filter, source: src, device: engine.device, queue: engine.commandQueue))
        let c = PixelTest.color(PixelTest.readBytes(dst), x: 4, y: 4, width: 8)
        XCTAssertLessThan(c.x, 0.05)
        XCTAssertGreaterThan(c.z, 0.95)
    }

    func testAdaptiveThresholdProcessesEdgeTexture() throws {
        let filter = try AdaptiveThresholdFilter(engine: engine)
        let src = try XCTUnwrap(PixelTest.verticalEdge(device: engine.device, size: 16))
        let dst = try XCTUnwrap(PixelTest.run(filter, source: src, device: engine.device, queue: engine.commandQueue))
        let bytes = PixelTest.readBytes(dst)
        // Output must be binary (every pixel 0 or 1 in luminance).
        for x in [1, 8, 14] {
            let c = PixelTest.color(bytes, x: x, y: 8, width: 16)
            let l = PixelTest.luma(c)
            XCTAssertTrue(l < 0.05 || l > 0.95, "adaptive threshold output must be binary, got \(l)")
        }
    }

    func testColorHalftoneProducesSpatialVariation() throws {
        let filter = try ColorHalftoneFilter(engine: engine)
        filter.dotSize = 6
        let src = try XCTUnwrap(PixelTest.solidColor(
            device: engine.device, size: 32, rgba: SIMD4(0.5, 0.5, 0.5, 1)))
        let dst = try XCTUnwrap(PixelTest.run(filter, source: src, device: engine.device, queue: engine.commandQueue))
        let bytes = PixelTest.readBytes(dst)
        var minL: Float = 1, maxL: Float = 0
        for y in 0..<32 {
            for x in 0..<32 {
                let l = PixelTest.luma(PixelTest.color(bytes, x: x, y: y, width: 32))
                minL = min(minL, l); maxL = max(maxL, l)
            }
        }
        XCTAssertGreaterThan(maxL - minL, 0.3, "halftone dots should create strong spatial contrast on flat grey")
    }

    func testHighlightShadowLiftsShadows() throws {
        let filter = try HighlightShadowFilter(engine: engine)
        filter.shadows = 1.0
        let src = try XCTUnwrap(PixelTest.solidColor(
            device: engine.device, size: 8, rgba: SIMD4(0.1, 0.1, 0.1, 1)))
        let dst = try XCTUnwrap(PixelTest.run(filter, source: src, device: engine.device, queue: engine.commandQueue))
        let c = PixelTest.color(PixelTest.readBytes(dst), x: 4, y: 4, width: 8)
        XCTAssertGreaterThan(PixelTest.luma(c), 0.12, "full shadow lift should brighten dark grey")
    }

    func testHighlightShadowTintTintsShadows() throws {
        let filter = try HighlightShadowTintFilter(engine: engine)
        filter.shadowTintColor = SIMD3(1, 0, 0)
        filter.shadowTintIntensity = 1.0
        let src = try XCTUnwrap(PixelTest.solidColor(
            device: engine.device, size: 8, rgba: SIMD4(0.15, 0.15, 0.15, 1)))
        let dst = try XCTUnwrap(PixelTest.run(filter, source: src, device: engine.device, queue: engine.commandQueue))
        let c = PixelTest.color(PixelTest.readBytes(dst), x: 4, y: 4, width: 8)
        XCTAssertGreaterThanOrEqual(c.x, c.y, "red shadow tint should not lower red below green")
    }

    func testToneCurveBakeIdentity() {
        let curve = ToneCurveFilter.bakeCurve([SIMD2(0, 0), SIMD2(1, 1)])
        XCTAssertEqual(curve.count, 256)
        XCTAssertEqual(curve[0], 0, accuracy: 0.005)
        XCTAssertEqual(curve[128], 128.0 / 255.0, accuracy: 0.01)
        XCTAssertEqual(curve[255], 1, accuracy: 0.005)
    }

    func testColorPackHandlesHDRTexture() throws {
        // Spot-check the HDR PSO path with a couple of representative filters.
        let filters: [any MetalForgeFilter] = [
            try GammaFilter(engine: engine),
            try GrayscaleFilter(engine: engine),
            try VibranceFilter(engine: engine),
        ]
        guard let src = PixelTest.makeTexture(
            device: engine.device, width: 8, height: 8, pixelFormat: .rgba16Float
        ) else { return XCTFail("no HDR texture") }
        PixelTest.fillHDR(src) { _, _ in SIMD4(1.5, 0.5, 0.25, 1) }

        for filter in filters {
            let dst = PixelTest.run(filter, source: src, device: engine.device, queue: engine.commandQueue)
            XCTAssertNotNil(dst, "HDR encode failed for \(type(of: filter))")
        }
    }
}
