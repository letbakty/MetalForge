import XCTest
import Metal
import simd
@testable import MetalForge

// ===========================================================================
// v0.7.0 Advanced Blur & Morphology Pack — behaviour tests.
// ===========================================================================

final class BlurMorphologyPackTests: XCTestCase {

    private var engine: MetalForgeEngine!

    override func setUpWithError() throws {
        engine = try MetalForgeEngine()
    }

    func testBlurFiltersAreConstructible() throws {
        XCTAssertNoThrow(try BoxBlurFilter(engine: engine))
        XCTAssertNoThrow(try DirectionalMotionBlurFilter(engine: engine))
        XCTAssertNoThrow(try ZoomBlurFilter(engine: engine))
        XCTAssertNoThrow(try TiltShiftFilter(engine: engine))
        XCTAssertNoThrow(try BilateralBlurFilter(engine: engine))
        XCTAssertNoThrow(try MedianBlurFilter(engine: engine))
        XCTAssertNoThrow(try LensBlurFilter(engine: engine))
        XCTAssertNoThrow(try SurfaceBlurFilter(engine: engine))
        XCTAssertNoThrow(try IOSBlurFilter(engine: engine))
    }

    func testMorphologyFiltersAreConstructible() throws {
        XCTAssertNoThrow(try DilationFilter(engine: engine))
        XCTAssertNoThrow(try ErosionFilter(engine: engine))
        XCTAssertNoThrow(try OpeningFilter(engine: engine))
        XCTAssertNoThrow(try ClosingFilter(engine: engine))
    }

    // MARK: - Blur behaviour

    func testBoxBlurSpreadsCentrePixel() throws {
        let filter = try BoxBlurFilter(engine: engine)
        filter.radius = 2
        let src = try XCTUnwrap(PixelTest.centerBright(device: engine.device, size: 9))
        let dst = try XCTUnwrap(PixelTest.run(filter, source: src, device: engine.device, queue: engine.commandQueue))
        let bytes = PixelTest.readBytes(dst)
        let centre    = PixelTest.luma(PixelTest.color(bytes, x: 4, y: 4, width: 9))
        let neighbour = PixelTest.luma(PixelTest.color(bytes, x: 6, y: 4, width: 9))
        XCTAssertLessThan(centre, 0.9, "energy must leave the centre")
        XCTAssertGreaterThan(neighbour, 0.01, "energy must arrive at neighbours")
    }

    func testDirectionalBlurSmearsAlongAxisOnly() throws {
        let filter = try DirectionalMotionBlurFilter(engine: engine)
        filter.angle = 0           // horizontal streaks
        filter.radius = 4
        let src = try XCTUnwrap(PixelTest.horizontalEdge(device: engine.device, size: 16))
        let dst = try XCTUnwrap(PixelTest.run(filter, source: src, device: engine.device, queue: engine.commandQueue))
        let bytes = PixelTest.readBytes(dst)
        // A horizontal edge blurred horizontally must stay perfectly sharp.
        XCTAssertEqual(PixelTest.luma(PixelTest.color(bytes, x: 8, y: 2,  width: 16)), 0, accuracy: 0.05)
        XCTAssertEqual(PixelTest.luma(PixelTest.color(bytes, x: 8, y: 13, width: 16)), 1, accuracy: 0.05)
    }

    func testDirectionalBlurVerticalSoftensHorizontalEdge() throws {
        let filter = try DirectionalMotionBlurFilter(engine: engine)
        filter.angle = .pi / 2     // vertical streaks
        filter.radius = 4
        let src = try XCTUnwrap(PixelTest.horizontalEdge(device: engine.device, size: 16))
        let dst = try XCTUnwrap(PixelTest.run(filter, source: src, device: engine.device, queue: engine.commandQueue))
        let bytes = PixelTest.readBytes(dst)
        let boundary = PixelTest.luma(PixelTest.color(bytes, x: 8, y: 8, width: 16))
        XCTAssertGreaterThan(boundary, 0.15, "vertical streaks must soften the boundary")
        XCTAssertLessThan(boundary, 0.85)
    }

    func testZoomBlurZeroStrengthIsIdentity() throws {
        let filter = try ZoomBlurFilter(engine: engine)
        filter.strength = 0
        let src = try XCTUnwrap(PixelTest.verticalEdge(device: engine.device, size: 16))
        let dst = try XCTUnwrap(PixelTest.run(filter, source: src, device: engine.device, queue: engine.commandQueue))
        let bytes = PixelTest.readBytes(dst)
        XCTAssertEqual(PixelTest.luma(PixelTest.color(bytes, x: 2,  y: 8, width: 16)), 0, accuracy: 0.05)
        XCTAssertEqual(PixelTest.luma(PixelTest.color(bytes, x: 13, y: 8, width: 16)), 1, accuracy: 0.05)
    }

    func testTiltShiftKeepsFocusBandSharp() throws {
        let filter = try TiltShiftFilter(engine: engine)
        filter.focusCenter = 0.5
        filter.focusWidth = 0.2
        filter.blurRadius = 6
        let src = try XCTUnwrap(PixelTest.verticalEdge(device: engine.device, size: 32))
        let dst = try XCTUnwrap(PixelTest.run(filter, source: src, device: engine.device, queue: engine.commandQueue))
        let bytes = PixelTest.readBytes(dst)
        // Mid-frame row is inside the focus band → edge stays a hard step.
        XCTAssertLessThan(PixelTest.luma(PixelTest.color(bytes, x: 14, y: 16, width: 32)), 0.15)
        XCTAssertGreaterThan(PixelTest.luma(PixelTest.color(bytes, x: 17, y: 16, width: 32)), 0.85)
    }

    func testBilateralSmoothsButKeepsStrongEdge() throws {
        let filter = try BilateralBlurFilter(engine: engine)
        filter.radius = 4
        filter.rangeSigma = 0.1
        let src = try XCTUnwrap(PixelTest.verticalEdge(device: engine.device, size: 16))
        let dst = try XCTUnwrap(PixelTest.run(filter, source: src, device: engine.device, queue: engine.commandQueue))
        let bytes = PixelTest.readBytes(dst)
        // The hard black/white edge survives because colour distance kills
        // cross-edge weights.
        XCTAssertLessThan(PixelTest.luma(PixelTest.color(bytes, x: 6, y: 8, width: 16)), 0.15)
        XCTAssertGreaterThan(PixelTest.luma(PixelTest.color(bytes, x: 9, y: 8, width: 16)), 0.85)
    }

    func testMedianRemovesSaltNoise() throws {
        // A single white pixel in a black field is the textbook median victim.
        let filter = try MedianBlurFilter(engine: engine)
        let src = try XCTUnwrap(PixelTest.centerBright(device: engine.device, size: 9))
        let dst = try XCTUnwrap(PixelTest.run(filter, source: src, device: engine.device, queue: engine.commandQueue))
        let bytes = PixelTest.readBytes(dst)
        XCTAssertLessThan(PixelTest.luma(PixelTest.color(bytes, x: 4, y: 4, width: 9)), 0.05,
                          "isolated bright pixel must vanish")
    }

    func testSurfaceBlurPreservesEdge() throws {
        let filter = try SurfaceBlurFilter(engine: engine)
        filter.threshold = 0.2
        let src = try XCTUnwrap(PixelTest.verticalEdge(device: engine.device, size: 16))
        let dst = try XCTUnwrap(PixelTest.run(filter, source: src, device: engine.device, queue: engine.commandQueue))
        let bytes = PixelTest.readBytes(dst)
        XCTAssertLessThan(PixelTest.luma(PixelTest.color(bytes, x: 6, y: 8, width: 16)), 0.1)
        XCTAssertGreaterThan(PixelTest.luma(PixelTest.color(bytes, x: 9, y: 8, width: 16)), 0.9)
    }

    func testLensBlurBloomsBrightPixel() throws {
        let filter = try LensBlurFilter(engine: engine)
        filter.radius = 4
        filter.brightness = 0.8
        let src = try XCTUnwrap(PixelTest.centerBright(device: engine.device, size: 16))
        let dst = try XCTUnwrap(PixelTest.run(filter, source: src, device: engine.device, queue: engine.commandQueue))
        let bytes = PixelTest.readBytes(dst)
        // Highlight-weighted bokeh: pixels a few px away light up strongly.
        let nearby = PixelTest.luma(PixelTest.color(bytes, x: 10, y: 8, width: 16))
        XCTAssertGreaterThan(nearby, 0.3, "bokeh must bloom the highlight outward")
    }

    func testIOSBlurProducesFrostedFlatField() throws {
        let filter = try IOSBlurFilter(engine: engine)
        let src = try XCTUnwrap(PixelTest.solidColor(
            device: engine.device, size: 16, rgba: SIMD4(0.5, 0.2, 0.2, 1)))
        let dst = try XCTUnwrap(PixelTest.run(filter, source: src, device: engine.device, queue: engine.commandQueue))
        let c = PixelTest.color(PixelTest.readBytes(dst), x: 8, y: 8, width: 16)
        // White mix must brighten; saturation boost keeps red dominant.
        XCTAssertGreaterThan(c.x, 0.5)
        XCTAssertGreaterThan(c.x, c.y)
    }

    // MARK: - Morphology behaviour

    func testDilationGrowsBrightDot() throws {
        let filter = try DilationFilter(engine: engine)
        filter.radius = 2
        let src = try XCTUnwrap(PixelTest.centerBright(device: engine.device, size: 9))
        let dst = try XCTUnwrap(PixelTest.run(filter, source: src, device: engine.device, queue: engine.commandQueue))
        let bytes = PixelTest.readBytes(dst)
        XCTAssertGreaterThan(PixelTest.luma(PixelTest.color(bytes, x: 6, y: 6, width: 9)), 0.9,
                             "dilation must spread the white dot")
    }

    func testErosionRemovesBrightDot() throws {
        let filter = try ErosionFilter(engine: engine)
        filter.radius = 1
        let src = try XCTUnwrap(PixelTest.centerBright(device: engine.device, size: 9))
        let dst = try XCTUnwrap(PixelTest.run(filter, source: src, device: engine.device, queue: engine.commandQueue))
        let bytes = PixelTest.readBytes(dst)
        XCTAssertLessThan(PixelTest.luma(PixelTest.color(bytes, x: 4, y: 4, width: 9)), 0.05,
                          "erosion must delete a 1-pixel speck")
    }

    func testOpeningRemovesSpeckPermanently() throws {
        let filter = try OpeningFilter(engine: engine)
        filter.radius = 1
        let src = try XCTUnwrap(PixelTest.centerBright(device: engine.device, size: 9))
        let dst = try XCTUnwrap(PixelTest.run(filter, source: src, device: engine.device, queue: engine.commandQueue))
        let bytes = PixelTest.readBytes(dst)
        var maxL: Float = 0
        for y in 0..<9 {
            for x in 0..<9 {
                maxL = max(maxL, PixelTest.luma(PixelTest.color(bytes, x: x, y: y, width: 9)))
            }
        }
        XCTAssertLessThan(maxL, 0.05, "opening = erode→dilate must not resurrect the speck")
    }

    func testClosingFillsDarkHole() throws {
        // Inverse test: a single dark pixel in a white field.
        guard let src = PixelTest.makeTexture(device: engine.device, width: 9, height: 9) else {
            return XCTFail("no texture")
        }
        PixelTest.fill(src) { x, y in
            (x == 4 && y == 4) ? SIMD4(0, 0, 0, 1) : SIMD4(1, 1, 1, 1)
        }
        let filter = try ClosingFilter(engine: engine)
        filter.radius = 1
        let dst = try XCTUnwrap(PixelTest.run(filter, source: src, device: engine.device, queue: engine.commandQueue))
        let bytes = PixelTest.readBytes(dst)
        XCTAssertGreaterThan(PixelTest.luma(PixelTest.color(bytes, x: 4, y: 4, width: 9)), 0.95,
                             "closing = dilate→erode must fill the dark hole")
    }

    func testBlurPackHandlesHDRTexture() throws {
        let filters: [any MetalForgeFilter] = [
            try BoxBlurFilter(engine: engine),
            try BilateralBlurFilter(engine: engine),
            try DilationFilter(engine: engine),
            try IOSBlurFilter(engine: engine),
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
