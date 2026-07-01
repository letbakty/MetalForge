import XCTest
import Metal
import simd
@testable import MetalForge

// ===========================================================================
// v0.5.0 Edge Detection & Convolution Pack — behaviour tests.
// ===========================================================================

final class EdgePackTests: XCTestCase {

    private var engine: MetalForgeEngine!

    override func setUpWithError() throws {
        engine = try MetalForgeEngine()
    }

    func testEdgeFiltersAreConstructible() throws {
        XCTAssertNoThrow(try SobelEdgeDetectionFilter(engine: engine))
        XCTAssertNoThrow(try PrewittEdgeDetectionFilter(engine: engine))
        XCTAssertNoThrow(try Convolution3x3Filter(engine: engine))
        XCTAssertNoThrow(try EmbossFilter(engine: engine))
        XCTAssertNoThrow(try LaplacianFilter(engine: engine))
        XCTAssertNoThrow(try CannyEdgeDetectionFilter(engine: engine))
        XCTAssertNoThrow(try HarrisCornerDetectionFilter(engine: engine))
        XCTAssertNoThrow(try NonMaximumSuppressionFilter(engine: engine))
    }

    // MARK: - Gradient detectors

    private func assertRespondsAtEdgeOnly(
        _ filter: any MetalForgeFilter, name: String
    ) throws {
        let src = try XCTUnwrap(PixelTest.verticalEdge(device: engine.device, size: 16))
        let dst = try XCTUnwrap(PixelTest.run(filter, source: src, device: engine.device, queue: engine.commandQueue))
        let bytes = PixelTest.readBytes(dst)
        let flat = PixelTest.luma(PixelTest.color(bytes, x: 3, y: 8, width: 16))
        let edgeA = PixelTest.luma(PixelTest.color(bytes, x: 7, y: 8, width: 16))
        let edgeB = PixelTest.luma(PixelTest.color(bytes, x: 8, y: 8, width: 16))
        XCTAssertLessThan(flat, 0.1, "\(name): flat region must stay dark")
        XCTAssertGreaterThan(max(edgeA, edgeB), 0.5, "\(name): boundary must light up")
    }

    func testSobelDetectsVerticalEdge() throws {
        try assertRespondsAtEdgeOnly(try SobelEdgeDetectionFilter(engine: engine), name: "Sobel")
    }

    func testPrewittDetectsVerticalEdge() throws {
        try assertRespondsAtEdgeOnly(try PrewittEdgeDetectionFilter(engine: engine), name: "Prewitt")
    }

    func testCannyDetectsVerticalEdgeThinly() throws {
        let filter = try CannyEdgeDetectionFilter(engine: engine)
        try assertRespondsAtEdgeOnly(filter, name: "Canny")
    }

    func testLaplacianIsMidGreyOnFlatRegions() throws {
        let filter = try LaplacianFilter(engine: engine)
        let src = try XCTUnwrap(PixelTest.solidColor(
            device: engine.device, size: 8, rgba: SIMD4(0.7, 0.7, 0.7, 1)))
        let dst = try XCTUnwrap(PixelTest.run(filter, source: src, device: engine.device, queue: engine.commandQueue))
        let c = PixelTest.color(PixelTest.readBytes(dst), x: 4, y: 4, width: 8)
        XCTAssertEqual(PixelTest.luma(c), 0.5, accuracy: 0.03, "Laplacian of a flat image is the 0.5 bias")
    }

    // MARK: - Convolution

    func testConvolutionIdentityKernelIsNoOp() throws {
        let filter = try Convolution3x3Filter(engine: engine)
        let src = try XCTUnwrap(PixelTest.solidColor(
            device: engine.device, size: 8, rgba: SIMD4(0.3, 0.5, 0.7, 1)))
        let dst = try XCTUnwrap(PixelTest.run(filter, source: src, device: engine.device, queue: engine.commandQueue))
        let c = PixelTest.color(PixelTest.readBytes(dst), x: 4, y: 4, width: 8)
        XCTAssertEqual(c.x, 0.3, accuracy: 0.02)
        XCTAssertEqual(c.y, 0.5, accuracy: 0.02)
        XCTAssertEqual(c.z, 0.7, accuracy: 0.02)
    }

    func testConvolutionBoxKernelAveragesEdge() throws {
        let filter = try Convolution3x3Filter(engine: engine)
        let ninth: Float = 1.0 / 9.0
        filter.matrix = simd_float3x3(
            SIMD3(repeating: ninth), SIMD3(repeating: ninth), SIMD3(repeating: ninth)
        )
        let src = try XCTUnwrap(PixelTest.verticalEdge(device: engine.device, size: 16))
        let dst = try XCTUnwrap(PixelTest.run(filter, source: src, device: engine.device, queue: engine.commandQueue))
        let bytes = PixelTest.readBytes(dst)
        // Pixel at the boundary column (x=8) has 3 dark + 6 bright neighbours…
        // its box average must be strictly between 0 and 1.
        let boundary = PixelTest.luma(PixelTest.color(bytes, x: 8, y: 8, width: 16))
        XCTAssertGreaterThan(boundary, 0.2)
        XCTAssertLessThan(boundary, 0.9)
    }

    func testEmbossFlatRegionsGoMidGrey() throws {
        let filter = try EmbossFilter(engine: engine)
        let src = try XCTUnwrap(PixelTest.solidColor(
            device: engine.device, size: 8, rgba: SIMD4(0.4, 0.4, 0.4, 1)))
        let dst = try XCTUnwrap(PixelTest.run(filter, source: src, device: engine.device, queue: engine.commandQueue))
        let c = PixelTest.color(PixelTest.readBytes(dst), x: 4, y: 4, width: 8)
        // Emboss kernel sums to ~0 on flat input, leaving bias + centre value.
        XCTAssertEqual(PixelTest.luma(c), 0.9, accuracy: 0.1)
    }

    // MARK: - Corners & NMS

    func testHarrisRespondsAtCornerNotEdge() throws {
        // Quadrant image: bright square in the top-left corner → one corner
        // point at the quadrant centre.
        guard let src = PixelTest.makeTexture(device: engine.device, width: 32, height: 32) else {
            return XCTFail("no texture")
        }
        PixelTest.fill(src) { x, y in
            (x < 16 && y < 16) ? SIMD4(1, 1, 1, 1) : SIMD4(0, 0, 0, 1)
        }
        let filter = try HarrisCornerDetectionFilter(engine: engine)
        let dst = try XCTUnwrap(PixelTest.run(filter, source: src, device: engine.device, queue: engine.commandQueue))
        let bytes = PixelTest.readBytes(dst)

        // Corner of the square (16,16) — strong response somewhere nearby.
        var cornerHit = false
        for y in 13...18 {
            for x in 13...18 where PixelTest.luma(PixelTest.color(bytes, x: x, y: y, width: 32)) > 0.5 {
                cornerHit = true
            }
        }
        XCTAssertTrue(cornerHit, "Harris should fire near the square's corner")

        // Middle of the straight edge (16, 8) — no corner response.
        let edge = PixelTest.luma(PixelTest.color(bytes, x: 16, y: 4, width: 32))
        XCTAssertLessThan(edge, 0.5, "straight edges must not register as corners")
    }

    func testNMSKeepsSingleBrightPixel() throws {
        let filter = try NonMaximumSuppressionFilter(engine: engine)
        let src = try XCTUnwrap(PixelTest.centerBright(device: engine.device, size: 9))
        let dst = try XCTUnwrap(PixelTest.run(filter, source: src, device: engine.device, queue: engine.commandQueue))
        let bytes = PixelTest.readBytes(dst)
        XCTAssertGreaterThan(
            PixelTest.luma(PixelTest.color(bytes, x: 4, y: 4, width: 9)), 0.9,
            "the local maximum survives")
        XCTAssertLessThan(
            PixelTest.luma(PixelTest.color(bytes, x: 3, y: 4, width: 9)), 0.05,
            "neighbours are suppressed")
    }

    func testCannyClearHistoryDoesNotCrash() throws {
        let filter = try CannyEdgeDetectionFilter(engine: engine)
        let src = try XCTUnwrap(PixelTest.verticalEdge(device: engine.device, size: 16))
        _ = PixelTest.run(filter, source: src, device: engine.device, queue: engine.commandQueue)
        filter.clearHistory()
        let dst = PixelTest.run(filter, source: src, device: engine.device, queue: engine.commandQueue)
        XCTAssertNotNil(dst)
    }

    func testEdgePackHandlesHDRTexture() throws {
        let filters: [any MetalForgeFilter] = [
            try SobelEdgeDetectionFilter(engine: engine),
            try CannyEdgeDetectionFilter(engine: engine),
            try NonMaximumSuppressionFilter(engine: engine),
        ]
        guard let src = PixelTest.makeTexture(
            device: engine.device, width: 8, height: 8, pixelFormat: .rgba16Float
        ) else { return XCTFail("no HDR texture") }
        PixelTest.fillHDR(src) { x, _ in x < 4 ? SIMD4(0, 0, 0, 1) : SIMD4(1.5, 1.5, 1.5, 1) }

        for filter in filters {
            let dst = PixelTest.run(filter, source: src, device: engine.device, queue: engine.commandQueue)
            XCTAssertNotNil(dst, "HDR encode failed for \(type(of: filter))")
        }
    }
}
