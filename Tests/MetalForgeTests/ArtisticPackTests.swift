import XCTest
import Metal
import simd
@testable import MetalForge

// ===========================================================================
// v0.6.0 Artistic Effects Pack — behaviour tests.
// ===========================================================================

final class ArtisticPackTests: XCTestCase {

    private var engine: MetalForgeEngine!

    override func setUpWithError() throws {
        engine = try MetalForgeEngine()
    }

    func testArtisticFiltersAreConstructible() throws {
        XCTAssertNoThrow(try ToonFilter(engine: engine))
        XCTAssertNoThrow(try SmoothToonFilter(engine: engine))
        XCTAssertNoThrow(try SketchFilter(engine: engine))
        XCTAssertNoThrow(try ThresholdSketchFilter(engine: engine))
        XCTAssertNoThrow(try CrosshatchFilter(engine: engine))
        XCTAssertNoThrow(try HalftoneFilter(engine: engine))
        XCTAssertNoThrow(try PolkaDotFilter(engine: engine))
        XCTAssertNoThrow(try KuwaharaFilter(engine: engine))
        XCTAssertNoThrow(try PixellateFilter(engine: engine))
        XCTAssertNoThrow(try PolarPixellateFilter(engine: engine))
        XCTAssertNoThrow(try MosaicFilter(engine: engine))
        XCTAssertNoThrow(try CGAColorspaceFilter(engine: engine))
    }

    // MARK: - Sketches

    func testSketchIsWhiteOnFlatAndDarkAtEdges() throws {
        let filter = try SketchFilter(engine: engine)
        let src = try XCTUnwrap(PixelTest.verticalEdge(device: engine.device, size: 16))
        let dst = try XCTUnwrap(PixelTest.run(filter, source: src, device: engine.device, queue: engine.commandQueue))
        let bytes = PixelTest.readBytes(dst)
        XCTAssertGreaterThan(PixelTest.luma(PixelTest.color(bytes, x: 3, y: 8, width: 16)), 0.9,
                             "flat paper stays white")
        let edge = min(
            PixelTest.luma(PixelTest.color(bytes, x: 7, y: 8, width: 16)),
            PixelTest.luma(PixelTest.color(bytes, x: 8, y: 8, width: 16)))
        XCTAssertLessThan(edge, 0.5, "strokes darken the boundary")
    }

    func testThresholdSketchIsBinary() throws {
        let filter = try ThresholdSketchFilter(engine: engine)
        let src = try XCTUnwrap(PixelTest.verticalEdge(device: engine.device, size: 16))
        let dst = try XCTUnwrap(PixelTest.run(filter, source: src, device: engine.device, queue: engine.commandQueue))
        let bytes = PixelTest.readBytes(dst)
        for x in [2, 7, 13] {
            let l = PixelTest.luma(PixelTest.color(bytes, x: x, y: 8, width: 16))
            XCTAssertTrue(l < 0.05 || l > 0.95, "threshold sketch output must be binary, got \(l)")
        }
    }

    // MARK: - Toon

    func testToonQuantisesFlatColour() throws {
        let filter = try ToonFilter(engine: engine)
        filter.quantizationLevels = 4
        let src = try XCTUnwrap(PixelTest.solidColor(
            device: engine.device, size: 8, rgba: SIMD4(0.61, 0.61, 0.61, 1)))
        let dst = try XCTUnwrap(PixelTest.run(filter, source: src, device: engine.device, queue: engine.commandQueue))
        let c = PixelTest.color(PixelTest.readBytes(dst), x: 4, y: 4, width: 8)
        // 0.61 quantised to 4 levels → round(0.61*4)/4 = 0.5.
        XCTAssertEqual(c.x, 0.5, accuracy: 0.03)
    }

    func testSmoothToonProcessesEdgeTexture() throws {
        let filter = try SmoothToonFilter(engine: engine)
        let src = try XCTUnwrap(PixelTest.verticalEdge(device: engine.device, size: 16))
        let dst = PixelTest.run(filter, source: src, device: engine.device, queue: engine.commandQueue)
        XCTAssertNotNil(dst)
    }

    // MARK: - Dots & hatching

    func testCrosshatchDarkAreasGetStrokes() throws {
        let filter = try CrosshatchFilter(engine: engine)
        let src = try XCTUnwrap(PixelTest.solidColor(
            device: engine.device, size: 32, rgba: SIMD4(0.1, 0.1, 0.1, 1)))
        let dst = try XCTUnwrap(PixelTest.run(filter, source: src, device: engine.device, queue: engine.commandQueue))
        let bytes = PixelTest.readBytes(dst)
        var dark = 0
        for y in 0..<32 {
            for x in 0..<32 where PixelTest.luma(PixelTest.color(bytes, x: x, y: y, width: 32)) < 0.5 {
                dark += 1
            }
        }
        XCTAssertGreaterThan(dark, 200, "dark input should be heavily hatched")
    }

    func testHalftoneDarkInputProducesInkDots() throws {
        let filter = try HalftoneFilter(engine: engine)
        filter.dotSize = 8
        let src = try XCTUnwrap(PixelTest.solidColor(
            device: engine.device, size: 32, rgba: SIMD4(0.15, 0.15, 0.15, 1)))
        let dst = try XCTUnwrap(PixelTest.run(filter, source: src, device: engine.device, queue: engine.commandQueue))
        let bytes = PixelTest.readBytes(dst)
        var minL: Float = 1, maxL: Float = 0
        for y in 0..<32 {
            for x in 0..<32 {
                let l = PixelTest.luma(PixelTest.color(bytes, x: x, y: y, width: 32))
                minL = min(minL, l); maxL = max(maxL, l)
            }
        }
        XCTAssertLessThan(minL, 0.1, "ink dots present")
        XCTAssertGreaterThan(maxL, 0.9, "paper between dots present")
    }

    func testPolkaDotKeepsCellColourInsideDots() throws {
        let filter = try PolkaDotFilter(engine: engine)
        filter.dotSize = 8
        let src = try XCTUnwrap(PixelTest.solidColor(
            device: engine.device, size: 32, rgba: SIMD4(1, 0, 0, 1)))
        let dst = try XCTUnwrap(PixelTest.run(filter, source: src, device: engine.device, queue: engine.commandQueue))
        let bytes = PixelTest.readBytes(dst)
        // Cell centre (4,4) is inside a dot → red. Cell corner (0,0) → black.
        let inDot  = PixelTest.color(bytes, x: 4, y: 4, width: 32)
        let corner = PixelTest.color(bytes, x: 0, y: 0, width: 32)
        XCTAssertGreaterThan(inDot.x, 0.9)
        XCTAssertLessThan(PixelTest.luma(corner), 0.1)
    }

    // MARK: - Painterly

    func testKuwaharaPreservesFlatColour() throws {
        let filter = try KuwaharaFilter(engine: engine)
        let src = try XCTUnwrap(PixelTest.solidColor(
            device: engine.device, size: 16, rgba: SIMD4(0.3, 0.6, 0.2, 1)))
        let dst = try XCTUnwrap(PixelTest.run(filter, source: src, device: engine.device, queue: engine.commandQueue))
        let c = PixelTest.color(PixelTest.readBytes(dst), x: 8, y: 8, width: 16)
        XCTAssertEqual(c.x, 0.3, accuracy: 0.03)
        XCTAssertEqual(c.y, 0.6, accuracy: 0.03)
        XCTAssertEqual(c.z, 0.2, accuracy: 0.03)
    }

    func testKuwaharaKeepsEdgeSharp() throws {
        let filter = try KuwaharaFilter(engine: engine)
        filter.radius = 3
        let src = try XCTUnwrap(PixelTest.verticalEdge(device: engine.device, size: 16))
        let dst = try XCTUnwrap(PixelTest.run(filter, source: src, device: engine.device, queue: engine.commandQueue))
        let bytes = PixelTest.readBytes(dst)
        // Kuwahara's whole point: the edge stays a hard step, no smearing.
        XCTAssertLessThan(PixelTest.luma(PixelTest.color(bytes, x: 6, y: 8, width: 16)), 0.2)
        XCTAssertGreaterThan(PixelTest.luma(PixelTest.color(bytes, x: 9, y: 8, width: 16)), 0.8)
    }

    // MARK: - Pixellation family

    func testPixellateMakesCellsUniform() throws {
        let filter = try PixellateFilter(engine: engine)
        filter.cellSize = 8
        let src = try XCTUnwrap(PixelTest.gradient(device: engine.device, size: 32))
        let dst = try XCTUnwrap(PixelTest.run(filter, source: src, device: engine.device, queue: engine.commandQueue))
        let bytes = PixelTest.readBytes(dst)
        // All pixels inside one 8×8 cell share the same value.
        let ref = PixelTest.color(bytes, x: 1, y: 1, width: 32)
        for x in 0..<8 {
            let c = PixelTest.color(bytes, x: x, y: 5, width: 32)
            XCTAssertEqual(c.x, ref.x, accuracy: 0.01, "cell must be flat")
        }
    }

    func testMosaicGroutDarkensTileBorders() throws {
        let filter = try MosaicFilter(engine: engine)
        filter.tileSize = 8
        filter.groutWidth = 1
        let src = try XCTUnwrap(PixelTest.solidColor(
            device: engine.device, size: 32, rgba: SIMD4(0.8, 0.8, 0.8, 1)))
        let dst = try XCTUnwrap(PixelTest.run(filter, source: src, device: engine.device, queue: engine.commandQueue))
        let bytes = PixelTest.readBytes(dst)
        let interior = PixelTest.luma(PixelTest.color(bytes, x: 4, y: 4, width: 32))
        let seam     = PixelTest.luma(PixelTest.color(bytes, x: 8, y: 4, width: 32))
        XCTAssertGreaterThan(interior, 0.7)
        XCTAssertLessThan(seam, interior - 0.2, "grout seams must be visibly darker")
    }

    func testPolarPixellateProcessesGradient() throws {
        let filter = try PolarPixellateFilter(engine: engine)
        let src = try XCTUnwrap(PixelTest.gradient(device: engine.device, size: 32))
        let dst = PixelTest.run(filter, source: src, device: engine.device, queue: engine.commandQueue)
        XCTAssertNotNil(dst)
    }

    // MARK: - CGA

    func testCGAOutputsOnlyPaletteColors() throws {
        let filter = try CGAColorspaceFilter(engine: engine)
        let src = try XCTUnwrap(PixelTest.gradient(device: engine.device, size: 16))
        let dst = try XCTUnwrap(PixelTest.run(filter, source: src, device: engine.device, queue: engine.commandQueue))
        let bytes = PixelTest.readBytes(dst)
        let palette: [SIMD3<Float>] = [
            SIMD3(0, 0, 0), SIMD3(0, 1, 1), SIMD3(1, 0, 1), SIMD3(1, 1, 1),
        ]
        for y in stride(from: 0, to: 16, by: 4) {
            for x in stride(from: 0, to: 16, by: 4) {
                let c = PixelTest.color(bytes, x: x, y: y, width: 16)
                let rgb = SIMD3(c.x, c.y, c.z)
                let isPalette = palette.contains { simd_distance($0, rgb) < 0.05 }
                XCTAssertTrue(isPalette, "non-palette colour \(rgb) at (\(x),\(y))")
            }
        }
    }

    func testArtisticPackHandlesHDRTexture() throws {
        let filters: [any MetalForgeFilter] = [
            try ToonFilter(engine: engine),
            try KuwaharaFilter(engine: engine),
            try PixellateFilter(engine: engine),
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
