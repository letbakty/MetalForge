import XCTest
import Metal
import simd
@testable import MetalForge

// ===========================================================================
// v0.3.0 Distortion Pack — constructibility + pixel-behaviour tests.
// ===========================================================================

final class DistortionPackTests: XCTestCase {

    private var engine: MetalForgeEngine!

    override func setUpWithError() throws {
        engine = try MetalForgeEngine()
    }

    func testDistortionFiltersAreConstructible() throws {
        XCTAssertNoThrow(try BulgeDistortionFilter(engine: engine))
        XCTAssertNoThrow(try PinchDistortionFilter(engine: engine))
        XCTAssertNoThrow(try StretchDistortionFilter(engine: engine))
        XCTAssertNoThrow(try SwirlFilter(engine: engine))
        XCTAssertNoThrow(try SphereRefractionFilter(engine: engine))
        XCTAssertNoThrow(try GlassSphereFilter(engine: engine))
        XCTAssertNoThrow(try CropFilter(engine: engine))
        XCTAssertNoThrow(try TransformFilter(engine: engine))
    }

    // MARK: - Identity behaviour

    func testBulgeZeroScaleIsIdentity() throws {
        let filter = try BulgeDistortionFilter(engine: engine)
        filter.scale = 0
        let src = try XCTUnwrap(PixelTest.verticalEdge(device: engine.device, size: 16))
        let dst = try XCTUnwrap(PixelTest.run(filter, source: src, device: engine.device, queue: engine.commandQueue))
        let bytes = PixelTest.readBytes(dst)
        XCTAssertEqual(PixelTest.luma(PixelTest.color(bytes, x: 2,  y: 8, width: 16)), 0, accuracy: 0.05)
        XCTAssertEqual(PixelTest.luma(PixelTest.color(bytes, x: 13, y: 8, width: 16)), 1, accuracy: 0.05)
    }

    func testCropIdentityLeavesImageUnchanged() throws {
        let filter = try CropFilter(engine: engine)
        let src = try XCTUnwrap(PixelTest.verticalEdge(device: engine.device, size: 16))
        let dst = try XCTUnwrap(PixelTest.run(filter, source: src, device: engine.device, queue: engine.commandQueue))
        let bytes = PixelTest.readBytes(dst)
        XCTAssertEqual(PixelTest.luma(PixelTest.color(bytes, x: 2,  y: 8, width: 16)), 0, accuracy: 0.05)
        XCTAssertEqual(PixelTest.luma(PixelTest.color(bytes, x: 13, y: 8, width: 16)), 1, accuracy: 0.05)
    }

    func testTransformIdentityLeavesImageUnchanged() throws {
        let filter = try TransformFilter(engine: engine)
        let src = try XCTUnwrap(PixelTest.verticalEdge(device: engine.device, size: 16))
        let dst = try XCTUnwrap(PixelTest.run(filter, source: src, device: engine.device, queue: engine.commandQueue))
        let bytes = PixelTest.readBytes(dst)
        XCTAssertEqual(PixelTest.luma(PixelTest.color(bytes, x: 2,  y: 8, width: 16)), 0, accuracy: 0.05)
        XCTAssertEqual(PixelTest.luma(PixelTest.color(bytes, x: 13, y: 8, width: 16)), 1, accuracy: 0.05)
    }

    // MARK: - Effect behaviour

    func testCropZoomsIntoRegion() throws {
        let filter = try CropFilter(engine: engine)
        // Crop the right (bright) half — the whole output should be bright.
        filter.origin = SIMD2(0.5, 0)
        filter.size   = SIMD2(0.5, 1)
        let src = try XCTUnwrap(PixelTest.verticalEdge(device: engine.device, size: 16))
        let dst = try XCTUnwrap(PixelTest.run(filter, source: src, device: engine.device, queue: engine.commandQueue))
        let bytes = PixelTest.readBytes(dst)
        XCTAssertEqual(PixelTest.luma(PixelTest.color(bytes, x: 2, y: 8, width: 16)), 1, accuracy: 0.05,
                       "left edge of the output should now show the bright half")
    }

    func testTransformRotation180FlipsEdge() throws {
        let filter = try TransformFilter(engine: engine)
        filter.rotation = .pi
        let src = try XCTUnwrap(PixelTest.verticalEdge(device: engine.device, size: 16))
        let dst = try XCTUnwrap(PixelTest.run(filter, source: src, device: engine.device, queue: engine.commandQueue))
        let bytes = PixelTest.readBytes(dst)
        // After 180° rotation the dark half is on the right.
        XCTAssertEqual(PixelTest.luma(PixelTest.color(bytes, x: 13, y: 8, width: 16)), 0, accuracy: 0.05)
        XCTAssertEqual(PixelTest.luma(PixelTest.color(bytes, x: 2,  y: 8, width: 16)), 1, accuracy: 0.05)
    }

    func testSwirlMovesPixelsNearCentre() throws {
        let filter = try SwirlFilter(engine: engine)
        filter.angle = 2.0
        filter.radius = 0.8
        let src = try XCTUnwrap(PixelTest.verticalEdge(device: engine.device, size: 32))
        let dst = try XCTUnwrap(PixelTest.run(filter, source: src, device: engine.device, queue: engine.commandQueue))
        let srcBytes = PixelTest.readBytes(src)
        let dstBytes = PixelTest.readBytes(dst)
        // Somewhere near the centre the twist must change pixel values.
        var changed = 0
        for y in 10..<22 {
            for x in 10..<22 {
                let a = PixelTest.luma(PixelTest.color(srcBytes, x: x, y: y, width: 32))
                let b = PixelTest.luma(PixelTest.color(dstBytes, x: x, y: y, width: 32))
                if abs(a - b) > 0.3 { changed += 1 }
            }
        }
        XCTAssertGreaterThan(changed, 5, "swirl should visibly move pixels near the centre")
    }

    func testSphereRefractionBlanksOutsideSphere() throws {
        let filter = try SphereRefractionFilter(engine: engine)
        filter.radius = 0.2
        let src = try XCTUnwrap(PixelTest.solidColor(
            device: engine.device, size: 32, rgba: SIMD4(1, 1, 1, 1)))
        let dst = try XCTUnwrap(PixelTest.run(filter, source: src, device: engine.device, queue: engine.commandQueue))
        let bytes = PixelTest.readBytes(dst)
        let corner = PixelTest.color(bytes, x: 1, y: 1, width: 32)
        let centre = PixelTest.color(bytes, x: 16, y: 16, width: 32)
        XCTAssertEqual(PixelTest.luma(corner), 0, accuracy: 0.02, "outside the sphere is black")
        XCTAssertGreaterThan(PixelTest.luma(centre), 0.9, "inside the sphere shows the (white) scene")
    }

    func testGlassSphereKeepsSceneOutside() throws {
        let filter = try GlassSphereFilter(engine: engine)
        filter.radius = 0.2
        let src = try XCTUnwrap(PixelTest.solidColor(
            device: engine.device, size: 32, rgba: SIMD4(0.6, 0.6, 0.6, 1)))
        let dst = try XCTUnwrap(PixelTest.run(filter, source: src, device: engine.device, queue: engine.commandQueue))
        let bytes = PixelTest.readBytes(dst)
        let corner = PixelTest.color(bytes, x: 1, y: 1, width: 32)
        XCTAssertEqual(PixelTest.luma(corner), 0.6, accuracy: 0.05, "outside the glass sphere the scene is untouched")
    }

    func testPinchDistortsEdgeRegion() throws {
        let filter = try PinchDistortionFilter(engine: engine)
        filter.scale = 1.0
        filter.radius = 1.0
        // Off-centre: a pinch centred exactly on the edge boundary would leave
        // the half-plane partition invariant and change nothing.
        filter.center = SIMD2(0.25, 0.5)
        let src = try XCTUnwrap(PixelTest.verticalEdge(device: engine.device, size: 32))
        let dst = try XCTUnwrap(PixelTest.run(filter, source: src, device: engine.device, queue: engine.commandQueue))
        let srcBytes = PixelTest.readBytes(src)
        let dstBytes = PixelTest.readBytes(dst)
        var changed = 0
        for y in 0..<32 {
            for x in 0..<32 {
                let a = PixelTest.luma(PixelTest.color(srcBytes, x: x, y: y, width: 32))
                let b = PixelTest.luma(PixelTest.color(dstBytes, x: x, y: y, width: 32))
                if abs(a - b) > 0.3 { changed += 1 }
            }
        }
        XCTAssertGreaterThan(changed, 10, "pinch should displace the edge boundary")
    }

    func testStretchProcessesWithoutArtifactsAtIdentityCentre() throws {
        let filter = try StretchDistortionFilter(engine: engine)
        let src = try XCTUnwrap(PixelTest.solidColor(
            device: engine.device, size: 16, rgba: SIMD4(0.4, 0.5, 0.6, 1)))
        let dst = try XCTUnwrap(PixelTest.run(filter, source: src, device: engine.device, queue: engine.commandQueue))
        let c = PixelTest.color(PixelTest.readBytes(dst), x: 8, y: 8, width: 16)
        // A flat image must stay flat under any pure UV remap.
        XCTAssertEqual(c.x, 0.4, accuracy: 0.03)
        XCTAssertEqual(c.y, 0.5, accuracy: 0.03)
        XCTAssertEqual(c.z, 0.6, accuracy: 0.03)
    }

    func testDistortionHandlesHDRTexture() throws {
        let filters: [any MetalForgeFilter] = [
            try BulgeDistortionFilter(engine: engine),
            try SwirlFilter(engine: engine),
            try TransformFilter(engine: engine),
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
