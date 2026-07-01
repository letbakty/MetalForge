import XCTest
import Metal
import simd
@testable import MetalForge

// ===========================================================================
// v0.8.0 Temporal Video Effects Pack — multi-frame behaviour tests.
// ===========================================================================

final class TemporalPackTests: XCTestCase {

    private var engine: MetalForgeEngine!

    override func setUpWithError() throws {
        engine = try MetalForgeEngine()
    }

    /// Run a temporal filter over a sequence of frames, returning each output.
    private func runSequence(
        _ filter: any MetalForgeFilter,
        frames: [MTLTexture]
    ) -> [MTLTexture] {
        var results: [MTLTexture] = []
        for frame in frames {
            guard let out = PixelTest.run(
                filter, source: frame, device: engine.device, queue: engine.commandQueue
            ) else { continue }
            results.append(out)
        }
        return results
    }

    func testTemporalFiltersAreConstructible() throws {
        XCTAssertNoThrow(try FrameBlendFilter(engine: engine))
        XCTAssertNoThrow(try LowPassFilter(engine: engine))
        XCTAssertNoThrow(try HighPassFilter(engine: engine))
        XCTAssertNoThrow(try MotionDetectorFilter(engine: engine))
        XCTAssertNoThrow(try OpticalFlowWarpFilter(engine: engine))
        XCTAssertNoThrow(try FrameInterpolationFilter(engine: engine))
    }

    // MARK: - FrameBlend

    func testFrameBlendLeavesTrailAfterBrightFrame() throws {
        let filter = try FrameBlendFilter(engine: engine)
        filter.decay = 0.9
        let bright = try XCTUnwrap(PixelTest.solidColor(device: engine.device, size: 8, rgba: SIMD4(1, 1, 1, 1)))
        let dark   = try XCTUnwrap(PixelTest.solidColor(device: engine.device, size: 8, rgba: SIMD4(0, 0, 0, 1)))
        let outs = runSequence(filter, frames: [bright, dark])
        XCTAssertEqual(outs.count, 2)
        let trail = PixelTest.color(PixelTest.readBytes(outs[1]), x: 4, y: 4, width: 8)
        XCTAssertEqual(PixelTest.luma(trail), 0.9, accuracy: 0.05,
                       "one frame after white, the trail should be ≈decay")
    }

    // MARK: - LowPass / HighPass

    func testLowPassSmoothsStepChange() throws {
        let filter = try LowPassFilter(engine: engine)
        filter.strength = 0.5
        let dark   = try XCTUnwrap(PixelTest.solidColor(device: engine.device, size: 8, rgba: SIMD4(0, 0, 0, 1)))
        let bright = try XCTUnwrap(PixelTest.solidColor(device: engine.device, size: 8, rgba: SIMD4(1, 1, 1, 1)))
        let outs = runSequence(filter, frames: [dark, bright])
        let c = PixelTest.color(PixelTest.readBytes(outs[1]), x: 4, y: 4, width: 8)
        XCTAssertEqual(PixelTest.luma(c), 0.5, accuracy: 0.05,
                       "EMA with strength 0.5 should land halfway after a 0→1 step")
    }

    func testHighPassShowsOnlyChange() throws {
        let filter = try HighPassFilter(engine: engine)
        filter.strength = 0.5
        let grey = try XCTUnwrap(PixelTest.solidColor(device: engine.device, size: 8, rgba: SIMD4(0.5, 0.5, 0.5, 1)))
        // Static frames → residual ≈ 0.
        let outs = runSequence(filter, frames: [grey, grey, grey])
        let still = PixelTest.color(PixelTest.readBytes(outs[2]), x: 4, y: 4, width: 8)
        XCTAssertLessThan(PixelTest.luma(still), 0.05, "static scene must go black in high-pass")

        // A sudden change → strong residual.
        let bright = try XCTUnwrap(PixelTest.solidColor(device: engine.device, size: 8, rgba: SIMD4(1, 1, 1, 1)))
        let changed = try XCTUnwrap(PixelTest.run(filter, source: bright, device: engine.device, queue: engine.commandQueue))
        let c = PixelTest.color(PixelTest.readBytes(changed), x: 4, y: 4, width: 8)
        XCTAssertGreaterThan(PixelTest.luma(c), 0.15, "a step change must produce residual energy")
    }

    // MARK: - MotionDetector

    func testMotionDetectorHighlightsChangedRegion() throws {
        let filter = try MotionDetectorFilter(engine: engine)
        filter.highlightColor = SIMD3(1, 0, 0)
        filter.threshold = 0.1
        let dark   = try XCTUnwrap(PixelTest.solidColor(device: engine.device, size: 8, rgba: SIMD4(0, 0, 0, 1)))
        let bright = try XCTUnwrap(PixelTest.solidColor(device: engine.device, size: 8, rgba: SIMD4(1, 1, 1, 1)))
        let outs = runSequence(filter, frames: [dark, bright])
        let c = PixelTest.color(PixelTest.readBytes(outs[1]), x: 4, y: 4, width: 8)
        XCTAssertGreaterThan(c.x, 0.8, "changed pixels are painted with the highlight colour")
        XCTAssertLessThan(c.y, 0.3)
    }

    func testMotionDetectorStaysCalmOnStaticScene() throws {
        let filter = try MotionDetectorFilter(engine: engine)
        filter.dimming = 0.4
        let grey = try XCTUnwrap(PixelTest.solidColor(device: engine.device, size: 8, rgba: SIMD4(0.5, 0.5, 0.5, 1)))
        let outs = runSequence(filter, frames: [grey, grey])
        let c = PixelTest.color(PixelTest.readBytes(outs[1]), x: 4, y: 4, width: 8)
        XCTAssertEqual(PixelTest.luma(c), 0.2, accuracy: 0.05,
                       "static scene shows as dimmed original (0.5 × 0.4)")
    }

    // MARK: - FrameInterpolation

    func testFrameInterpolationPhaseBlendsFrames() throws {
        let filter = try FrameInterpolationFilter(engine: engine)
        filter.phase = 0.25
        let dark   = try XCTUnwrap(PixelTest.solidColor(device: engine.device, size: 8, rgba: SIMD4(0, 0, 0, 1)))
        let bright = try XCTUnwrap(PixelTest.solidColor(device: engine.device, size: 8, rgba: SIMD4(1, 1, 1, 1)))
        let outs = runSequence(filter, frames: [dark, bright])
        let c = PixelTest.color(PixelTest.readBytes(outs[1]), x: 4, y: 4, width: 8)
        XCTAssertEqual(PixelTest.luma(c), 0.25, accuracy: 0.05,
                       "phase 0.25 should sit a quarter of the way from prev to current")
    }

    // MARK: - OpticalFlowWarp

    func testOpticalFlowWarpIsStableOnStaticScene() throws {
        let filter = try OpticalFlowWarpFilter(engine: engine)
        let src = try XCTUnwrap(PixelTest.verticalEdge(device: engine.device, size: 16))
        let outs = runSequence(filter, frames: [src, src])
        // No motion → zero flow → output equals input.
        let bytes = PixelTest.readBytes(outs[1])
        XCTAssertEqual(PixelTest.luma(PixelTest.color(bytes, x: 2,  y: 8, width: 16)), 0, accuracy: 0.05)
        XCTAssertEqual(PixelTest.luma(PixelTest.color(bytes, x: 13, y: 8, width: 16)), 1, accuracy: 0.05)
    }

    func testOpticalFlowWarpRespondsToMotion() throws {
        let filter = try OpticalFlowWarpFilter(engine: engine)
        filter.strength = 8
        // Edge moves 2px right between frames.
        guard
            let f0 = PixelTest.makeTexture(device: engine.device, width: 16, height: 16),
            let f1 = PixelTest.makeTexture(device: engine.device, width: 16, height: 16)
        else { return XCTFail("no textures") }
        PixelTest.fill(f0) { x, _ in x < 8  ? SIMD4(0, 0, 0, 1) : SIMD4(1, 1, 1, 1) }
        PixelTest.fill(f1) { x, _ in x < 10 ? SIMD4(0, 0, 0, 1) : SIMD4(1, 1, 1, 1) }

        let outs = runSequence(filter, frames: [f0, f1])
        let srcBytes = PixelTest.readBytes(f1)
        let dstBytes = PixelTest.readBytes(outs[1])
        var changed = 0
        for y in 0..<16 {
            for x in 0..<16 {
                let a = PixelTest.luma(PixelTest.color(srcBytes, x: x, y: y, width: 16))
                let b = PixelTest.luma(PixelTest.color(dstBytes, x: x, y: y, width: 16))
                if abs(a - b) > 0.3 { changed += 1 }
            }
        }
        XCTAssertGreaterThan(changed, 3, "moving edge must trigger a visible warp")
    }

    // MARK: - Lifecycle

    func testClearHistoryResetsAccumulation() throws {
        let filter = try FrameBlendFilter(engine: engine)
        let bright = try XCTUnwrap(PixelTest.solidColor(device: engine.device, size: 8, rgba: SIMD4(1, 1, 1, 1)))
        let dark   = try XCTUnwrap(PixelTest.solidColor(device: engine.device, size: 8, rgba: SIMD4(0, 0, 0, 1)))
        _ = runSequence(filter, frames: [bright])
        filter.clearHistory()
        // After reset, the first dark frame seeds dark history → no trail.
        let out = try XCTUnwrap(PixelTest.run(filter, source: dark, device: engine.device, queue: engine.commandQueue))
        let c = PixelTest.color(PixelTest.readBytes(out), x: 4, y: 4, width: 8)
        XCTAssertLessThan(PixelTest.luma(c), 0.05, "clearHistory must drop the white trail")
    }

    func testTemporalPackHandlesHDRTexture() throws {
        let filters: [any MetalForgeFilter] = [
            try FrameBlendFilter(engine: engine),
            try LowPassFilter(engine: engine),
            try HighPassFilter(engine: engine),
            try OpticalFlowWarpFilter(engine: engine),
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
