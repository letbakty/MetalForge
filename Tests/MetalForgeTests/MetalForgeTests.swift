import XCTest
import Metal
@testable import MetalForge

final class MetalForgeEngineTests: XCTestCase {

    // Most GPU tests are guarded — CI machines may not have a Metal device.
    private var engine: MetalForgeEngine?

    override func setUp() async throws {
        engine = try? MetalForgeEngine()
    }

    func testEngineInitialises() throws {
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("No Metal device available in this environment.")
        }
        XCTAssertNoThrow(try MetalForgeEngine())
    }

    func testTextureCacheFlushDoesNotCrash() throws {
        guard let engine else { throw XCTSkip("No Metal device.") }
        // Flushing an empty cache must not crash or assert.
        engine.flushTextureCache()
    }

    func testTexturePoolAcquireRecycle() throws {
        guard let engine else { throw XCTSkip("No Metal device.") }
        let pool = TexturePool(device: engine.device)

        let tex1 = pool.acquire(width: 256, height: 256, pixelFormat: .bgra8Unorm)
        XCTAssertNotNil(tex1)

        if let tex1 {
            pool.recycle(tex1)
            // After recycling, the next acquire of the same spec should return the
            // same MTLTexture object (pointer equality).
            let tex2 = pool.acquire(width: 256, height: 256, pixelFormat: .bgra8Unorm)
            XCTAssertTrue(tex1 === tex2, "Pool should return the recycled texture.")
        }
    }

    func testPipelineIsConstructible() throws {
        guard let engine else { throw XCTSkip("No Metal device.") }
        // Now throws because it eagerly compiles the embedded YUVToRGBConverter PSO.
        XCTAssertNoThrow(try MetalForgePipeline(engine: engine))
    }

    func testPipelineProcessesYUV8BitPixelBuffer() throws {
        guard let engine else { throw XCTSkip("No Metal device.") }
        let pipeline = try MetalForgePipeline(engine: engine)

        // Build an empty 8-bit YUV bi-planar buffer marked Metal-compatible.
        var pb: CVPixelBuffer?
        let attrs: [CFString: Any] = [
            kCVPixelBufferMetalCompatibilityKey: true,
            kCVPixelBufferIOSurfacePropertiesKey: [:] as CFDictionary
        ]
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault, 128, 128,
            kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
            attrs as CFDictionary, &pb
        )
        XCTAssertEqual(status, kCVReturnSuccess)
        guard let buffer = pb else { return }

        // No downstream filters: result is the YUV-converter output.
        let result = pipeline.process(pixelBuffer: buffer)
        XCTAssertNotNil(result)
        XCTAssertEqual(result?.pixelFormat, .bgra8Unorm)
        if let result { pipeline.recycle(result) }
    }

    func testPipelineProcessesYUV10BitPixelBufferAsHDR() throws {
        guard let engine else { throw XCTSkip("No Metal device.") }
        let pipeline = try MetalForgePipeline(engine: engine)

        var pb: CVPixelBuffer?
        let attrs: [CFString: Any] = [
            kCVPixelBufferMetalCompatibilityKey: true,
            kCVPixelBufferIOSurfacePropertiesKey: [:] as CFDictionary
        ]
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault, 128, 128,
            kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange,
            attrs as CFDictionary, &pb
        )
        XCTAssertEqual(status, kCVReturnSuccess)
        guard let buffer = pb else { return }

        let result = pipeline.process(pixelBuffer: buffer)
        XCTAssertNotNil(result)
        // 10-bit YUV should auto-route through the HDR working space.
        XCTAssertEqual(result?.pixelFormat, .rgba16Float)
        if let result { pipeline.recycle(result) }
    }

    func testAdjustmentFilterIdentityIsConstructible() throws {
        guard let engine else { throw XCTSkip("No Metal device.") }
        // This will fail in CI if the .metallib isn't bundled — that's intentional.
        // Run locally with `swift test` in an environment with Metal support.
        XCTAssertNoThrow(try AdjustmentFilter(engine: engine))
    }

    func testGlitchFilterIsConstructible() throws {
        guard let engine else { throw XCTSkip("No Metal device.") }
        XCTAssertNoThrow(try GlitchFilter(engine: engine))
    }

    // MARK: - YUV converter

    func testYUVToRGBConverterIsConstructible() throws {
        guard let engine else { throw XCTSkip("No Metal device.") }
        XCTAssertNoThrow(try YUVToRGBConverter(engine: engine))
    }

    func testMatrixSelectionFallsBackToBT709() {
        // A synthetic CVPixelBuffer with no colour attachments should fall back
        // to BT.709 video range (the standard SDR default).
        var pb: CVPixelBuffer?
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault, 64, 64,
            kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
            nil, &pb
        )
        XCTAssertEqual(status, kCVReturnSuccess)
        guard let buffer = pb else { return }

        let result = YUVColorMatrices.matrix(for: buffer)
        XCTAssertFalse(result.isFullRange)
        XCTAssertEqual(result.matrix.columns.0.x, 1.16438, accuracy: 0.0001)
    }

    func testMatrixSelectionDetectsFullRange() {
        var pb: CVPixelBuffer?
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault, 64, 64,
            kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
            nil, &pb
        )
        XCTAssertEqual(status, kCVReturnSuccess)
        guard let buffer = pb else { return }

        let result = YUVColorMatrices.matrix(for: buffer)
        XCTAssertTrue(result.isFullRange)
        // Full-range BT.709 has Y coefficient = 1.0 (no range expansion).
        XCTAssertEqual(result.matrix.columns.0.x, 1.0, accuracy: 0.0001)
    }

    func testColorSpaceIsHDRReport() {
        XCTAssertFalse(MetalForgeColorSpace.sdr.isHDR)
        XCTAssertTrue(MetalForgeColorSpace.hdr10PQ.isHDR)
        XCTAssertTrue(MetalForgeColorSpace.hlg.isHDR)
    }

    // MARK: - HDR transfer

    func testHDRDecodeFilterIsConstructible() throws {
        guard let engine else { throw XCTSkip("No Metal device.") }
        XCTAssertNoThrow(try HDRDecodeFilter(engine: engine))
    }

    func testHDREncodeFilterIsConstructible() throws {
        guard let engine else { throw XCTSkip("No Metal device.") }
        XCTAssertNoThrow(try HDREncodeFilter(engine: engine))
    }

    func testColorSpaceDetectionDefaultsTo10BitAsPQ() {
        var pb: CVPixelBuffer?
        let attrs: [CFString: Any] = [
            kCVPixelBufferMetalCompatibilityKey: true,
            kCVPixelBufferIOSurfacePropertiesKey: [:] as CFDictionary
        ]
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault, 64, 64,
            kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange,
            attrs as CFDictionary, &pb
        )
        XCTAssertEqual(status, kCVReturnSuccess)
        guard let buffer = pb else { return }

        // No transfer function attachment → PQ default.
        XCTAssertEqual(MetalForgeColorSpace.detect(from: buffer), .hdr10PQ)
    }

    // MARK: - MetalForgeView

    @MainActor
    func testMetalForgeViewIsConstructible() throws {
        guard let engine else { throw XCTSkip("No Metal device.") }
        let view = try MetalForgeView(engine: engine, frame: CGRect(x: 0, y: 0, width: 256, height: 256))
        XCTAssertEqual(view.colorPixelFormat, .bgra8Unorm)
        XCTAssertTrue(view.isPaused)
        XCTAssertFalse(view.enableSetNeedsDisplay)
    }

    @MainActor
    func testMetalForgeViewSwitchesColorSpaceForHDR() throws {
        guard let engine else { throw XCTSkip("No Metal device.") }
        let view = try MetalForgeView(engine: engine, frame: CGRect(x: 0, y: 0, width: 256, height: 256))

        view.workingColorSpace = .hdr10PQ
        XCTAssertEqual(view.colorPixelFormat, .bgr10a2Unorm)

        view.workingColorSpace = .hlg
        XCTAssertEqual(view.colorPixelFormat, .bgr10a2Unorm)

        view.workingColorSpace = .sdr
        XCTAssertEqual(view.colorPixelFormat, .bgra8Unorm)
    }

    // MARK: - Analog distortion pack

    func testChromaticAberrationFilterIsConstructible() throws {
        guard let engine else { throw XCTSkip("No Metal device.") }
        XCTAssertNoThrow(try ChromaticAberrationFilter(engine: engine))
    }

    func testAnalogNoiseFilterIsConstructible() throws {
        guard let engine else { throw XCTSkip("No Metal device.") }
        XCTAssertNoThrow(try AnalogNoiseFilter(engine: engine))
    }

    func testHorizontalJitterFilterIsConstructible() throws {
        guard let engine else { throw XCTSkip("No Metal device.") }
        XCTAssertNoThrow(try HorizontalJitterFilter(engine: engine))
    }

    // MARK: - Temporal effects pack

    func testMotionBlurFilterIsConstructible() throws {
        guard let engine else { throw XCTSkip("No Metal device.") }
        XCTAssertNoThrow(try MotionBlurFilter(engine: engine))
    }

    func testNeonTrailsFilterIsConstructible() throws {
        guard let engine else { throw XCTSkip("No Metal device.") }
        XCTAssertNoThrow(try NeonTrailsFilter(engine: engine))
    }

    // MARK: - Color grading pack

    func testColorCorrectionFilterIsConstructible() throws {
        guard let engine else { throw XCTSkip("No Metal device.") }
        XCTAssertNoThrow(try ColorCorrectionFilter(engine: engine))
    }

    func testLUTFilterWithIdentityPresetIsConstructible() throws {
        guard let engine else { throw XCTSkip("No Metal device.") }
        XCTAssertNoThrow(try MetalForgeLUTFilter(engine: engine, preset: .identity, size: 16))
    }

    func testLUTFilterWithWarmPresetIsConstructible() throws {
        guard let engine else { throw XCTSkip("No Metal device.") }
        XCTAssertNoThrow(try MetalForgeLUTFilter(engine: engine, preset: .warm, size: 32))
    }

    func testLUTFilterRejectsWrongDataSize() throws {
        guard let engine else { throw XCTSkip("No Metal device.") }
        let badData = Data(count: 100)   // 16³×4 = 16384, definitely not 100
        XCTAssertThrowsError(
            try MetalForgeLUTFilter(engine: engine, lutSize: 16, lutData: badData)
        )
    }

    func testLUTPresetDataMatchesExpectedSize() {
        let data = MetalForgeLUTFilter.makePresetLUTData(preset: .warm, size: 32)
        XCTAssertEqual(data.count, 32 * 32 * 32 * 4)
    }

    func testMotionBlurClearHistoryDoesNotCrash() throws {
        guard let engine else { throw XCTSkip("No Metal device.") }
        let filter = try MotionBlurFilter(engine: engine)
        // Clearing without ever encoding is a valid no-op.
        filter.clearHistory()
        XCTAssertTrue(true, "clearHistory() should be safe to call before first encode")
    }

    // MARK: - Recorder

    func testRGBToYUVConverterIsConstructible() throws {
        guard let engine else { throw XCTSkip("No Metal device.") }
        XCTAssertNoThrow(try RGBToYUVConverter(engine: engine))
    }

    func testRecorderSDRIsConstructible() throws {
        guard let engine else { throw XCTSkip("No Metal device.") }
        let recorder = try MetalForgeRecorder(
            engine: engine,
            videoSize: CGSize(width: 1280, height: 720),
            workingColorSpace: .sdr,
            frameRate: 30
        )
        XCTAssertEqual(recorder.state, .idle)
    }

    func testRecorderHDRIsConstructible() throws {
        guard let engine else { throw XCTSkip("No Metal device.") }
        let recorder = try MetalForgeRecorder(
            engine: engine,
            videoSize: CGSize(width: 3840, height: 2160),
            workingColorSpace: .hdr10PQ,
            frameRate: 60
        )
        XCTAssertEqual(recorder.state, .idle)
    }

    func testRecorderStartStopRoundTrip() async throws {
        guard let engine else { throw XCTSkip("No Metal device.") }
        let recorder = try MetalForgeRecorder(
            engine: engine,
            videoSize: CGSize(width: 640, height: 480),
            workingColorSpace: .sdr,
            frameRate: 30
        )

        let tmpURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("metalforge-test-\(UUID().uuidString).mp4")
        defer { try? FileManager.default.removeItem(at: tmpURL) }

        try recorder.startRecording(outputURL: tmpURL)
        XCTAssertEqual(recorder.state, .recording)

        // Stop without any frames — writer should fail finalisation because
        // there's no session, which is the expected behaviour we want to verify.
        do {
            try await recorder.stopRecording()
            // If it succeeds, state should be .finished
            XCTAssertEqual(recorder.state, .finished)
        } catch {
            // Or it fails — also acceptable for an empty session.
            if case .failed = recorder.state {} else {
                XCTFail("Expected .failed state after failed finalisation, got \(recorder.state)")
            }
        }
    }

    // MARK: - GPU effects pack (blur + stylization)

    /// Allocate a private, writable test texture and run a filter end-to-end on
    /// it, returning whether a non-nil destination survived a committed GPU pass.
    private func runFilterOnTestTexture(
        _ filter: any MetalForgeFilter,
        engine: MetalForgeEngine,
        pixelFormat: MTLPixelFormat = .bgra8Unorm,
        size: Int = 64
    ) -> Bool {
        let desc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: pixelFormat, width: size, height: size, mipmapped: false
        )
        desc.storageMode = .private
        desc.usage = [.shaderRead, .shaderWrite]

        guard
            let source = engine.device.makeTexture(descriptor: desc),
            let destination = engine.device.makeTexture(descriptor: desc),
            let commandBuffer = engine.commandQueue.makeCommandBuffer()
        else { return false }

        filter.encode(source: source, destination: destination, commandBuffer: commandBuffer)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        return commandBuffer.status == .completed
    }

    func testGaussianBlurFilterIsConstructible() throws {
        guard let engine else { throw XCTSkip("No Metal device.") }
        XCTAssertNoThrow(try GaussianBlurFilter(engine: engine))
    }

    func testSharpenFilterIsConstructible() throws {
        guard let engine else { throw XCTSkip("No Metal device.") }
        XCTAssertNoThrow(try SharpenFilter(engine: engine))
    }

    func testVignetteFilterIsConstructible() throws {
        guard let engine else { throw XCTSkip("No Metal device.") }
        XCTAssertNoThrow(try VignetteFilter(engine: engine))
    }

    func testScanlineFilterIsConstructible() throws {
        guard let engine else { throw XCTSkip("No Metal device.") }
        XCTAssertNoThrow(try ScanlineFilter(engine: engine))
    }

    func testRGBSplitFilterIsConstructible() throws {
        guard let engine else { throw XCTSkip("No Metal device.") }
        XCTAssertNoThrow(try RGBSplitFilter(engine: engine))
    }

    func testGaussianBlurProcessesTestTexture() throws {
        guard let engine else { throw XCTSkip("No Metal device.") }
        let filter = try GaussianBlurFilter(engine: engine)
        filter.radius = 6
        XCTAssertTrue(runFilterOnTestTexture(filter, engine: engine))
        // Second pass exercises intermediate-texture reuse (no realloc).
        XCTAssertTrue(runFilterOnTestTexture(filter, engine: engine))
    }

    func testSharpenProcessesTestTexture() throws {
        guard let engine else { throw XCTSkip("No Metal device.") }
        let filter = try SharpenFilter(engine: engine)
        filter.amount = 1.0
        XCTAssertTrue(runFilterOnTestTexture(filter, engine: engine))
    }

    func testVignetteProcessesTestTexture() throws {
        guard let engine else { throw XCTSkip("No Metal device.") }
        XCTAssertTrue(runFilterOnTestTexture(try VignetteFilter(engine: engine), engine: engine))
    }

    func testScanlineProcessesTestTexture() throws {
        guard let engine else { throw XCTSkip("No Metal device.") }
        XCTAssertTrue(runFilterOnTestTexture(try ScanlineFilter(engine: engine), engine: engine))
    }

    func testRGBSplitProcessesTestTexture() throws {
        guard let engine else { throw XCTSkip("No Metal device.") }
        XCTAssertTrue(runFilterOnTestTexture(try RGBSplitFilter(engine: engine), engine: engine))
    }

    func testGPUEffectsProcessHDRTexture() throws {
        guard let engine else { throw XCTSkip("No Metal device.") }
        // Exercise the HDR-specialised PSO variants on an .rgba16Float target.
        let filters: [any MetalForgeFilter] = [
            try GaussianBlurFilter(engine: engine),
            try SharpenFilter(engine: engine),
            try VignetteFilter(engine: engine),
            try ScanlineFilter(engine: engine),
            try RGBSplitFilter(engine: engine)
        ]
        for filter in filters {
            XCTAssertTrue(
                runFilterOnTestTexture(filter, engine: engine, pixelFormat: .rgba16Float),
                "\(type(of: filter)) should process an HDR texture"
            )
        }
    }

    // MARK: - GPU effects pack (pixel behavior)

    func testVignetteDarkensCornersNotCentre() throws {
        guard let engine else { throw XCTSkip("No Metal device.") }
        let size = 16
        guard let source = PixelTest.solidColor(
            device: engine.device, size: size, rgba: SIMD4(0.5, 0.5, 0.5, 1)
        ) else { throw XCTSkip("Texture allocation failed.") }

        let filter = try VignetteFilter(engine: engine)
        filter.intensity = 0.7
        filter.radius    = 0.4
        filter.softness   = 0.4

        guard let dest = PixelTest.run(filter, source: source,
                                       device: engine.device, queue: engine.commandQueue)
        else { throw XCTSkip("GPU run failed.") }
        let out = PixelTest.readBytes(dest)

        let centre = PixelTest.color(out, x: size / 2, y: size / 2, width: size)
        let corner = PixelTest.color(out, x: 0,        y: 0,        width: size)

        // Centre stays roughly the original mid-grey.
        XCTAssertEqual(PixelTest.luma(centre), 0.5, accuracy: 0.08)
        // Corner is darker than the centre, and darker than the original.
        XCTAssertLessThan(PixelTest.luma(corner), PixelTest.luma(centre) - 0.05)
        XCTAssertLessThan(PixelTest.luma(corner), 0.45)
    }

    func testScanlineProducesPerRowBrightnessVariation() throws {
        guard let engine else { throw XCTSkip("No Metal device.") }
        let size = 16
        let base: Float = 0.6
        guard let source = PixelTest.solidColor(
            device: engine.device, size: size, rgba: SIMD4(base, base, base, 1)
        ) else { throw XCTSkip("Texture allocation failed.") }

        let filter = try ScanlineFilter(engine: engine)
        filter.intensity = 0.6
        filter.lineWidth = 2.0
        filter.timeSeed  = 0.0

        guard let dest = PixelTest.run(filter, source: source,
                                       device: engine.device, queue: engine.commandQueue)
        else { throw XCTSkip("GPU run failed.") }
        let out = PixelTest.readBytes(dest)

        // Sample the brightness of column 0 across every row.
        let rowLuma = (0..<size).map { y in
            PixelTest.luma(PixelTest.color(out, x: 0, y: y, width: size))
        }
        let minL = rowLuma.min()!
        let maxL = rowLuma.max()!

        // Rows must differ from each other (scanline banding)…
        XCTAssertGreaterThan(maxL - minL, 0.05, "rows should vary in brightness")
        // …and at least one row must be darker than the original signal.
        XCTAssertLessThan(minL, base - 0.05, "some row should be darker than the input")
    }

    func testRGBSplitShiftsChannelsDifferently() throws {
        guard let engine else { throw XCTSkip("No Metal device.") }
        let size = 16
        // Left half red, right half blue — a colour boundary the split can smear.
        guard let source = PixelTest.rgbZones(device: engine.device, size: size)
        else { throw XCTSkip("Texture allocation failed.") }

        let filter = try RGBSplitFilter(engine: engine)
        filter.redOffset   = SIMD2(0.18, 0.0)   // R sampled well to the right
        filter.greenOffset = SIMD2(0.0, 0.0)
        filter.blueOffset  = SIMD2(-0.18, 0.0)  // B sampled well to the left
        filter.intensity   = 1.0

        guard let dest = PixelTest.run(filter, source: source,
                                       device: engine.device, queue: engine.commandQueue)
        else { throw XCTSkip("GPU run failed.") }
        let out = PixelTest.readBytes(dest)
        let inp = PixelTest.readBytes(source)

        // Near the boundary the three channels are pulled from different zones,
        // so the per-channel result diverges and the output differs from input.
        let bx = size / 2
        let by = size / 2
        let o = PixelTest.color(out, x: bx, y: by, width: size)
        let i = PixelTest.color(inp, x: bx, y: by, width: size)

        // R is pulled from one zone, B from the other, so the two channels
        // change by different amounts relative to the input.
        let dR = abs(o.x - i.x)
        let dG = abs(o.y - i.y)
        let dB = abs(o.z - i.z)
        XCTAssertGreaterThan(abs(dR - dB), 0.1, "R and B channels should change by different amounts")
        XCTAssertGreaterThan(dR + dG + dB, 0.1, "output should differ from input near the boundary")
    }

    func testGaussianBlurSpreadsCentrePixel() throws {
        guard let engine else { throw XCTSkip("No Metal device.") }
        let size = 16
        guard let source = PixelTest.centerBright(device: engine.device, size: size)
        else { throw XCTSkip("Texture allocation failed.") }

        let filter = try GaussianBlurFilter(engine: engine)
        filter.radius    = 3.0
        filter.intensity = 1.0   // fully blurred

        guard let dest = PixelTest.run(filter, source: source,
                                       device: engine.device, queue: engine.commandQueue)
        else { throw XCTSkip("GPU run failed.") }
        let out = PixelTest.readBytes(dest)

        let c  = size / 2
        let centre   = PixelTest.luma(PixelTest.color(out, x: c,     y: c, width: size))
        let near     = PixelTest.luma(PixelTest.color(out, x: c + 1, y: c, width: size))
        let far      = PixelTest.luma(PixelTest.color(out, x: c + 5, y: c, width: size))

        // The lone white centre spreads out: centre dims below 1, the neighbour
        // lights up above 0, and energy falls off with distance.
        XCTAssertLessThan(centre, 0.95, "centre should darken as it spreads")
        XCTAssertGreaterThan(near, 0.01, "the neighbour should brighten above zero")
        XCTAssertGreaterThan(near, far, "near pixels should be brighter than far pixels")
    }

    func testSharpenChangesPixelsNearEdge() throws {
        guard let engine else { throw XCTSkip("No Metal device.") }
        let size = 16
        guard let source = PixelTest.verticalEdge(
            device: engine.device, size: size,
            dark: SIMD4(0.2, 0.2, 0.2, 1), bright: SIMD4(0.8, 0.8, 0.8, 1)
        ) else { throw XCTSkip("Texture allocation failed.") }

        let filter = try SharpenFilter(engine: engine)
        filter.amount = 1.0

        guard let dest = PixelTest.run(filter, source: source,
                                       device: engine.device, queue: engine.commandQueue)
        else { throw XCTSkip("GPU run failed.") }
        let out = PixelTest.readBytes(dest)
        let inp = PixelTest.readBytes(source)

        // Look at the columns straddling the boundary (x = size/2-1 and size/2)
        // on a middle row; unsharp masking should over/undershoot there.
        let y = size / 2
        var maxDelta: Float = 0
        for x in [size / 2 - 1, size / 2] {
            let o = PixelTest.luma(PixelTest.color(out, x: x, y: y, width: size))
            let i = PixelTest.luma(PixelTest.color(inp, x: x, y: y, width: size))
            maxDelta = max(maxDelta, abs(o - i))
        }
        XCTAssertGreaterThan(maxDelta, 0.03, "sharpen should change pixels near the edge")
    }

    func testVignetteHDRDarkensCorners() throws {
        guard let engine else { throw XCTSkip("No Metal device.") }
        let size = 16
        guard let source = PixelTest.makeTexture(
            device: engine.device, width: size, height: size, pixelFormat: .rgba16Float
        ) else { throw XCTSkip("Texture allocation failed.") }
        PixelTest.fillHDR(source) { _, _ in SIMD4(0.5, 0.5, 0.5, 1) }

        let filter = try VignetteFilter(engine: engine)
        filter.intensity = 0.7
        filter.radius    = 0.4
        filter.softness   = 0.4

        guard let dest = PixelTest.run(filter, source: source,
                                       device: engine.device, queue: engine.commandQueue)
        else { throw XCTSkip("GPU run failed.") }

        let centre = PixelTest.colorHDR(dest, x: size / 2, y: size / 2)
        let corner = PixelTest.colorHDR(dest, x: 0,        y: 0)
        XCTAssertEqual(PixelTest.luma(centre), 0.5, accuracy: 0.08)
        XCTAssertLessThan(PixelTest.luma(corner), PixelTest.luma(centre) - 0.05)
    }

    // MARK: - Effect presets

    func testPresetAllCasesIsNotEmpty() {
        XCTAssertFalse(MetalForgeEffectPreset.allCases.isEmpty)
    }

    func testEveryPresetHasNonEmptyDisplayName() {
        for preset in MetalForgeEffectPreset.allCases {
            XCTAssertFalse(
                preset.displayName.isEmpty,
                "\(preset) should have a non-empty displayName"
            )
        }
    }

    func testEveryPresetHasNonEmptyDescription() {
        for preset in MetalForgeEffectPreset.allCases {
            XCTAssertFalse(
                preset.description.isEmpty,
                "\(preset) should have a non-empty description"
            )
        }
    }

    func testMakeFiltersDoesNotThrowForEveryPreset() throws {
        guard let engine else { throw XCTSkip("No Metal device.") }
        for preset in MetalForgeEffectPreset.allCases {
            XCTAssertNoThrow(
                try preset.makeFilters(engine: engine),
                "makeFilters should not throw for \(preset)"
            )
        }
    }

    func testMakeFiltersReturnsNonEmptyChainForEveryPreset() throws {
        guard let engine else { throw XCTSkip("No Metal device.") }
        for preset in MetalForgeEffectPreset.allCases {
            let filters = try preset.makeFilters(engine: engine)
            XCTAssertFalse(filters.isEmpty, "\(preset) should produce at least one filter")
        }
    }

    func testApplyPresetAddsFiltersToPipeline() throws {
        guard let engine else { throw XCTSkip("No Metal device.") }
        let pipeline = try MetalForgePipeline(engine: engine)
        XCTAssertEqual(pipeline.filterCount, 0)

        try pipeline.applyPreset(.vhs)
        XCTAssertGreaterThan(
            pipeline.filterCount, 0,
            "applyPreset should append the preset's filters"
        )
    }

    func testApplyPresetReplacesPreviousChain() throws {
        guard let engine else { throw XCTSkip("No Metal device.") }
        let pipeline = try MetalForgePipeline(engine: engine)

        try pipeline.applyPreset(.vhs)
        let vhsCount = pipeline.filterCount
        XCTAssertEqual(vhsCount, try MetalForgeEffectPreset.vhs.makeFilters(engine: engine).count)

        // Applying another preset should replace, not accumulate.
        try pipeline.applyPreset(.noir)
        XCTAssertEqual(
            pipeline.filterCount,
            try MetalForgeEffectPreset.noir.makeFilters(engine: engine).count
        )
    }

    func testOriginalIsNotAPresetCase() {
        // "Original" is the no-effect state and must not be modelled as a preset.
        let rawValues = MetalForgeEffectPreset.allCases.map { $0.rawValue.lowercased() }
        XCTAssertFalse(rawValues.contains("original"))

        let displayNames = MetalForgeEffectPreset.allCases.map { $0.displayName.lowercased() }
        XCTAssertFalse(displayNames.contains("original"))
    }

    /// Whether a preset's filter chain contains an instance of the given type.
    private func preset(
        _ preset: MetalForgeEffectPreset,
        engine: MetalForgeEngine,
        contains type: (any MetalForgeFilter.Type)
    ) throws -> Bool {
        try preset.makeFilters(engine: engine).contains { Swift.type(of: $0) == type }
    }

    func testVHSPresetUsesGPUStylizationFilter() throws {
        guard let engine else { throw XCTSkip("No Metal device.") }
        let hasScanline = try preset(.vhs, engine: engine, contains: ScanlineFilter.self)
        let hasRGBSplit = try preset(.vhs, engine: engine, contains: RGBSplitFilter.self)
        XCTAssertTrue(hasScanline || hasRGBSplit,
                      "vhs should include ScanlineFilter or RGBSplitFilter")
    }

    func testDreamyPresetUsesGaussianBlur() throws {
        guard let engine else { throw XCTSkip("No Metal device.") }
        XCTAssertTrue(try preset(.dreamy, engine: engine, contains: GaussianBlurFilter.self))
    }

    func testHighContrastPresetUsesSharpen() throws {
        guard let engine else { throw XCTSkip("No Metal device.") }
        XCTAssertTrue(try preset(.highContrast, engine: engine, contains: SharpenFilter.self))
    }

    func testVintageFilmPresetUsesVignette() throws {
        guard let engine else { throw XCTSkip("No Metal device.") }
        XCTAssertTrue(try preset(.vintageFilm, engine: engine, contains: VignetteFilter.self))
    }

    func testCyberpunkPresetUsesGPUEffect() throws {
        guard let engine else { throw XCTSkip("No Metal device.") }
        let hasRGBSplit   = try preset(.cyberpunk, engine: engine, contains: RGBSplitFilter.self)
        let hasNeonTrails = try preset(.cyberpunk, engine: engine, contains: NeonTrailsFilter.self)
        XCTAssertTrue(hasRGBSplit || hasNeonTrails,
                      "cyberpunk should include RGBSplitFilter or NeonTrailsFilter")
    }

    func testColorSpaceDetectionRecognisesHLG() {
        var pb: CVPixelBuffer?
        let attrs: [CFString: Any] = [
            kCVPixelBufferMetalCompatibilityKey: true,
            kCVPixelBufferIOSurfacePropertiesKey: [:] as CFDictionary
        ]
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault, 64, 64,
            kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange,
            attrs as CFDictionary, &pb
        )
        XCTAssertEqual(status, kCVReturnSuccess)
        guard let buffer = pb else { return }

        CVBufferSetAttachment(
            buffer,
            kCVImageBufferTransferFunctionKey,
            kCVImageBufferTransferFunction_ITU_R_2100_HLG,
            .shouldPropagate
        )
        XCTAssertEqual(MetalForgeColorSpace.detect(from: buffer), .hlg)
    }
}
