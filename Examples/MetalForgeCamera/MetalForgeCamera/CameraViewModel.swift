import SwiftUI
import AVFoundation
import CoreVideo
import CoreMedia
@preconcurrency import Metal
import MetalForge

// MARK: - Preset choices exposed to the UI

/// The set of looks the example app offers in its picker.
///
/// `original` is the no-effect state — an empty pipeline that renders the raw
/// camera signal. It is intentionally **not** a member of
/// `MetalForgeEffectPreset`; the library models "no preset" as the absence of a
/// preset, so the demo wraps the library enum in this UI-only type to add the
/// `original` entry.
enum PresetChoice: Hashable, Identifiable, Sendable {
    case original
    case preset(MetalForgeEffectPreset)

    /// Original first, followed by every library preset in declaration order.
    static var allCases: [PresetChoice] {
        [.original] + MetalForgeEffectPreset.allCases.map(PresetChoice.preset)
    }

    var id: String {
        switch self {
        case .original:          return "original"
        case .preset(let preset): return preset.rawValue
        }
    }

    /// Label shown in the picker.
    var displayName: String {
        switch self {
        case .original:           return "Original"
        case .preset(let preset): return preset.displayName
        }
    }

    /// The wrapped library preset, or `nil` for `original`.
    var preset: MetalForgeEffectPreset? {
        switch self {
        case .original:           return nil
        case .preset(let preset): return preset
        }
    }
}

// MARK: - View model

/// SwiftUI-bound controller that owns the MetalForge engine + pipeline and
/// drives the camera preview.
///
/// Threading:
/// - `@MainActor` for `@Published` state and UI access.
/// - Frames arrive on `MetalForgeCaptureManager.captureQueue` (background) and
///   are forwarded into `MetalForgePipeline.process(pixelBuffer:)` directly,
///   without hopping to the main actor (the pipeline does its own GPU sync).
/// - `view.present(texture:)` runs back on the main actor.
@MainActor
final class CameraViewModel: ObservableObject {

    // MARK: Published UI state

    @Published var permissionsGranted = false
    @Published var setupError: String?

    /// Currently selected preset (or `.original` for the raw camera signal).
    @Published var activePreset: PresetChoice = .original {
        didSet { applyActivePreset() }
    }

    /// When `true`, the selected preset is bypassed and the pipeline renders the
    /// raw camera signal (empty filter chain).
    @Published var showOriginal: Bool = false {
        didSet { applyActivePreset() }
    }

    /// Frames per second over a rolling 1-second window. Updated on the main
    /// actor from the capture callback.
    @Published private(set) var fps: Double = 0

    // MARK: MetalForge components

    /// Exposed so `CameraPreviewView` can hand the underlying `MetalForgeView`
    /// to SwiftUI through `MetalForgeViewRepresentable`.
    let engine: MetalForgeEngine
    let view: MetalForgeView

    private let pipeline: MetalForgePipeline
    private let capture: MetalForgeCaptureManager

    /// Serialises access to the pipeline's filter chain. `applyPreset` rebuilds
    /// the chain (a mutation of the filter array) while frames are processed on
    /// the capture queue — both sides take this lock so the swap never races a
    /// `process(_:)` call.
    ///
    /// Trade-off (acceptable for a demo): the lock is held across a whole
    /// `process(_:)` call — which blocks on the GPU via `waitUntilCompleted` —
    /// so a preset switch waits at most one frame, and conversely a switch that
    /// compiles new shader pipelines can briefly stall the capture thread. A
    /// production app would build the new chain off-thread and swap only an
    /// atomic reference.
    ///
    /// No deadlock risk: this is a plain, non-recursive `NSLock` acquired in
    /// exactly two places (`handleVideoFrame` and `applyActivePreset`), never
    /// nested and never held across an `await`.
    private let pipelineLock = NSLock()

    // MARK: FPS bookkeeping

    private var fpsFrameCount: Int = 0
    private var fpsWindowStart: CFTimeInterval = CACurrentMediaTime()

    // MARK: Init

    init() {
        // Init failure is fatal for the demo. A production app should surface
        // this through a proper error UI.
        do {
            let engine = try MetalForgeEngine()
            self.engine = engine
            self.view = try MetalForgeView(engine: engine)
            self.pipeline = try MetalForgePipeline(engine: engine)
        } catch {
            fatalError("MetalForge initialisation failed: \(error)")
        }

        self.capture = MetalForgeCaptureManager()

        // Start on the raw camera signal (empty chain).
        applyActivePreset()

        // Recycle each presented texture back into the pool after the GPU
        // finishes the draw. Capturing `pipeline` weakly avoids any retain
        // cycle through the closure.
        view.recycleHandler = { [weak pipeline] texture in
            pipeline?.recycle(texture)
        }
    }

    // MARK: Setup

    /// Request camera permission and start the capture session.
    /// Call from a SwiftUI `.task { ... }`.
    func setup() async {
        let granted = await MetalForgeCaptureManager.requestCameraAccess()
        permissionsGranted = granted
        guard granted else {
            setupError = "Camera access denied. Enable it in Settings to use the demo."
            return
        }

        do {
            // The demo runs in SDR — keeps the example simple. The library
            // supports HDR via `preferHDR: true`; see MetalForge README.
            try capture.configure(position: .back, preferHDR: false)
        } catch {
            setupError = "Camera configure failed: \(error.localizedDescription)"
            return
        }

        capture.onVideoFrame = { [weak self] pixelBuffer, _ in
            self?.handleVideoFrame(pixelBuffer)
        }
        // No recording in this initial example, so no onAudioSample handler.

        capture.startCapture()
    }

    // MARK: Frame ingest (captureQueue)

    /// Called on `MetalForgeCaptureManager`'s background capture queue.
    nonisolated private func handleVideoFrame(_ pixelBuffer: CVPixelBuffer) {
        // The pipeline drives its own GPU sync internally; we can call it
        // straight from the capture queue. The lock guards against a concurrent
        // preset swap rebuilding the filter chain mid-frame.
        pipelineLock.lock()
        let texture = pipeline.process(pixelBuffer: pixelBuffer)
        pipelineLock.unlock()
        guard let texture else { return }

        // Hop to the main actor to present + bump FPS. The actor isolation
        // means the view's `present(texture:)` call is safe.
        Task { @MainActor [weak self] in
            self?.present(texture: texture)
        }
    }

    @MainActor
    private func present(texture: MTLTexture) {
        view.present(texture: texture)
        tickFPS()
    }

    private func tickFPS() {
        fpsFrameCount += 1
        let now = CACurrentMediaTime()
        let elapsed = now - fpsWindowStart
        if elapsed >= 1.0 {
            fps = Double(fpsFrameCount) / elapsed
            fpsFrameCount = 0
            fpsWindowStart = now
        }
    }

    // MARK: Preset routing

    /// Rebuild the pipeline's filter chain for the current `activePreset`.
    ///
    /// `showOriginal` (or selecting `.original`) clears the chain to render the
    /// raw camera signal. Otherwise the library's `applyPreset(_:)` swaps in the
    /// preset's curated filter chain. The swap is serialised against frame
    /// processing via `pipelineLock`.
    private func applyActivePreset() {
        pipelineLock.lock()
        defer { pipelineLock.unlock() }

        guard !showOriginal, let preset = activePreset.preset else {
            pipeline.removeAllFilters()
            return
        }

        do {
            try pipeline.applyPreset(preset)
        } catch {
            // Fall back to the raw signal if a preset fails to build.
            pipeline.removeAllFilters()
            setupError = "Failed to apply preset \(preset.displayName): \(error.localizedDescription)"
        }
    }
}
