import Foundation

public extension MetalForgePipeline {

    /// Replace the pipeline's current filter chain with the given preset.
    ///
    /// Existing filters are removed first, then the preset's freshly-built chain
    /// is appended in order. The filters are constructed with the pipeline's own
    /// `engine`, so they share the same Metal device and shader libraries.
    ///
    /// - Parameter preset: The look to apply.
    /// - Throws: Any error thrown while compiling the preset's filters.
    func applyPreset(_ preset: MetalForgeEffectPreset) throws {
        let filters = try preset.makeFilters(engine: engine)
        removeAllFilters()
        for filter in filters {
            append(filter)
        }
    }
}
