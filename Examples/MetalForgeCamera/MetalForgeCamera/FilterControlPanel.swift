import SwiftUI

/// Bottom control panel: preset picker + before/after toggle.
///
/// Pure-UI; all state is bound to `CameraViewModel`.
struct FilterControlPanel: View {

    @ObservedObject var viewModel: CameraViewModel

    var body: some View {
        VStack(spacing: 16) {
            // ----- Preset picker -----
            HStack {
                Image(systemName: "wand.and.stars")
                Text("Preset")
                    .font(.subheadline)
                Spacer()
                Picker("Preset", selection: $viewModel.activePreset) {
                    ForEach(PresetChoice.allCases) { choice in
                        Text(choice.displayName).tag(choice)
                    }
                }
                .pickerStyle(.menu)
                .tint(.white)
                .disabled(viewModel.showOriginal)
            }
            .foregroundStyle(.white)

            // ----- Before / After toggle -----
            HStack {
                Image(systemName: "rectangle.righthalf.inset.filled.arrow.right")
                Toggle("Show Original (Before)", isOn: $viewModel.showOriginal)
                    .font(.subheadline)
            }
            .foregroundStyle(.white)
            .tint(.white)
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 18)
        .background(.ultraThinMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 24, style: .continuous))
        .padding(.horizontal, 12)
        .padding(.bottom, 12)
    }
}
