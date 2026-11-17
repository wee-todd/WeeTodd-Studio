import StudioCore
import SwiftUI

struct ImageGenerationDetailsView: View {
  @EnvironmentObject private var store: StudioStore
  @Environment(\.dismiss) private var dismiss
  let asset: MediaAsset

  private var enhancement: [String: JSONValue]? {
    guard case .object(let value) = asset.generation?.nativeRequest?["promptEnhancement"] else { return nil }
    return value
  }
  private var effectivePrompt: String? {
    if let value = asset.generation?.effectivePrompt { return value }
    guard case .string(let value) = asset.generation?.nativeRequest?["prompt"] else { return nil }
    return value
  }
  private var isEnhanced: Bool { enhancement?["enabled"] == .boolean(true) }

  var body: some View {
    VStack(alignment: .leading, spacing: 16) {
      HStack {
        Text("Generation details").font(.title2.bold())
        Spacer()
        Button("Done") { dismiss() }.keyboardShortcut(.defaultAction)
      }
      HSplitView {
        WorkspaceImage(path: asset.path).aspectRatio(contentMode: .fit)
          .frame(minWidth: 260, idealWidth: 360, maxWidth: .infinity, maxHeight: .infinity)
          .accessibilityLabel("Generated image")
        ScrollView {
          VStack(alignment: .leading, spacing: 16) {
            Text(asset.name).font(.headline)
            if let generation = asset.generation {
              Text(generation.modelID).font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
              if let seed = generation.configuration?["seed"] {
                switch seed {
                case .integer(let value): Text("Seed · \(value)").font(.caption)
                case .number(let value): Text("Seed · \(value.formatted(.number.precision(.fractionLength(0))))").font(.caption)
                default: EmptyView()
                }
              }
              prompt("Authored prompt", text: generation.prompt)
              if generation.provider == ImageExecutionProvider.nativeMOSS.rawValue, generation.mossLayout != nil {
                Button("Open layout as a new draft") {
                  store.reopenMOSSLayout(asset); dismiss()
                }.disabled(store.operationBusy)
              }
              if let effectivePrompt, isEnhanced || effectivePrompt != generation.prompt {
                prompt(isEnhanced ? "Enhanced prompt used" : "Prompt used", text: effectivePrompt)
              }
              if let seconds = enhancement?["enhancementSeconds"] {
                switch seconds {
                case .number(let value) where value.isFinite && value >= 0:
                  Text("Prompt enhancement · \(value.formatted(.number.precision(.fractionLength(2)))) s").font(.caption)
                case .integer(let value) where value >= 0:
                  Text("Prompt enhancement · \(value) s").font(.caption)
                default: EmptyView()
                }
              }
            }
          }.frame(maxWidth: .infinity, alignment: .leading).padding(.leading, 16)
        }.frame(minWidth: 360, idealWidth: 440)
      }
    }.padding(24).frame(minWidth: 800, idealWidth: 940, minHeight: 560, idealHeight: 680)
  }

  private func prompt(_ title: String, text: String) -> some View {
    VStack(alignment: .leading, spacing: 6) {
      Text(title).font(.caption.bold())
      Text(text).textSelection(.enabled).frame(maxWidth: .infinity, alignment: .leading)
    }
  }
}
