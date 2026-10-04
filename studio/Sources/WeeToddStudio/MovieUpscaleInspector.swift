import StudioCore
import SwiftUI

struct MovieUpscaleInspector:View {
  @EnvironmentObject var store:StudioStore
  var clip:Clip
  private var settings:LTX25MovieUpscaleSettings { clip.generationSelection?.ltx25MovieUpscale ?? .init() }
  private func binding<T>(_ key:WritableKeyPath<LTX25MovieUpscaleSettings,T>)->Binding<T> {
    Binding(get:{ settings[keyPath:key] },set:{ value in store.editClip {
      var selection=$0.generationSelection ?? GenerationSelection(task:"video_upscale",preset:.custom)
      var options=selection.ltx25MovieUpscale ?? LTX25MovieUpscaleSettings()
      options[keyPath:key]=value;selection.ltx25MovieUpscale=options;$0.generationSelection=selection
    } })
  }
  var body:some View {
    VStack(alignment:.leading,spacing:8) {
      Text("Source movie 2× upscale").font(.headline)
      Toggle("Enable experimental movie upscaling",isOn:binding(\.experimentalEnabled))
      Text("Uses the Reference movie's selected interval, original frame rate and audio. Output dimensions are twice the prepared source grid.").font(.caption2).foregroundStyle(.secondary)
      Picker("Mode",selection:binding(\.mode)) {
        ForEach(LTX25MovieUpscaleSettings.Mode.allCases) { Text($0.label).tag($0) }
      }
      Picker("Source grid",selection:binding(\.sizePolicy)) {
        Text("Nearest grid preserving aspect").tag(LTX25MovieUpscaleSettings.SizePolicy.fitNearest)
        Text("Center crop").tag(LTX25MovieUpscaleSettings.SizePolicy.centerCrop)
        Text("Require exact grid").tag(LTX25MovieUpscaleSettings.SizePolicy.strict)
      }
      if settings.mode != .latentOnly {
        LabeledContent("Refinement strength",value:settings.refinementStrength.formatted(.number.precision(.fractionLength(2))))
        Slider(value:binding(\.refinementStrength),in:0.05...0.85)
        Picker("Source anchors",selection:binding(\.anchors)) {
          Text("None").tag(LTX25MovieUpscaleSettings.Anchors.none)
          Text("First frame").tag(LTX25MovieUpscaleSettings.Anchors.first)
          Text("First and last frames").tag(LTX25MovieUpscaleSettings.Anchors.firstLast)
        }
        Slider(value:binding(\.anchorStrength),in:0...1) { Text("Anchor strength") }
        Text("Three refinement updates. Optional First/Last image attachments replace only the outer source anchors.").font(.caption2).foregroundStyle(.secondary)
      } else {
        Text("Learned 2× only requires No anchors and an empty prompt.").font(.caption2).foregroundStyle(.secondary)
        Picker("Source anchors",selection:binding(\.anchors)) { Text("None").tag(LTX25MovieUpscaleSettings.Anchors.none) }
      }
      if settings.mode == .pixelSpatial {
        TextField("Pixel-Spatial adapter path",text:Binding(get:{ settings.pixelSpatialAdapterPath ?? "" },set:{ value in
          store.editClip { $0.generationSelection?.ltx25MovieUpscale?.pixelSpatialAdapterPath=value.isEmpty ? nil:value }
        }))
        Text("Select its compatible Pixel-Spatial task adapter.").font(.caption2).foregroundStyle(.secondary)
        Slider(value:binding(\.pixelStrength),in:0.05...2) { Text("Pixel-Spatial LoRA strength") }
      }
      Picker("Audio",selection:binding(\.audioPolicy)) {
        Text("Original movie audio").tag(LTX25MovieUpscaleSettings.AudioPolicy.source)
        Text("Audio driver attachment").tag(LTX25MovieUpscaleSettings.AudioPolicy.sidecar)
        Text("Silence").tag(LTX25MovieUpscaleSettings.AudioPolicy.silence)
      }
      DisclosureGroup("Chunks and resume") {
        Toggle("Split source into bounded chunks",isOn:binding(\.chunking))
        if settings.chunking {
          TextField("Chunk frame-megapixel budget",value:binding(\.chunkFrameMegapixelBudget),format:.number)
          Toggle("Resume verified completed chunks",isOn:binding(\.resume))
        }
        Toggle("Keep completed chunks",isOn:binding(\.keepChunks))
        Text("Resume requires the same source, recipe, model files and worker. It does not repeat completed chunks.").font(.caption2).foregroundStyle(.secondary)
      }
    }
  }
}
