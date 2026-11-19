import InferenceContracts
import AppKit
import StudioCore
import SwiftUI

struct MovieUpscaleInspector:View {
  @EnvironmentObject var store:StudioStore
  var clip:Clip
  var showLTXSettings = true
  private var settings:LTX25MovieUpscaleSettings { clip.generationSelection?.ltx25MovieUpscale ?? .init() }
  private func binding<T>(_ key:WritableKeyPath<LTX25MovieUpscaleSettings,T>)->Binding<T> {
    Binding(get:{ settings[keyPath:key] },set:{ value in store.editClip {
      var selection=$0.generationSelection ?? GenerationSelection(task:"video_upscale",preset:.custom)
      var options=selection.ltx25MovieUpscale ?? LTX25MovieUpscaleSettings()
      options[keyPath:key]=value;selection.ltx25MovieUpscale=options;$0.generationSelection=selection
    } })
  }
  var body:some View {
    VStack(alignment: .leading, spacing: 8) {
      if clip.usesNativeMovieUpscaler { GenerationPresetControls(target: .clip(clip.id)) }
      Text(clip.usesNativeMovieUpscaler ? "Source movie upscale" : settings.mode == .enhance ? "Source movie enhancement" : "Source movie 2× upscale").font(.headline)
      Picker("Upscaler", selection: Binding(get: {
        clip.generationSelection?.movieUpscaleMethod ?? .ltx25
      }, set: { method in
        store.editClip {
          var selection = $0.generationSelection ?? GenerationSelection(task: "video_upscale", preset: .custom)
          selection.movieUpscaleMethod = method == .ltx25 ? nil : method
          if method == .ltx25 && selection.ltx25MovieUpscale == nil {
            selection.ltx25MovieUpscale = LTX25MovieUpscaleSettings()
            if $0.prompt.isEmpty { $0.prompt = "Preserve the source subject, motion and scene while refining detail." }
          }
          $0.generationSelection = selection
        }
      })) {
        ForEach(MovieUpscaleMethod.allCases) { Text($0.label).tag($0) }
      }
      .accessibilityIdentifier("movie-upscale-method")
      .disabled(store.operationBusy)
      if clip.usesReCaVSR {
        ReCaVSRControls(settings: Binding(get: {
          var value = clip.generationSelection?.reCaVSRSettings ?? (try! ReCaVSRSettings())
          value.seed = UInt64(max(0, clip.seed)); return value
        },
          set: { value in
            guard value.seed <= UInt64(Int.max) else { store.error = "This clip supports seeds up to \(Int.max)."; return }
            store.editClip {
            var selection = $0.generationSelection ?? GenerationSelection(task: "video_upscale", preset: .custom)
            selection.reCaVSRSettings = value; $0.seed = Int(value.seed); $0.generationSelection = selection
          } }))
        Button("Upscale clip") { Task { await store.generateSelected() } }
          .buttonStyle(.borderedProminent).accessibilityIdentifier("recavsr-upscale-clip").disabled(store.operationBusy)
      } else if clip.usesRelayVSR {
        relayControls
      } else if showLTXSettings {
        ltxControls
      }
    }
    .task(id: clip.usesNativeMovieUpscaler ? store.generationDescriptionTaskKey(for: clip) : nil) {
      if clip.usesNativeMovieUpscaler { await store.describeGeneration() }
    }
  }

  private var relayControls: some View {
    VStack(alignment: .leading, spacing: 8) {
      Picker("Scale", selection: Binding(get: { clip.generationSelection?.relayVSRScale ?? .x2 }, set: { scale in
        store.editClip {
          var selection = $0.generationSelection ?? GenerationSelection(task: "video_upscale", preset: .custom)
          selection.relayVSRScale = scale == .x2 ? nil : scale; $0.generationSelection = selection
        }
      })) {
        ForEach(NativeRelayVSRScale.allCases, id: \.self) { Text($0.label).tag($0) }
      }.accessibilityIdentifier("relayvsr-scale")
      if let reference = clip.attachments.first(where: { $0.role == .reference && $0.isEnabled }),
        let asset = store.allAssets.first(where: { $0.id == reference.assetID }) {
        LabeledContent("Reference movie", value: asset.name)
        if let duration = reference.sourceDurationSeconds {
          let start = reference.sourceStartSeconds ?? 0
          LabeledContent("Source interval", value: "\(start.formatted(.number.precision(.fractionLength(3))))–\((start + duration).formatted(.number.precision(.fractionLength(3)))) s")
        }
      }
      Text("Uses the matching native 2× or 4× model and retains source frame timing and audio.")
        .font(.caption).foregroundStyle(.secondary)
      if clip.attachments.contains(where: { [.first, .last, .audioDriver].contains($0.role) && $0.isEnabled }) {
        Text("LTX endpoint images and audio drivers apply when LTX 2× is selected.")
          .font(.caption).foregroundStyle(.secondary)
      }
      RelayVSRModelFolderControl(scale: clip.generationSelection?.relayVSRScale ?? .x2)
      if let error = store.validationErrors[clip.id] {
        Text(error).font(.caption).foregroundStyle(.red).textSelection(.enabled)
      }
      Button("Upscale clip", systemImage: "arrow.up.left.and.arrow.down.right") {
        Task { await store.generateSelected() }
      }
      .buttonStyle(.borderedProminent)
      .accessibilityIdentifier("relayvsr-upscale-clip")
      .disabled(store.operationBusy)
    }.font(.caption)
  }

  private var ltxControls: some View {
    VStack(alignment:.leading,spacing:8) {
      Toggle("Enable experimental movie upscaling",isOn:binding(\.experimentalEnabled))
      Text(settings.mode == .enhance ? "Uses the Reference movie’s selected interval, original frame rate and audio. Output dimensions match the prepared source grid." : "Uses the Reference movie’s selected interval, original frame rate and audio. Output dimensions are twice the prepared source grid.").font(.caption2).foregroundStyle(.secondary)
      Picker("Mode",selection:binding(\.mode)) {
        ForEach(LTX25MovieUpscaleSettings.Mode.allCases.filter { $0 != .enhance || settings.mode == .enhance }) { Text($0.label).tag($0) }
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

struct RelayVSRModelFolderControl: View {
  @EnvironmentObject var store: StudioStore
  var scale: NativeRelayVSRScale = .x2
  private var resolvedFolder: String? {
    store.runtime.relayVSRModelDirectory.flatMap { try? scale.resolveDirectory(URL(fileURLWithPath: $0)).path }
  }

  var body: some View {
    HStack {
      Button("Choose RelayVSR model folder…") {
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        if let path = store.runtime.relayVSRModelDirectory { panel.directoryURL = URL(fileURLWithPath: path) }
        if panel.runModal() == .OK, let folder = panel.url {
          store.runtime.relayVSRModelDirectory = folder.path
          store.saveRuntime(reloadProfiles: false)
        }
      }.disabled(store.operationBusy)
      Text(resolvedFolder ?? "Select the matching \(scale.label) checkpoint folder.")
        .font(.caption).lineLimit(1).truncationMode(.middle).foregroundStyle(.secondary)
        .textSelection(.enabled)
    }
  }
}
