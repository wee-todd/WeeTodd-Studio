import AVKit
import StudioCore
import SwiftUI

struct VideoUpscaleInspector: View {
  @EnvironmentObject var store: StudioStore
  @State private var setupOpen = false
  private func binding<T>(_ key: WritableKeyPath<FlashVSRSettings, T>) -> Binding<T> {
    Binding(get: { store.videoUpscaleSettings[keyPath: key] }, set: { store.videoUpscaleSettings[keyPath: key] = $0; store.videoUpscalePlan = nil })
  }
  var body: some View {
    VStack(alignment: .leading, spacing: 16) {
      HStack {
        VStack(alignment: .leading) {
          Text("Upscale Video").font(.title2.bold())
          Text("Native Swift/MLX · original frame rate and audio").foregroundStyle(.secondary)
        }
        Spacer()
        Button("Done") { store.saveVideoUpscalePreferences(); store.showVideoUpscale = false }.disabled(store.videoUpscaleBusy)
      }
      Picker("Restoration model", selection: $store.videoUpscaleTool) {
        Text("FlashVSR+").tag(NativeUpscalerTool.flashVSR)
        Text("H3 Regenerate").tag(NativeUpscalerTool.h3Regenerate)
      }.pickerStyle(.segmented).disabled(store.videoUpscaleBusy)
        .accessibilityIdentifier("native-upscaler-model")
      ScrollView {
        VStack(alignment: .leading, spacing: 16) {
          HStack {
            Button("Choose movie…") { store.chooseVideoUpscaleSource() }
            Text(store.videoUpscaleSourcePath.isEmpty ? "Choose a movie or a video in the media library." : URL(fileURLWithPath: store.videoUpscaleSourcePath).lastPathComponent)
              .lineLimit(1).truncationMode(.middle).textSelection(.enabled)
          }.disabled(store.videoUpscaleBusy)
          Text("Processes the complete source movie. Use a clip’s finishing controls for generation and ordered processing. Originals and completed results stay available.")
            .font(.caption).foregroundStyle(.secondary)
          Group {
            if store.videoUpscaleTool == .h3Regenerate { H3RegenerateInspector() }
            else { flashControls }
            DisclosureGroup("Model setup", isExpanded: $setupOpen) { setupControls.padding(.top, 8) }
          }.disabled(store.videoUpscaleBusy)
          if store.videoUpscaleTool == .flashVSR, let receipt = store.videoUpscaleResult {
            VideoUpscaleComparison(receipt: receipt, busy: store.videoUpscaleBusy).id(receipt.video.path)
            reveal(receipt.video.path)
          } else if store.videoUpscaleTool == .h3Regenerate, let receipt = store.h3RegenerateResult {
            H3RegenerateComparison(receipt: receipt).id(receipt.video.path)
            reveal(receipt.video.path)
          }
        }
      }
      Divider()
      Text(store.videoUpscaleStatus).font(.callout).textSelection(.enabled).lineLimit(4).frame(maxWidth: .infinity, alignment: .leading)
      HStack {
        if store.videoUpscaleBusy { ProgressView().controlSize(.small); Button("Cancel") { store.cancelVideoProcessing() } }
        Spacer()
        Button("Check setup") { run(preflightOnly: true) }.disabled(store.videoUpscaleBusy || store.videoUpscaleSourcePath.isEmpty)
        Button("Upscale") { run() }.buttonStyle(.borderedProminent)
          .disabled(store.videoUpscaleBusy || store.videoUpscaleSourcePath.isEmpty)
      }
    }.padding(24).frame(width: 840, height: 780)
      .onAppear { setupOpen = store.videoUpscaleTool == .h3Regenerate ? store.h3RegenerateSettings.baseProfilePath.isEmpty : store.videoUpscaleRuntime.modelRoot.isEmpty }
  }
  private func run(preflightOnly: Bool = false) {
    if store.videoUpscaleTool == .h3Regenerate { store.runH3Regenerate(preflightOnly: preflightOnly) }
    else { store.runVideoUpscale(preflightOnly: preflightOnly) }
  }
  private func reveal(_ path: String) -> some View {
    HStack {
      Button("Reveal result") { NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: path)]) }
      Text("Original retained · reproducible settings saved with this result").font(.caption).foregroundStyle(.secondary)
    }
  }
  private var flashControls: some View {
    VStack(alignment: .leading, spacing: 12) {
      HStack {
        Picker("Model", selection: binding(\.variant)) {
          Text("Turbo").tag(FlashVSRSettings.Variant.turbo); Text("Base").tag(FlashVSRSettings.Variant.base)
        }.disabled(store.videoUpscaleSettings.backend == .vpipe)
        Picker("Decoder", selection: binding(\.decoder)) {
          ForEach(FlashVSRSettings.Decoder.allCases, id: \.self) { Text($0.label).tag($0) }
        }
      }
      HStack {
        Text("Output")
        TextField("Width", value: binding(\.width), format: .number).frame(width: 85)
        Text("×"); TextField("Height", value: binding(\.height), format: .number).frame(width: 85)
        Picker("Aspect", selection: binding(\.fit)) { Text("Fit entire image").tag(FlashVSRSettings.Fit.fit); Text("Fill and crop").tag(FlashVSRSettings.Fit.fill) }
        Menu("Presets") {
          Button("720p") { target(1280, 720) }; Button("1080p") { target(1920, 1080) }
          Button("1440p") { target(2560, 1440) }; Button("4K") { target(3840, 2160) }
        }
      }.textFieldStyle(.roundedBorder)
      if let canvas = try? store.videoUpscaleSettings.canvas() {
        Text("Exact output \(canvas.visibleWidth) × \(canvas.visibleHeight) · model canvas \(canvas.width) × \(canvas.height)")
          .font(.caption.monospacedDigit()).foregroundStyle(.secondary)
      }
      HStack {
        if !store.videoUpscalePresets.isEmpty {
          Menu("Saved presets") { ForEach(store.videoUpscalePresets) { preset in Button(preset.name) { store.videoUpscaleSettings = preset.settings; store.videoUpscalePlan = nil } } }
        }
        TextField("Preset name", text: $store.videoUpscalePresetName)
        Button("Save preset") { store.saveVideoUpscalePreset() }.disabled(store.videoUpscalePresetName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
      }
      DisclosureGroup("Restoration settings") {
        VStack(alignment: .leading, spacing: 10) {
          HStack { Text("Seed"); TextField("Seed", value: binding(\.seed), format: .number).frame(width: 150)
            Button("Randomize") { store.videoUpscaleSettings.seed = .random(in: .min ... .max); store.videoUpscalePlan = nil } }
          Picker("Motion", selection: binding(\.temporalCompression)) {
            Text("Normal · 4").tag(4); Text("High motion · 2 · more work").tag(2); Text("Compression 1 · most work").tag(1)
          }
          Picker("Color correction", selection: binding(\.colorFix)) {
            Text("Wavelet").tag(FlashVSRSettings.ColorFix.wavelet); Text("AdaIN").tag(FlashVSRSettings.ColorFix.adain); Text("Off").tag(FlashVSRSettings.ColorFix.off)
          }
          if store.videoUpscaleSettings.colorFix == .wavelet { Stepper("Wavelet levels: \(store.videoUpscaleSettings.waveletLevels)", value: binding(\.waveletLevels), in: 1...8) }
          Toggle("Keep original audio", isOn: binding(\.preserveAudio))
          Text("Native restoration uses stateful streaming and one flow update. The selected model and decoder are fixed for the run.")
            .font(.caption).foregroundStyle(.secondary)
          DisclosureGroup("Historical backend") {
            Picker("Backend", selection: binding(\.backend)) {
              Text("Swift/MLX · native").tag(FlashVSRSettings.Backend.swiftMLX)
              Text("vpipe · saved legacy jobs").tag(FlashVSRSettings.Backend.vpipe)
            }
            if store.videoUpscaleSettings.backend == .vpipe {
              setupRow("vpipe", path: store.videoUpscaleRuntime.vpipePath)
              Picker("Continuity", selection: binding(\.continuity)) {
                Text("Stateful stream").tag(FlashVSRSettings.Continuity.stream); Text("Overlapping clips").tag(FlashVSRSettings.Continuity.clips)
              }
            }
          }
        }.padding(.top, 8)
      }
      if let plan = store.videoUpscalePlan { Text("\(plan.frames) frames · \(plan.media.fps.argument) FPS · temporary disk budget \(plan.temporaryBytes / 1_000_000) MB").font(.caption.monospacedDigit()).foregroundStyle(.secondary) }
    }
  }
  private var setupControls: some View {
    VStack(alignment: .leading, spacing: 10) {
      ModelDownloadAccessView()
      HStack {
        Button("Choose model library…") { store.chooseNativeUpscalerLibrary() }
        Text(store.nativeUpscalerModelLibrary.isEmpty ? "Select your shared model library before downloading." : store.nativeUpscalerModelLibrary)
          .font(.caption).lineLimit(1).truncationMode(.middle).textSelection(.enabled)
      }
      if store.videoUpscaleTool == .flashVSR {
        setupRow("models", path: store.videoUpscaleRuntime.modelRoot)
        Button("Prepare selected FlashVSR+ models") { store.downloadVideoUpscaleModels() }
        if store.videoUpscaleSettings.backend == .swiftMLX { setupRow("Swift worker", path: store.videoUpscaleRuntime.swiftWorkerPath ?? "Bundled native helper") }
      } else {
        Button("Prepare H3 restoration models") { store.prepareH3RegenerateModels() }
        Button("Connect existing Ref2VA profile…") { store.chooseH3RegenerateProfile() }
      }
      setupRow("FFmpeg", path: store.videoUpscaleRuntime.ffmpegPath)
      setupRow("FFprobe", path: store.videoUpscaleRuntime.ffprobePath)
      Text("Setup reuses checksum-verified installed weights before downloading pinned components into the selected library. Native tensor preparation and inference run in Swift/MLX. FFmpeg handles media transport.")
        .font(.caption).foregroundStyle(.secondary)
    }
  }
  private func target(_ width: Int, _ height: Int) { store.videoUpscaleSettings.width = width; store.videoUpscaleSettings.height = height; store.videoUpscalePlan = nil }
  private func setupRow(_ name: String, path: String) -> some View {
    HStack {
      Button("Connect " + name + "…") { store.connectVideoUpscale(name) }.frame(width: 150)
      Text(path.isEmpty ? "Not connected" : path).font(.caption).foregroundStyle(.secondary).lineLimit(1).truncationMode(.middle)
    }
  }
}

private struct H3RegenerateComparison: View {
  let receipt: H3RegenerateReceipt
  @State private var source = AVPlayer()
  @State private var result = AVPlayer()
  var body: some View {
    HStack {
      VStack { Text("Original"); VideoPlayer(player: source) }
      VStack { Text("H3 Regenerate"); VideoPlayer(player: result) }
    }.frame(height: 220).onAppear {
      source.replaceCurrentItem(with: AVPlayerItem(url: URL(fileURLWithPath: receipt.source.path))); source.isMuted = true
      result.replaceCurrentItem(with: AVPlayerItem(url: URL(fileURLWithPath: receipt.video.path)))
    }.onDisappear { source.pause(); result.pause() }
  }
}

private struct VideoUpscalePlayer: NSViewRepresentable {
  let player: AVPlayer
  func makeNSView(context: Context) -> AVPlayerView { let view = AVPlayerView(); view.controlsStyle = .none; view.player = player; return view }
  func updateNSView(_ view: AVPlayerView, context: Context) { view.player = player }
}
private struct VideoUpscaleComparison: View {
  let receipt: FlashVSRReceipt
  let busy: Bool
  @State private var source = AVPlayer()
  @State private var output = AVPlayer()
  @State private var position = 0.0
  @State private var playing = false
  @State private var observer: Any?
  @State private var endObserver: NSObjectProtocol?
  @State private var playbackID = UUID()
  private var duration: Double { Double(receipt.plan.frames) / receipt.plan.media.fps.value }
  var body: some View {
    VStack(alignment: .leading, spacing: 8) {
      HStack {
        VStack { Text("Original"); VideoUpscalePlayer(player: source) }
        VStack { Text(receipt.plan.request.settings.variant.rawValue.capitalized + " · " + receipt.plan.request.settings.decoder.label); VideoUpscalePlayer(player: output) }
      }.frame(height: 210)
      HStack {
        Button(playing ? "Pause comparison" : "Play comparison") {
          if playing { source.pause(); output.pause(); playing = false }
          else { startPlayback() }
        }.disabled(busy)
        Slider(value: $position, in: 0...max(duration, 0.001), onEditingChanged: { editing in
          source.pause(); output.pause(); playing = false
          if !editing { seek() }
        })
        Text(position.formatted(.number.precision(.fractionLength(2))) + " s").font(.caption.monospacedDigit())
      }
    }.onAppear {
      let original = receipt.sourcePreview?.path ?? receipt.plan.source.path
      source.replaceCurrentItem(with: AVPlayerItem(url: URL(fileURLWithPath: original)))
      output.replaceCurrentItem(with: AVPlayerItem(url: URL(fileURLWithPath: receipt.video.path)))
      source.isMuted = true
      source.automaticallyWaitsToMinimizeStalling = false
      output.automaticallyWaitsToMinimizeStalling = false
      observer = output.addPeriodicTimeObserver(forInterval: CMTime(seconds: 0.1, preferredTimescale: 600), queue: .main) { time in
        if playing { position = min(duration, max(0, time.seconds)) }
      }
      endObserver = NotificationCenter.default.addObserver(forName: .AVPlayerItemDidPlayToEndTime,
        object: output.currentItem, queue: .main) { _ in
          source.pause(); output.pause(); playing = false; position = duration
        }
    }.onChange(of: busy) { _, value in if value { source.pause(); output.pause(); playing = false } }
      .onDisappear {
        source.pause(); output.pause(); playing = false
        if let observer { output.removeTimeObserver(observer) }
        if let endObserver { NotificationCenter.default.removeObserver(endObserver) }
      }
  }
  private func startPlayback() {
    if position >= duration - 0.5 / receipt.plan.media.fps.value { position = 0 }
    let offset = receipt.sourcePreview == nil ? Double(receipt.plan.request.firstFrame) / receipt.plan.media.fps.value : 0
    let sourceTime = CMTime(seconds: offset + position, preferredTimescale: 60000)
    let outputTime = CMTime(seconds: position, preferredTimescale: 60000)
    playbackID = UUID(); let id = playbackID; playing = true
    let ready = DispatchGroup()
    ready.enter(); source.seek(to: sourceTime, toleranceBefore: .zero, toleranceAfter: .zero) { _ in ready.leave() }
    ready.enter(); output.seek(to: outputTime, toleranceBefore: .zero, toleranceAfter: .zero) { _ in ready.leave() }
    ready.notify(queue: .main) {
      guard playing, playbackID == id else { return }
      let host = CMTimeAdd(CMClockGetTime(CMClockGetHostTimeClock()), CMTime(seconds: 0.1, preferredTimescale: 60000))
      source.setRate(1, time: sourceTime, atHostTime: host)
      output.setRate(1, time: outputTime, atHostTime: host)
    }
  }
  private func seek() {
    let offset = receipt.sourcePreview == nil ? Double(receipt.plan.request.firstFrame) / receipt.plan.media.fps.value : 0
    source.seek(to: CMTime(seconds: offset + position, preferredTimescale: 600), toleranceBefore: .zero, toleranceAfter: .zero)
    output.seek(to: CMTime(seconds: position, preferredTimescale: 600), toleranceBefore: .zero, toleranceAfter: .zero)
  }
}
