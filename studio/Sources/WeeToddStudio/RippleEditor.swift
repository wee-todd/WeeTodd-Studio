import AppKit
import AVKit
import StudioCore
import SwiftUI

struct RippleEditor: View {
  @EnvironmentObject var store: StudioStore
  @State private var previewPlayer = AVPlayer()
  @State private var previewPath = ""
  @State private var sourcePlayer = AVPlayer()
  @State private var sourceIdentity = ""
  @State private var sourceFrame: Double = 0
  @State private var editingReference: RippleEditorReference?
  @State private var sourcePlaying = false
  private let playbackClock = Timer.publish(every: 0.04, on: .main, in: .common).autoconnect()
  var body: some View {
    VStack(spacing: 0) {
      HStack {
        Button { previewPlayer.pause(); sourcePlayer.pause(); store.rippleClipID = nil } label: {
          Label("Back to movie", systemImage: "arrow.left")
        }.keyboardShortcut(.escape, modifiers: [])
        Text("LTX 2.5 Ripple Director · \(store.rippleClip?.name ?? "Clip")").font(.title2)
        Spacer()
        Button("Runtime Settings…") { store.showRuntime = true }
      }.padding(20)
      Divider()
      if let clip = store.rippleClip, let draft = clip.rippleDraft {
        GeometryReader { geometry in
          VSplitView {
            sourceDirector(draft)
              .frame(minHeight: 330, idealHeight: max(330, geometry.size.height * 0.55), maxHeight: .infinity)
            HSplitView {
              ScrollView {
                VStack(alignment: .leading, spacing: 18) {
                  Text("Restyle a video with edited source frames").font(.title3)
                  Text("Extract a source frame, edit it in your image editor, then import the restyled image into its matching slot. The first frame is required; add up to eight other images, each assigned to a different frame.")
                    .font(.callout).foregroundStyle(.secondary)
                  Text("The author workflow uses one guide at frame 0 with strength 1. Other frame placements and multiple guides are Studio’s experimental timed-reference extension; nine-guide visual quality has not been qualified.")
                    .font(.caption).foregroundStyle(.secondary)
                  sourceControls(clip, draft)
                  Divider()
                  ForEach(Array(draft.references.enumerated()), id: \.element.id) { number, reference in
                    RippleReferenceRow(reference: reference, number: number + 1, draft: draft, editImage: {
                      Task { await openReference(reference.id) }
                    })
                    Divider()
                  }
                  Button("Add edited-frame slot") {
                    store.updateRipple { value in
                      let occupied = Set(value.references.map(\.frame))
                      if let frame = (0..<value.frameCount).first(where: { !occupied.contains($0) }) {
                        value.references.append(RippleReference(frame: frame))
                      }
                    }
                  }.disabled(draft.references.count >= 9 || draft.references.count >= draft.frameCount || store.operationBusy)
                }.padding(24)
              }.frame(minWidth: 530, maxWidth: .infinity)
              ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                  Text("Render settings").font(.headline)
                  Text("Native LTX 2.5 · Ripple adapter").font(.caption).foregroundStyle(.secondary)
                  TextField("Optional direction for the restyled clip", text: field(\.prompt), axis: .vertical)
                    .lineLimit(4...10).textFieldStyle(.roundedBorder)
                  HStack {
                    Text("Width"); TextField("Width", value: field(\.width), format: .number)
                    Text("Height"); TextField("Height", value: field(\.height), format: .number)
                  }.textFieldStyle(.roundedBorder)
                  Text("Resolution: multiples of 32, up to 1920 per side.").font(.caption).foregroundStyle(.secondary)
                  HStack {
                    Text("Seed"); TextField("Seed", value: field(\.seed), format: .number)
                    Button { store.updateRipple { $0.seed = Int.random(in: 0...Int(Int32.max)) } } label: { Image(systemName: "dice") }
                  }.textFieldStyle(.roundedBorder)
                  HStack {
                    Text("Ripple LoRA strength")
                    TextField("Adapter strength", value: field(\.loraStrength), format: .number).textFieldStyle(.roundedBorder)
                  }
                  Text("Author baseline: strength 1.35 · eight-step full-resolution render.").font(.caption).foregroundStyle(.secondary)
                  Picker("Audio", selection: field(\.audioPolicy)) {
                    Text("Preserve source audio").tag(RippleAudioPolicy.preserve)
                    Text("Silent output").tag(RippleAudioPolicy.silent)
                  }
                  Text("Silent input is supported. Preserve copies the original interval’s audio; it does not generate a replacement soundtrack.").font(.caption).foregroundStyle(.secondary)
                  if let validation = validationMessage(draft) {
                    Label(validation, systemImage: "info.circle").font(.caption).foregroundStyle(.secondary)
                  }
                  RippleProgress(bridge: store.bridge)
                  Button("Generate new Ripple take") { Task { await store.generateRipple() } }
                    .buttonStyle(.borderedProminent)
                    .disabled(store.operationBusy || validationMessage(draft) != nil)
                  takeReview(clip)
                }.padding(24)
              }.frame(minWidth: 350, idealWidth: 430, maxWidth: 520)
            }.frame(minHeight: 240, idealHeight: max(240, geometry.size.height * 0.45), maxHeight: .infinity)
          }
        }
      }
    }.background(Theme.background)
      .onDisappear {
        previewPlayer.pause(); previewPlayer.replaceCurrentItem(with: nil)
        sourcePlayer.pause(); sourcePlayer.replaceCurrentItem(with: nil)
      }
      .sheet(item: $editingReference) { reference in
        RippleReferenceGenerator(clipID: reference.clipID, referenceID: reference.id)
          .environmentObject(store)
      }
      .onReceive(playbackClock) { _ in
        guard sourcePlaying, let draft = store.rippleClip?.rippleDraft else { return }
        let seconds = sourcePlayer.currentTime().seconds - (draft.sourcePreviewStart ?? draft.sourceIn)
        guard seconds.isFinite else { return }
        if seconds >= draft.duration { sourcePlayer.pause(); sourcePlaying = false }
        sourceFrame = min(Double(max(0, draft.frameCount - 1)), max(0, floor(seconds * draft.frameRate)))
      }
      .task(id: store.rippleClipID) {
        await store.inspectRipple()
        if let first = store.rippleClip?.rippleDraft?.references.first(where: { $0.frame == 0 }),
          first.originalPath.isEmpty, store.rippleInspection != nil {
          await store.extractRippleFrame(referenceID: first.id)
        }
      }
  }

  private func sourceDirector(_ draft: RippleDraft) -> some View {
    VStack(spacing: 10) {
      NativePlayer(player: sourcePlayer).frame(minHeight: 200, idealHeight: 420, maxHeight: .infinity)
        .task(id: "\(draft.sourcePath)|\(draft.sourcePreviewStart ?? draft.sourceIn)|\(draft.duration)|\(draft.frameRate)") {
          let identity = "\(draft.sourcePath)|\(draft.sourcePreviewStart ?? draft.sourceIn)|\(draft.duration)|\(draft.frameRate)"
          guard identity != sourceIdentity else { return }
          sourceIdentity = identity; sourceFrame = 0; sourcePlaying = false
          sourcePlayer.replaceCurrentItem(with: AVPlayerItem(url: URL(fileURLWithPath: draft.sourcePath)))
          seekSource(frame: 0, draft: draft)
        }
      VStack(spacing: 6) {
        HStack {
          Button {
            if sourcePlaying { sourcePlayer.pause() } else {
              seekSource(frame: Int(sourceFrame), draft: draft); sourcePlayer.play()
            }
            sourcePlaying.toggle()
          } label: { Image(systemName: sourcePlaying ? "pause.fill" : "play.fill") }
          Button { scrubSource(max(0, Int(sourceFrame) - 1), draft) } label: { Image(systemName: "backward.frame") }
          Button { scrubSource(min(draft.frameCount - 1, Int(sourceFrame) + 1), draft) } label: { Image(systemName: "forward.frame") }
          Text(String(format: "Frame %d / %d · %.3f s", Int(sourceFrame), max(0, draft.frameCount - 1), sourceFrame / draft.frameRate))
            .font(.caption).monospacedDigit()
          Spacer()
          Text("\(draft.references.count) / 9 references").font(.caption)
          Button("Add reference at playhead…") {
            sourcePlayer.pause(); sourcePlaying = false
            Task {
              do {
                let id = try store.addRippleReference(frame: Int(sourceFrame))
                await openReference(id)
              } catch { store.error = error.localizedDescription }
            }
          }.buttonStyle(.borderedProminent).disabled(store.operationBusy || draft.frameCount == 0)
        }
        Slider(value: Binding(get: { sourceFrame }, set: { scrubSource(Int($0.rounded()), draft) }),
          in: 0...Double(max(1, draft.frameCount - 1)), step: 1)
          .accessibilityLabel("Ripple source frame timeline")
        GeometryReader { geometry in
          ZStack(alignment: .leading) {
            Rectangle().fill(Theme.line).frame(height: 1)
            ForEach(Array(draft.references.sorted { $0.frame < $1.frame }.enumerated()), id: \.element.id) { index, reference in
              Button {
                scrubSource(reference.frame, draft)
                Task { await openReference(reference.id) }
              } label: {
                VStack(spacing: 0) {
                  CachedImageThumbnail(path: reference.path.isEmpty ? reference.originalPath : reference.path,
                    maximumPixelSize: 144).frame(width: 64, height: 40).clipped()
                  Text("\(reference.frame == 0 ? "First" : "Guide \(index + 1)") · \(reference.frame)")
                    .font(.system(size: 8, weight: .medium)).lineLimit(1)
                }.frame(width: 68, height: 54).background(Theme.raised, in: RoundedRectangle(cornerRadius: 4))
                  .overlay(RoundedRectangle(cornerRadius: 4).strokeBorder(Color.accentColor.opacity(0.6)))
              }.buttonStyle(.plain).disabled(store.operationBusy)
                .offset(x: min(max(CGFloat(index) * 72,
                  CGFloat(reference.frame) / CGFloat(max(1, draft.frameCount - 1)) * (geometry.size.width - 68)),
                  geometry.size.width - 68 - CGFloat(draft.references.count - 1 - index) * 72))
                .help("Edit guide \(index + 1) · source frame \(reference.frame)")
                .contextMenu {
                  Button("Edit reference frame…") { Task { await openReference(reference.id) } }
                  Button("Delete reference frame clip", role: .destructive) { store.removeRippleReference(reference.id) }
                    .disabled(reference.frame == 0 || store.operationBusy)
                  if reference.frame == 0 { Text("The first frame is required") }
                }
            }
          }
        }.frame(height: 58)
      }.padding(.horizontal, 24).padding(.bottom, 8)
    }
  }

  private func seekSource(frame: Int, draft: RippleDraft) {
    sourcePlayer.seek(to: CMTime(seconds: (draft.sourcePreviewStart ?? draft.sourceIn) + Double(frame) / draft.frameRate, preferredTimescale: 60000),
      toleranceBefore: .zero, toleranceAfter: .zero)
  }
  private func scrubSource(_ frame: Int, _ draft: RippleDraft) {
    sourcePlayer.pause(); sourcePlaying = false
    sourceFrame = Double(max(0, min(frame, draft.frameCount - 1)))
    seekSource(frame: Int(sourceFrame), draft: draft)
  }
  private func openReference(_ id: UUID) async {
    sourcePlayer.pause(); sourcePlaying = false
    guard let clip = store.rippleClip, let draft = clip.rippleDraft,
      let reference = draft.references.first(where: { $0.id == id }) else { return }
    let session = store.documentSessionID
    if reference.originalPath.isEmpty { await store.extractRippleFrame(referenceID: id) }
    guard store.documentSessionID == session, store.rippleClipID == clip.id,
      let current = store.rippleClip?.rippleDraft?.references.first(where: { $0.id == id }),
      !current.originalPath.isEmpty, current.frame == reference.frame else { return }
    editingReference = RippleEditorReference(id: id, clipID: clip.id)
  }

  private func field<T>(_ key: WritableKeyPath<RippleDraft, T>) -> Binding<T> {
    let fallback = store.rippleClip!.rippleDraft![keyPath: key]
    return Binding(get: { store.rippleClip?.rippleDraft?[keyPath: key] ?? fallback },
      set: { value in store.updateRipple { $0[keyPath: key] = value } })
  }

  private func validationMessage(_ draft: RippleDraft) -> String? {
    do { try draft.validate(); return nil } catch { return error.localizedDescription }
  }

  @ViewBuilder private func sourceControls(_ clip: Clip, _ draft: RippleDraft) -> some View {
    VStack(alignment: .leading, spacing: 8) {
      Text("Source: \(URL(fileURLWithPath: draft.sourcePath).lastPathComponent)").font(.headline)
      Text(String(format: "Trim %.3f–%.3f s · %.3f fps · frames 0–%d", draft.sourceIn,
        draft.sourceIn + draft.duration, draft.frameRate, max(0, draft.frameCount - 1)))
        .font(.caption).monospacedDigit()
      Text("Frame numbers and timestamps are relative to this captured source interval. The sampling rate is fixed after images are assigned.")
        .font(.caption).foregroundStyle(.secondary)
      if let inspection = store.rippleInspection {
        Label(inspection["has_audio"] as? Bool == true ? "Source audio available" : "Silent source · supported",
          systemImage: inspection["has_audio"] as? Bool == true ? "waveform" : "speaker.slash")
          .font(.caption)
      }
      HStack {
        Button("Inspect source") { Task { await store.inspectRipple() } }.disabled(store.operationBusy)
        Button("Open source video") { NSWorkspace.shared.open(URL(fileURLWithPath: draft.sourcePath)) }
      }
      if !draft.sourceMatches(clip) {
        Label("The timeline source or trim changed. This draft still uses its captured original interval.", systemImage: "info.circle")
          .font(.caption).foregroundStyle(.orange)
      }
      Button("Start new draft from current timeline clip") {
        store.restartRippleFromCurrentClip()
        Task {
          await store.inspectRipple()
          if let first = store.rippleClip?.rippleDraft?.references.first, store.rippleInspection != nil {
            await store.extractRippleFrame(referenceID: first.id)
          }
        }
      }.disabled(store.operationBusy)
    }
  }

  @ViewBuilder private func takeReview(_ clip: Clip) -> some View {
    if let takes = clip.rippleTakes, !takes.isEmpty {
      Divider()
      Text("Review takes").font(.headline)
      ForEach(takes.reversed()) { take in
        HStack {
          Button {
            store.rippleSelectedTakeID = take.id
            previewPath = take.path
            previewPlayer.replaceCurrentItem(with: AVPlayerItem(url: URL(fileURLWithPath: take.path)))
          } label: {
            Label(take.created.formatted(date: .omitted, time: .shortened), systemImage: "play.rectangle")
          }
          Text("\(take.draft.references.count) guide\(take.draft.references.count == 1 ? "" : "s") · \(take.hasAudio ? "source audio" : "silent")")
            .font(.caption).foregroundStyle(.secondary)
        }
      }
      if let take = takes.first(where: { $0.id == store.rippleSelectedTakeID }) {
        VideoPlayer(player: previewPlayer).frame(height: 200)
          .onAppear { setPreview(take.path) }
          .onChange(of: take.path) { _, value in setPreview(value) }
        HStack {
          Button("Apply take") { previewPlayer.pause(); store.applyRipple(take) }
            .buttonStyle(.borderedProminent).disabled(store.operationBusy || !store.canApplyRipple(take, to: clip))
          Button("Reveal files") { NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: take.path)]) }
        }
        if !store.canApplyRipple(take, to: clip) {
          Text(clip.sourcePath == take.path ? "This take is on the timeline." : "Source or settings changed since this take was generated.")
            .font(.caption).foregroundStyle(.secondary)
          Button("Restore original source and this take’s inputs") { store.restoreRippleTakeInputs(take) }
            .disabled(store.operationBusy)
        }
      }
    }
  }
  private func setPreview(_ path: String) {
    guard previewPath != path else { return }
    previewPath = path
    previewPlayer.replaceCurrentItem(with: AVPlayerItem(url: URL(fileURLWithPath: path)))
  }
}

private struct RippleEditorReference: Identifiable { let id: UUID; let clipID: UUID }

private struct RippleProgress: View {
  @ObservedObject var bridge: Bridge
  var body: some View {
    if bridge.busy {
      HStack { ProgressView().controlSize(.small); Text(bridge.message).font(.caption); Button("Cancel") { bridge.cancel() } }
    }
  }
}

private struct RippleReferenceRow: View {
  @EnvironmentObject var store: StudioStore
  let reference: RippleReference
  let number: Int
  let draft: RippleDraft
  var editImage: () -> Void
  private func edit(_ body: (inout RippleReference) -> Void) {
    store.updateRipple { draft in
      if let index = draft.references.firstIndex(where: { $0.id == reference.id }) { body(&draft.references[index]) }
    }
  }
  var body: some View {
    VStack(alignment: .leading, spacing: 10) {
      HStack {
        Text(reference.frame == 0 ? "First frame · required" : "Guide \(number)").font(.headline)
        Text("Frame")
        TextField("Frame", value: Binding(get: { reference.frame }, set: { frame in
          do { try store.setRippleReferenceFrame(reference.id, frame: frame) }
          catch { store.error = error.localizedDescription }
        }), format: .number).frame(width: 90).textFieldStyle(.roundedBorder).disabled(reference.frame == 0)
          .help("Changing the frame clears the old original/edited pair so an edit cannot silently move to a different source frame.")
        Text(String(format: "%.3f s", Double(reference.frame) / draft.frameRate)).font(.caption).monospacedDigit()
        Spacer()
        Button { store.removeRippleReference(reference.id) } label: {
          Image(systemName: "trash")
        }.disabled(reference.frame == 0)
      }
      HStack(alignment: .top, spacing: 14) {
        VStack {
          image(reference.originalPath, placeholder: "Original source frame")
          Button("Extract source frame") { Task { await store.extractRippleFrame(referenceID: reference.id) } }
          if !reference.originalPath.isEmpty {
            Button("Reveal for editing") { NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: reference.originalPath)]) }
              .font(.caption)
          }
        }
        VStack {
          image(reference.path, placeholder: "Edited / restyled image")
          Button("Edit with Draw Things…", action: editImage)
          Button(reference.path.isEmpty ? "Import edited image…" : "Replace edited image…") {
            store.importRippleImage(referenceID: reference.id)
          }
        }
      }
      HStack {
        Text("Image strength").font(.caption)
        Slider(value: Binding(get: { reference.strength }, set: { value in edit { $0.strength = value } }), in: 0...1)
        Text(reference.strength, format: .number.precision(.fractionLength(2))).font(.caption).monospacedDigit()
      }
      if draft.references.filter({ $0.frame == reference.frame }).count > 1 {
        Text("This frame is already assigned. Each guide needs a distinct frame.").font(.caption).foregroundStyle(.orange)
      }
    }.disabled(store.operationBusy)
  }
  private func image(_ path: String, placeholder: String) -> some View {
    VStack {
      if path.isEmpty { Image(systemName: "photo").font(.title); Text(placeholder).font(.caption) }
      else { CachedImageThumbnail(path: path, maximumPixelSize: 640).help(URL(fileURLWithPath: path).lastPathComponent) }
    }.frame(maxWidth: .infinity).frame(height: 155).background(Theme.raised, in: RoundedRectangle(cornerRadius: 8))
  }
}

struct RippleRuntimeSettings: View {
  @EnvironmentObject var store: StudioStore
  var body: some View {
    VStack(alignment: .leading, spacing: 10) {
      Text("LTX Ripple video restyling").font(.headline)
      PathPicker(label: "Installed Ripple adapter (.safetensors)", value: Binding(
        get: { store.runtime.rippleAdapterPath ?? "" }, set: { store.runtime.rippleAdapterPath = $0 }))
      Picker("LTX 2.5 components", selection: Binding(
        get: { store.runtime.rippleProfileID ?? "auto" }, set: { store.runtime.rippleProfileID = $0 })) {
        Text("Automatic · compatible installed distilled profile").tag("auto")
        ForEach(store.profiles.filter { $0.engine == Engine.ltx25.rawValue }) { profile in
          Text(profile.name).tag(profile.id)
        }
        if let saved = store.runtime.rippleProfileID, saved != "auto", !saved.isEmpty,
          !store.profiles.contains(where: { $0.id == saved && $0.engine == Engine.ltx25.rawValue }) {
          Text("Unavailable saved profile · choose an installed profile").tag(saved)
        }
      }
      Text("Uses your installed LTX 2.5 components and a separately installed Ripple IC-LoRA. Compatibility is checked before weights load. Open Ripple from an existing video clip to assign edited frames.")
        .font(.caption).foregroundStyle(.secondary)
    }
  }
}

/// The clip inspector exposes the same saved draft used by the Director.
struct RippleClipInspector: View {
  @EnvironmentObject var store: StudioStore
  let clip: Clip
  var body: some View {
    DisclosureGroup("LTX 2.5 Ripple") {
      VStack(alignment: .leading, spacing: 8) {
        Button("Open Ripple Director…") { store.openRipple() }
        if let draft = clip.rippleDraft {
          Text("\(draft.references.filter { !$0.path.isEmpty }.count) edited images · \(draft.width) × \(draft.height)")
            .font(.caption).foregroundStyle(.secondary)
          TextField("Optional restyling direction", text: binding(\.prompt), axis: .vertical)
            .lineLimit(2...4).textFieldStyle(.roundedBorder)
          Picker("Audio", selection: binding(\.audioPolicy)) {
            Text("Preserve source").tag(RippleAudioPolicy.preserve)
            Text("Silent").tag(RippleAudioPolicy.silent)
          }
          HStack { Text("Seed"); TextField("Seed", value: binding(\.seed), format: .number) }
            .textFieldStyle(.roundedBorder)
          HStack { Text("Ripple strength"); TextField("Strength", value: binding(\.loraStrength), format: .number) }
            .textFieldStyle(.roundedBorder)
          Button("Generate Ripple take") { Task { await store.generateRipple(clipID: clip.id) } }
            .disabled(store.operationBusy || (try? draft.validate()) == nil)
          if let count = clip.rippleTakes?.count, count > 0 {
            Text("\(count) saved take\(count == 1 ? "" : "s"). Open the Director to review and apply.").font(.caption)
          }
        } else {
          Text("Restyle this clip with one to nine edited source frames. Configure images in the Director.")
            .font(.caption).foregroundStyle(.secondary)
        }
      }.padding(.top, 6)
    }.font(.caption)
  }
  private func binding<T>(_ key: WritableKeyPath<RippleDraft, T>) -> Binding<T> {
    let fallback = clip.rippleDraft![keyPath: key]
    return Binding(get: { store.project.clips.first(where: { $0.id == clip.id })?.rippleDraft?[keyPath: key] ?? fallback }, set: { value in
      guard let index = store.project.clips.firstIndex(where: { $0.id == clip.id }) else { return }
      store.change { $0.clips[index].rippleDraft?[keyPath: key] = value }
    })
  }
}
