import StudioCore
import SwiftUI

struct InspectorColumn: View {
  @EnvironmentObject var store: StudioStore
  var body: some View {
    VStack(spacing: 0) {
      PanelHeading(
        title: "CLIP", icon: "slider.horizontal.3",
        trailing: store.selectedClip.map { $0.engine.rawValue.uppercased() } ?? "—")
      ScrollView {
        VStack(alignment: .leading, spacing: 16) {
          if store.selectedTitleID != nil {
            TitleInspector()
          } else if store.selectedAudioID != nil {
            AudioInspector()
          } else if store.selectedTrackID != nil {
            TrackInspector()
          } else if let clip = store.selectedClip {
            ClipInspector(clip: clip)
          } else {
            Text("Select a clip to shape its shot, timing and references.").font(.system(size: 12))
              .foregroundStyle(.secondary).padding(.vertical, 20)
          }
        }.padding(16)
      }.frame(maxHeight: .infinity)
      Divider().overlay(Theme.line)
      PanelHeading(title: "MOVIE / PROJECT", icon: "film.stack", trailing: "DEFAULTS")
      ScrollView {
        MovieSettingsForm(
          settings: Binding(
            get: { store.project.settings }, set: { v in store.change { $0.settings = v } })
        ).padding(16)
      }.frame(height: 300)
    }.background(Theme.panel)
  }
}
struct ClipInspector: View {
  @EnvironmentObject var store: StudioStore
  var clip: Clip
  @FocusState private var seedFocused: Bool
  @State private var seedUndoGroup = UUID()
  func binding<T>(_ key: WritableKeyPath<Clip, T>) -> Binding<T> {
    Binding(
      get: { store.selectedClip?[keyPath: key] ?? clip[keyPath: key] },
      set: { v in store.editClip { $0[keyPath: key] = v } })
  }
  var geometryAndSeed: some View {
    VStack(alignment: .leading, spacing: 10) {
    HStack {
      Text("Render size")
      TextField("Width", value: binding(\.generationWidth), format: .number)
      Text("×")
      TextField("Height", value: binding(\.generationHeight), format: .number)
    }.textFieldStyle(.roundedBorder)
    HStack {
      Text("Seed")
      TextField(
        "Seed",
        value: Binding(
          get: { store.selectedClip?.seed ?? clip.seed },
          set: { value in
            store.editClip(undoGroup: seedFocused ? seedUndoGroup : nil) { $0.seed = value }
          }), format: .number
      ).focused($seedFocused)
        .onChange(of: seedFocused) { _, focused in
          if focused { seedUndoGroup = UUID() }
        }
      Button {
        store.editClip { $0.seed = Int.random(in: 0...Int(Int32.max)) }
      } label: {
        Image(systemName: "dice")
      }
    }.textFieldStyle(.roundedBorder)
    }.font(.caption)
  }
  var body: some View {
    VStack(alignment: .leading, spacing: 14) {
      TextField("Clip name", text: binding(\.name)).font(.system(size: 16, weight: .semibold))
        .textFieldStyle(.plain)
      HStack {
        Circle().fill(Theme.engine(clip.engine)).frame(width: 6, height: 6)
        Text(clip.displayTask).font(.system(size: 10)).foregroundStyle(.secondary)
      }
      if !clip.sourcePath.isEmpty { RippleClipInspector(clip: clip) }
      if clip.engine != .movie {
        field("GENERATION") {
          Picker("Generation", selection: Binding(get: { clip.generationProvider }, set: { provider in
            store.editClip { $0.selectGenerationProvider(provider) }
          })) {
            ForEach(GenerationProvider.allCases) {
              Text($0.label).tag($0)
            }
          }.labelsHidden()
        }
        if clip.engine == .drawThings {
          DrawThingsClipInspector(clip: clip)
        } else {
          GenerationInspector(clip: clip)
          ContinuityInspector(clip: clip)
        }
        geometryAndSeed
        Button {
          store.showPrompt = true
        } label: {
          HStack {
            Image(systemName: "text.bubble")
            Text(clip.prompt.isEmpty ? "Write your shot" : "Edit prompt")
            Spacer()
            Image(systemName: "arrow.up.left.and.arrow.down.right")
          }
        }
        .buttonStyle(.bordered).controlSize(.large)
        if !clip.prompt.isEmpty {
          Text(clip.prompt).font(.system(size: 11)).foregroundStyle(.secondary).lineLimit(3)
        }
      }
      HStack {
        field("DURATION · SEC") {
          TextField(
            "Duration", value: binding(\.duration), format: .number.precision(.fractionLength(2))
          ).textFieldStyle(.roundedBorder)
        }
        field("IN POINT · SEC") {
          TextField(
            "Source in", value: binding(\.sourceIn), format: .number.precision(.fractionLength(2))
          ).textFieldStyle(.roundedBorder).disabled(clip.sourcePath.isEmpty)
        }
      }
      HStack {
        Text("Source audio").font(.system(size: 11))
        Slider(value: binding(\.volume), in: 0...2)
        Text("\(Int(clip.volume*100))%").font(.system(size: 10, design: .monospaced)).frame(
          width: 32)
      }
      HStack {
        Text("Source pan / balance").font(.caption)
        Slider(value: Binding(get: { clip.sourcePan ?? 0 }, set: { value in store.editClip { $0.sourcePan = value } }), in: -1...1)
      }
      AudioDriverInspector()
      field("INCOMING TRANSITION") {
        Picker("Transition", selection: binding(\.transition)) {
          Text("Cut").tag("cut")
          Text("Cross dissolve").tag("dissolve")
          Text("Fade through black").tag("fadeBlack")
          Text("Wipe left").tag("wipe")
        }.labelsHidden()
        if clip.transition != "cut" {
          HStack {
            Text("Seconds").font(.caption)
            TextField("Transition duration", value: binding(\.transitionDuration), format: .number)
              .textFieldStyle(.roundedBorder)
          }
        }
      }
      Divider()
      HStack {
        SmallLabel(text: "Conditioning")
        Spacer()
        Text("\(clip.attachments.filter { $0.role != .lora }.count)").font(.caption.monospacedDigit()).foregroundStyle(
          .secondary)
      }
      if !clip.attachments.contains(where: { $0.role != .lora }) {
        Text(
          clip.engine == .drawThings
            ? "Select an image in Media & Assets, then Use in clip → First frame. H3 FL2VA also supports Last frame: it is sent as Draw Things’ first enabled mood-board image. Both images are center-cropped to the clip dimensions. LTX currently supports First frame only."
            : clip.engine == .ltx25
              ? "Add a first or last frame from Media & Assets. Reference and control tasks require their compatible model adapters, which can be installed in model setup."
              : "Use an asset as a first frame, reference, audio driver or control. The selected model exposes compatible tasks and settings."
        ).font(.system(size: 11)).foregroundStyle(.secondary)
      }
      ForEach(clip.attachments.filter { $0.role != .lora }) { a in AttachmentRow(attachment: a) }
      if clip.engine != .movie && clip.engine != .drawThings { ClipLoRAInspector(clip: clip) }
      if !clip.sourcePath.isEmpty {
        HStack {
          Button("Split") { store.split() }
          Button("Duplicate") { store.duplicateClip() }
          Menu("Extend") {
            Button("After") { store.extend("after") }
            Button("Before · LTX 2.3") { store.extend("before") }.disabled(clip.engine != .ltx23)
          }.disabled(clip.engine == .drawThings)
        }.controlSize(.small)
      }
      if clip.engine != .drawThings { MotionFidelityInspector(clip: clip) }
      DisclosureGroup("Advanced generation") {
        VStack(alignment: .leading, spacing: 10) {
          if clip.engine != .movie {
            if clip.engine != .drawThings {
              Picker("Execution preset", selection: Binding(get: { clip.generationSelection?.preset ?? .custom }, set: { preset in
                store.editClip { $0.selectGenerationPreset(preset) }
              })) {
                ForEach(GenerationPreset.allCases) { Text($0.label).tag($0) }
              }
              Picker("Custom recipe", selection: Binding(get: { clip.profileID }, set: { value in
                store.editClip {
                  $0.profileID = value
                  $0.generationSelection = GenerationSelection(task: $0.inferredTask, preset: .custom)
                }
              })) {
                Text("Automatic").tag("auto")
                ForEach(store.profiles.filter { $0.engine == clip.engine.rawValue }) {
                  Text($0.name).tag($0.id)
                }
              }
            }
            if clip.engine == .h3 {
              Picker("H3 page cache", selection: binding(\.h3PagingCacheGB)) {
                Text("Recipe default").tag(Optional<Double>.none)
                Text("Off").tag(Optional(0.0))
                ForEach([4.0, 8, 12, 16], id: \.self) { value in
                  Text("\(Int(value)) GB").tag(Optional(value))
                }
              }
              Text("Experimental: retains extra weight pages between denoising steps. This is a cache budget, not a total RAM cap. Off is the default.")
                .font(.caption2).foregroundStyle(.secondary)
            }
            Text(
              "Generation stays on the model grid. Finishing applies the movie canvas and frame rate to each clip."
            ).font(.caption2).foregroundStyle(.secondary)
          }
          Toggle(
            "Override movie settings",
            isOn: Binding(
              get: { clip.settingsOverride != nil },
              set: { v in store.editClip { $0.settingsOverride = v ? store.project.settings : nil }
              }))
          if clip.settingsOverride != nil {
            MovieSettingsForm(
              settings: Binding(
                get: { store.selectedClip?.settingsOverride ?? store.project.settings },
                set: { v in store.editClip { $0.settingsOverride = v } }))
          }
          DisclosureGroup("MetalFX interpolation guides") {
            Text(
              "Experimental: matching per-frame float32 depth (R32) and backward pixel motion (RG32) folders. Ordinary movies can use RIFE."
            ).font(.caption2).foregroundStyle(.secondary)
            PathPicker(label: "Depth folder", value: binding(\.depthDirectory), directory: true)
            PathPicker(label: "Motion folder", value: binding(\.motionDirectory), directory: true)
          }
        }.font(.system(size: 11)).padding(.top, 8)
      }.font(.system(size: 11))
      if !clip.versions.isEmpty {
        if let selectedVersion = clip.versions.last(where: { $0.path == clip.sourcePath }) {
          RenderStatsView(stats: selectedVersion.stats)
        }
        DisclosureGroup("Versions · \(clip.versions.count)") {
          ForEach(clip.versions.reversed()) { v in
            Button {
              store.activateRenderVersion(v, for: clip)
            } label: {
              HStack {
                Image(systemName: "play.rectangle")
                Text(v.created, style: .time)
                Spacer()
                Text("Seed \(v.seed)").foregroundStyle(.secondary)
              }
            }.font(.caption)
            if let settings = v.generationSettings {
              Text("Steps: \(settings.controls.evaluations.map(String.init) ?? "custom")"
                + (settings.controls.refinementSteps.map { " · refinement: \($0)" } ?? ""))
                .font(.caption2).foregroundStyle(.secondary)
            }
            RenderStatsView(stats: v.stats)
          }
        }.font(.system(size: 11))
      }
    }.pickerStyle(.menu)
  }
  func field<Content: View>(_ title: String, @ViewBuilder content: () -> Content) -> some View {
    VStack(alignment: .leading, spacing: 6) {
      SmallLabel(text: title)
      content()
    }
  }
}
struct AttachmentRow: View {
  @EnvironmentObject var store: StudioStore
  var attachment: Attachment
  private var isDrawThings: Bool { store.selectedClip?.engine == .drawThings }
  private var guideTypes: [(String, String)] {
    let kind = store.allAssets.first { $0.id == attachment.assetID }?.kind
    let engine = store.selectedClip?.engine
    if kind == .image { return engine == .ltx25 ? [("Ingredients reference sheet", "ingredients_reference_sheet")] : [] }
    guard kind == .video || kind == .sequence else { return [] }
    var values = [("Canny edges", "canny_edges"), ("Depth", "depth_map"), ("Pose", "pose_skeleton")]
    if engine == .h3 { values += [("HED edges", "hed_edges"), ("MLSD lines", "mlsd_lines")] }
    else { values += [("Motion tracks", "motion_track")] }
    if engine == .ltx25 { values += [("Crossview warp", "crossview_warp")] }
    return values
  }
  private var supportedDrawThingsInput: Bool {
    guard let clip = store.selectedClip,
      let asset = store.allAssets.first(where: { $0.id == attachment.assetID }) else { return false }
    return clip.canAssignDrawThingsInput(asset, role: attachment.role)
  }
  private var availableRoles: [MediaRole] {
    MediaRole.allCases.filter { role in
      guard role != .lora else { return false }
      guard let clip = store.selectedClip,
        let asset = store.allAssets.first(where: { $0.id == attachment.assetID }) else { return role == attachment.role }
      return role == attachment.role || clip.canAssignMedia(asset, role: role)
    }
  }
  var body: some View {
    VStack(alignment: .leading, spacing: 5) {
      HStack {
        Image(systemName: "link").foregroundStyle(Theme.mint)
        Text(store.allAssets.first { $0.id == attachment.assetID }?.name ?? "Missing asset")
          .lineLimit(1)
        Spacer()
        Button {
          if [.first, .last].contains(attachment.role), let id = store.selectedClipID {
            store.removeEndpoint(from: id, role: attachment.role)
          } else {
            store.editClip { $0.attachments.removeAll { $0.id == attachment.id } }
          }
        } label: {
          Image(systemName: "xmark")
        }
      }.font(.system(size: 10))
      Picker(
        "Role", selection: Binding(get: { attachment.role }, set: { v in
          store.editClip { clip in
            guard let index = clip.attachments.firstIndex(where: { $0.id == attachment.id }),
              let asset = store.allAssets.first(where: { $0.id == attachment.assetID }),
              clip.canAssignMedia(asset, role: v) else { return }
            clip.attachments[index].role = v
            if let action = clip.referenceActions(for: asset).first(where: { $0.role == v && $0.preparation == nil }) {
              clip.attachments[index].controlType = action.controlType
            }
            clip.generationSelection?.task = v == .audioDriver ? "a2v" : v == .control ? "control"
              : v == .reference ? "ref2va" : "fflf"
          }
        })
      ) { ForEach(availableRoles) { Text($0.label).tag($0) } }
      .labelsHidden()
      if isDrawThings && !supportedDrawThingsInput {
        Text("Unsupported Draw Things input. Remove this attachment with ×; its media stays in Clip Assets.")
          .font(.caption2).foregroundStyle(.orange)
      }
      if attachment.role == .audioDriver && !isDrawThings {
        HStack {
          Text("Song start (s)")
          TextField("Start", value: Binding(get: { attachment.audioSourceStart ?? 0 }, set: { value in
            edit { $0.audioSourceStart = value; if $0.audioSourceDuration == nil { $0.audioSourceDuration = store.selectedClip?.duration } }
          }), format: .number)
        }.font(.caption)
        HStack {
          Text("Song duration (s)")
          TextField("Duration", value: Binding(get: { attachment.audioSourceDuration ?? store.selectedClip?.duration ?? 0 }, set: { value in
            edit { $0.audioSourceDuration = value; if $0.audioSourceStart == nil { $0.audioSourceStart = 0 } }
          }), format: .number)
        }.font(.caption)
        Button("Use clip length") { edit { $0.audioSourceStart = $0.audioSourceStart ?? 0; $0.audioSourceDuration = store.selectedClip?.duration } }
          .font(.caption)
      }
      if attachment.role == .keyframe {
        HStack {
          Text("Time (s)")
          TextField(
            "Time", value: Binding(get: { attachment.time }, set: { v in edit { $0.time = v } }),
            format: .number)
        }.font(.caption).textFieldStyle(.roundedBorder)
      }
      if attachment.role == .control {
        Picker(
          "Guide",
          selection: Binding(
            get: { attachment.controlType }, set: { v in edit { $0.controlType = v } })
        ) {
          if !guideTypes.contains(where: { $0.1 == attachment.controlType }) {
            Text("\(attachment.controlType) · incompatible input").tag(attachment.controlType).disabled(true)
          }
          ForEach(guideTypes, id: \.1) { type in Text(type.0).tag(type.1) }
        }.labelsHidden()
      }
      if let clip = store.selectedClip,
        let asset = store.allAssets.first(where: { $0.id == attachment.assetID }) {
        if let action = clip.referenceActions(for: asset).first(where: {
          $0.role == attachment.role && ($0.role != .control || $0.controlType == attachment.controlType)
        }) {
          Text(action.detail).font(.caption2).foregroundStyle(.secondary)
        } else if !clip.canAssignMedia(asset, role: attachment.role) {
          Text("This media role is unsupported by the selected model. Choose a supported purpose from the asset menu.")
            .font(.caption2).foregroundStyle(.orange)
        }
      }
      if isDrawThings && supportedDrawThingsInput {
        HStack {
          Text("Input strength · \(attachment.strength, specifier: "%.2f")")
          if attachment.strength != 1 {
            Button("Reset to 1") { edit { $0.strength = 1 } }
          }
        }.font(.caption2)
      } else if !isDrawThings && (store.selectedClip?.engine == .h3 && attachment.role != .control || attachment.role == .audioDriver) {
        HStack {
          Text("Strength · fixed at 1")
          if attachment.strength != 1 { Button("Reset to 1") { edit { $0.strength = 1 } } }
        }.font(.caption2)
      } else if !isDrawThings {
        HStack {
          Text("Strength")
          Slider(
            value: Binding(get: { attachment.strength }, set: { v in edit { $0.strength = v } }),
            in: 0...(attachment.role == .lora ? 2 : 1))
          Text("\(attachment.strength,specifier:"%.2f")").monospacedDigit()
        }.font(.caption2)
      }
      if attachment.role == .reference || attachment.controlType == "ingredients_reference_sheet" {
        TextField(
          "Describe this reference",
          text: Binding(get: { attachment.description }, set: { v in edit { $0.description = v } })
        ).font(.caption).textFieldStyle(.roundedBorder)
        if store.selectedClip?.engine == .ltx25 && attachment.role == .reference {
          msrControls
        }
      }
    }.padding(9).background(Theme.raised, in: RoundedRectangle(cornerRadius: 7))
  }
  private func referenceBinding(_ key: WritableKeyPath<Attachment, String?>) -> Binding<String> {
    Binding(
      get: { attachment[keyPath: key] ?? "" },
      set: { v in edit { $0[keyPath: key] = v.isEmpty ? nil : v } })
  }
  private var msrControls: some View {
    VStack(alignment: .leading, spacing: 6) {
      Picker("MSR role", selection: referenceBinding(\.referenceRole)) {
        Text("Recipe default · otherwise subject").tag("")
        Text("Subject").tag("subject")
        Text("Object").tag("object")
        Text("Clothing").tag("clothing")
        Text("Background").tag("background")
      }
      Picker("Priority", selection: referenceBinding(\.referencePriority)) {
        Text("Recipe default").tag("")
        Text("Automatic").tag("auto")
        Text("Primary").tag("primary")
        Text("Supporting").tag("supporting")
        Text("Background").tag("background")
      }
      Picker("Reference frames", selection: referenceBinding(\.referenceFrames)) {
        Text("Recipe default").tag("")
        Text("Automatic").tag("auto")
        Text("25").tag("25")
        Text("33").tag("33")
      }
      Picker("Reference sizing", selection: referenceBinding(\.referenceSizePolicy)) {
        Text("Recipe default").tag("")
        Text("Automatic").tag("sol_auto")
        Text("Quality").tag("quality")
        Text("Balanced").tag("balanced")
        Text("Speed").tag("speed")
      }
      Toggle("Override attention strength", isOn: Binding(
        get: { attachment.attentionStrength != nil },
        set: { value in edit { $0.attentionStrength = value ? 1 : nil } }))
      if attachment.attentionStrength != nil {
        HStack {
          Text("Attention")
          Slider(value: Binding(
            get: { attachment.attentionStrength ?? 1 },
            set: { v in edit { $0.attentionStrength = v } }), in: 0...1)
          Text("\(attachment.attentionStrength ?? 1, specifier: "%.2f")").monospacedDigit()
        }
      }
      Text("MSR uses 1–5 still images and at most one background. Model Setup must include its dedicated MSR adapter; Automatic selects the compatible components. Reference frames control encoding, not clip duration.")
        .foregroundStyle(.secondary)
    }.font(.caption2)
  }
  func edit(_ body: (inout Attachment) -> Void) {
    store.editClip { c in
      if let i = c.attachments.firstIndex(where: { $0.id == attachment.id }) {
        body(&c.attachments[i])
      }
    }
  }
}
struct MovieSettingsForm: View {
  @Binding var settings: MovieSettings
  var body: some View {
    VStack(alignment: .leading, spacing: 13) {
      HStack {
        SmallLabel(text: "Canvas")
        Spacer()
        Menu {
          Button("1920 × 1080") {
            settings.width = 1920
            settings.height = 1080
          }
          Button("1080 × 1920") {
            settings.width = 1080
            settings.height = 1920
          }
          Button("1280 × 720") {
            settings.width = 1280
            settings.height = 720
          }
          Button("1080 × 1080") {
            settings.width = 1080
            settings.height = 1080
          }
        } label: {
          Image(systemName: "rectangle.3.group")
        }
      }
      HStack {
        TextField("Width", value: $settings.width, format: .number)
        Text("×").foregroundStyle(.secondary)
        TextField("Height", value: $settings.height, format: .number)
        Text("px").foregroundStyle(.secondary)
      }
      HStack {
        Text("Frame rate")
        Spacer()
        Picker("FPS", selection: $settings.fps) {
          ForEach([24.0, 25, 30, 48, 50, 60], id: \.self) {
            Text("\($0,specifier:"%.0f") fps").tag($0)
          }
        }.labelsHidden()
      }
      HStack {
        Text("Fit")
        Spacer()
        Picker("Fit", selection: $settings.fit) {
          Text("Fit · preserve all").tag("fit")
          Text("Fill · crop edges").tag("fill")
        }.labelsHidden()
      }
      HStack {
        Text("File type")
        Spacer()
        Picker("Format", selection: $settings.format) {
          Text("MP4 · H.264").tag(MovieFormat.mp4)
          Text("MOV · H.264").tag(MovieFormat.mov)
          Text("MOV · ProRes 422 HQ").tag(MovieFormat.proRes)
          Text("PNG sequence + WAV").tag(MovieFormat.pngSequence)
        }.labelsHidden()
      }
      DisclosureGroup("Upscale & interpolate") {
        VStack(alignment: .leading, spacing: 10) {
          Picker("Upscaling", selection: $settings.upscaling) {
            Text("Off").tag(Upscaling.off)
            Text("Lanczos").tag(Upscaling.lanczos)
            Text("MetalFX spatial").tag(Upscaling.metalFX)
          }
          if settings.upscaling != .off {
            HStack {
              TextField("Upscale width", value: $settings.upscaleWidth, format: .number)
              Text("×")
              TextField("Upscale height", value: $settings.upscaleHeight, format: .number)
            }
          }
          Picker("Interpolation", selection: $settings.interpolation) {
            Text("Off").tag(Interpolation.off)
            Text("RIFE · MLX").tag(Interpolation.rife)
            Text("MetalFX · guide inputs").tag(Interpolation.metalFX)
          }
          if settings.interpolation != .off {
            HStack {
              Text("Output FPS")
              TextField("Interpolated FPS", value: $settings.interpolatedFPS, format: .number)
            }
          }
          if settings.interpolation == .rife {
            Picker("RIFE scale", selection: $settings.rifeScale) {
              Text("1.0 · full").tag(1.0)
              Text("0.5 · lower memory").tag(0.5)
              Text("0.25 · minimum memory").tag(0.25)
            }
          }
          Text(
            "Applied per clip: upscale first, then interpolate. Titles and transitions are assembled afterward."
          ).font(.caption2).foregroundStyle(.secondary)
        }.padding(.top, 8)
      }
    }.font(.system(size: 11)).textFieldStyle(.roundedBorder).pickerStyle(.menu)
  }
}
struct TitleInspector: View {
  @EnvironmentObject var store: StudioStore
  var body: some View {
    if let t = store.project.titles.first(where: { $0.id == store.selectedTitleID }) {
      VStack(alignment: .leading, spacing: 14) {
        Text("Title overlay").font(.headline)
        TextField("Title text", text: binding(t, \.text), axis: .vertical).lineLimit(2...5)
          .textFieldStyle(.roundedBorder)
        HStack {
          Text("Start")
          TextField("Start", value: binding(t, \.start), format: .number)
        }
        HStack {
          Text("Duration")
          TextField("Duration", value: binding(t, \.duration), format: .number)
        }
        HStack {
          Text("Size")
          TextField("Font size", value: binding(t, \.fontSize), format: .number)
        }
        Picker("Position", selection: binding(t, \.position)) {
          Text("Lower third").tag("lower")
          Text("Center").tag("center")
        }
        Button("Remove title", role: .destructive) {
          store.change { $0.titles.removeAll { $0.id == t.id } }
          store.selectedTitleID = nil
        }
      }.font(.system(size: 12)).textFieldStyle(.roundedBorder)
    }
  }
  func binding<T>(_ t: TitleOverlay, _ key: WritableKeyPath<TitleOverlay, T>) -> Binding<T> {
    Binding(
      get: { store.project.titles.first { $0.id == t.id }?[keyPath: key] ?? t[keyPath: key] },
      set: { v in
        store.change { p in
          if let i = p.titles.firstIndex(where: { $0.id == t.id }) { p.titles[i][keyPath: key] = v }
        }
      })
  }
}
struct AudioInspector: View {
  @EnvironmentObject var store: StudioStore
  var body: some View {
    if let a = store.project.audio.first(where: { $0.id == store.selectedAudioID }) {
      VStack(alignment: .leading, spacing: 14) {
        Text("Audio region").font(.headline)
        Picker("Track", selection: binding(a, \.trackID)) {
          ForEach(store.project.audioTracks) { t in Text(t.name).tag(Optional(t.id)) }
        }
        Text(URL(fileURLWithPath: a.path).lastPathComponent).lineLimit(2).foregroundStyle(
          .secondary)
        HStack {
          Text("Start")
          TextField("Start", value: binding(a, \.start), format: .number).disabled(a.anchor != nil)
        }
        HStack {
          Text("Source in")
          TextField("In point", value: binding(a, \.sourceIn), format: .number)
        }
        HStack {
          Text("Duration")
          TextField("Duration", value: binding(a, \.duration), format: .number)
        }
        HStack {
          Text("Volume")
          Slider(value: binding(a, \.volume), in: 0...2)
        }
        HStack {
          Text("Fade in")
          TextField("Seconds", value: Binding(get: { a.effectiveFadeIn }, set: { binding(a, \.fadeIn).wrappedValue = $0 }), format: .number)
          Text("Fade out")
          TextField("Seconds", value: Binding(get: { a.effectiveFadeOut }, set: { binding(a, \.fadeOut).wrappedValue = $0 }), format: .number)
        }
        if a.envelope != nil { Text("Retains the original fade through this split. Editing a fade starts a new region envelope.").font(.caption).foregroundStyle(.secondary) }
        Picker("Fade curve", selection: Binding(get: { a.fadeCurve ?? "linear" }, set: { binding(a, \.fadeCurve).wrappedValue = $0 })) {
          Text("Linear").tag("linear"); Text("Equal power").tag("equalPower")
        }
        if let anchor = a.anchor {
          HStack { Text("Clip offset"); TextField("Seconds", value: Binding(get: { anchor.offsetSeconds }, set: { value in
            binding(a, \.anchor).wrappedValue = ClipAudioAnchor(clipID: anchor.clipID, offsetSeconds: value)
          }), format: .number) }
          Button("Detach from clip") { store.change { p in if let i = p.audio.firstIndex(where: { $0.id == a.id }) {
            p.audio[i].start = (try? resolvedAudioStart(a, in: p)) ?? a.start; p.audio[i].anchor = nil
          } } }
        }
        Button("Crossfade with next region") { store.crossfadeAudio(a.id) }

        Button("Remove audio", role: .destructive) {
          store.change { $0.audio.removeAll { $0.id == a.id } }
          store.selectedAudioID = nil
        }
      }.font(.system(size: 12)).textFieldStyle(.roundedBorder)
    }
  }
  func binding<T>(_ a: AudioRegion, _ key: WritableKeyPath<AudioRegion, T>) -> Binding<T> {
    Binding(
      get: { store.project.audio.first { $0.id == a.id }?[keyPath: key] ?? a[keyPath: key] },
      set: { v in
        store.change { p in
          if let i = p.audio.firstIndex(where: { $0.id == a.id }) {
            p.audio[i][keyPath: key] = v
            if key == \AudioRegion.fadeIn || key == \AudioRegion.fadeOut || key == \AudioRegion.fadeCurve {
              p.audio[i].envelope = nil
            }
          }
        }
      })
  }
}

struct TrackInspector: View {
  @EnvironmentObject var store: StudioStore
  var body: some View {
    if let t = store.project.audioTracks.first(where: { $0.id == store.selectedTrackID }) {
      VStack(alignment: .leading, spacing: 14) {
        Text("Audio track").font(.headline)
        TextField("Track name", text: binding(t, \.name)).textFieldStyle(.roundedBorder)
        Picker("Role", selection: binding(t, \.role)) {
          ForEach(AudioTrackRole.allCases, id: \.self) { Text($0.rawValue.capitalized).tag($0) }
        }
        HStack { Text("Gain"); Slider(value: binding(t, \.gainDb), in: -60...12); Text("\(t.gainDb, specifier: "%.1f") dB").monospacedDigit() }
        HStack { Text("Pan / balance"); Slider(value: binding(t, \.pan), in: -1...1) }
        Button("Reset gain and pan") { store.change { p in
          if let i = p.audioTracks.firstIndex(where: { $0.id == t.id }) { p.audioTracks[i].gainDb = 0; p.audioTracks[i].pan = 0 }
        } }
        if t.role == .music {
          Toggle("Duck music under voice", isOn: Binding(get: { t.ducking != nil }, set: { enabled in
            store.change { p in if let i = p.audioTracks.firstIndex(where: { $0.id == t.id }) { p.audioTracks[i].ducking = enabled ? AudioDucking() : nil } }
          }))
          if t.ducking != nil { Text("Up to 12 dB reduction · 20 ms attack · 250 ms release").font(.caption).foregroundStyle(.secondary) }
        }
        AudioReverbControls(effect: binding(t, \.reverb))
        Toggle("Mute", isOn: binding(t, \.muted))
        Toggle("Solo", isOn: binding(t, \.solo))
        Toggle("Replace source audio", isOn: binding(t, \.replacesSource))
        Text(
          "Replacement mutes clip audio only where this track has an active region. Other music and effects tracks remain mixed."
        ).font(.caption).foregroundStyle(.secondary)
        Button("Import audio…") { store.chooseImports() }
        Button("Generate Music…") { store.openMusic() }
        Button("Generate Voice…") { store.openVoice() }
        Button("Add another track") { store.addAudioTrack() }
      }.font(.system(size: 12))
    }
  }
  func binding<T>(_ t: AudioTrack, _ key: WritableKeyPath<AudioTrack, T>) -> Binding<T> {
    Binding(
      get: { store.project.audioTracks.first { $0.id == t.id }?[keyPath: key] ?? t[keyPath: key] },
      set: { v in
        store.change { p in
          if let i = p.audioTracks.firstIndex(where: { $0.id == t.id }) {
            p.audioTracks[i][keyPath: key] = v
          }
        }
      })
  }
}
