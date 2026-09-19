import AppKit
import StudioCore
import SwiftUI
import UniformTypeIdentifiers

struct PromptEditor: View {
  @EnvironmentObject var store: StudioStore
  @State private var assistant: PromptAssistantContext?
  var body: some View {
    VStack(spacing: 0) {
      HStack(spacing: 14) {
        Button {
          store.showPrompt = false
        } label: {
          Label("Back to movie", systemImage: "arrow.left")
        }.keyboardShortcut(.escape, modifiers: [])
        Divider().frame(height: 20)
        Text(store.selectedClip?.name ?? "Prompt").font(.headline)
        Button("Prompt Assistant…") {
          if let clip = store.selectedClip { assistant = PromptAssistantContext(project: store.project, clip: clip, assets: store.allAssets, documentSessionID: store.documentSessionID) }
        }.disabled(store.bridge.busy || store.selectedClip == nil)
        Spacer()
        Text(store.selectedClip?.displayTask ?? "").font(.system(size: 11)).foregroundStyle(
          .secondary)
      }.padding(.horizontal, 24).frame(height: 64)
      Divider()
      if let clip = store.selectedClip {
        HSplitView {
          VStack(alignment: .leading, spacing: 16) {
            HStack {
              SmallLabel(text: "Shot direction")
              Spacer()
              Text("⌘ Return to open · Esc to close").font(.caption2).foregroundStyle(.tertiary)
            }
            Text("Describe what happens.").font(.system(size: 28, weight: .light))
            Text(
              "Action, camera, lighting, dialogue and sound. The complete prompt is shown before generation."
            ).font(.system(size: 12)).foregroundStyle(.secondary)
            TextEditor(
              text: Binding(
                get: { store.selectedClip?.prompt ?? "" },
                set: { v in store.editClip { $0.prompt = v } })
            ).font(.system(size: 15)).scrollContentBackground(.hidden).padding(12).background(
              Theme.raised, in: RoundedRectangle(cornerRadius: 8)
            ).overlay(RoundedRectangle(cornerRadius: 8).strokeBorder(Theme.line)).frame(
              minHeight: 250)
            if clip.engine == .h3 {
              HStack(alignment: .top, spacing: 16) {
                VStack(alignment: .leading, spacing: 7) {
                  SmallLabel(text: "Soundscape")
                  TextField(
                    "Ambience and physical sounds",
                    text: Binding(
                      get: { store.selectedClip?.soundscape ?? "" },
                      set: { v in store.editClip { $0.soundscape = v } }), axis: .vertical
                  ).lineLimit(2...4)
                }
                VStack(alignment: .leading, spacing: 7) {
                  SmallLabel(text: "Music")
                  TextField(
                    "N/A or music direction",
                    text: Binding(
                      get: { store.selectedClip?.music ?? "N/A" },
                      set: { v in store.editClip { $0.music = v } }), axis: .vertical
                  ).lineLimit(2...4)
                }
              }.textFieldStyle(.roundedBorder)
              if ["ref2va", "a2v", "extension"].contains(clip.inferredTask) {
                Label(
                  "Paste the complete native H3 reference prompt for this task. Reference descriptions are never invented.",
                  systemImage: "info.circle"
                ).font(.caption).foregroundStyle(.secondary)
              }
            } else {
              DisclosureGroup("Negative prompt") {
                TextEditor(
                  text: Binding(
                    get: { store.selectedClip?.negativePrompt ?? "" },
                    set: { v in store.editClip { $0.negativePrompt = v } })
                ).font(.system(size: 12)).frame(height: 65)
              }.font(.caption)
            }
            HStack {
              Button("Save as prompt asset") {
                var a = MediaAsset(
                  name: clip.name + " prompt", kind: .text, scope: .clip, owner: clip.id)
                a.text = clip.prompt
                store.change { $0.assets.append(a) }
                store.notice = "Prompt saved in Clip Assets."
              }
              Spacer()
              Text("\(clip.prompt.count) characters").font(.caption).foregroundStyle(.secondary)
            }
          }.padding(28).frame(minWidth: 520, maxWidth: .infinity)
          VStack(alignment: .leading, spacing: 16) {
            SmallLabel(text: "Shot context")
            if clip.engine == .drawThings { DrawThingsCUStatus(clip: clip) }
            HStack {
              Text(clip.engine.label).font(.headline)
              Spacer()
              Text("\(clip.duration,specifier:"%.1f") sec").font(
                .system(size: 12, design: .monospaced))
            }
            Text(
              "\(clip.generationWidth) × \(clip.generationHeight) generation · movie settings applied at finishing"
            ).font(.caption).foregroundStyle(.secondary)
            if !store.selectedContinuousScene.isEmpty {
              let members = store.selectedContinuousScene
              Label("Generation covers all \(members.count) shots · \(members.reduce(0) { $0 + $1.duration }, specifier: "%.2f") seconds", systemImage: "film.stack")
                .font(.caption).foregroundStyle(.secondary)
            }
            ScrollView {
              VStack(alignment: .leading, spacing: 10) {
                if clip.reviewUsesContinuityFrame {
                  Label("First frame: the accepted source clip’s visible ending. Your stored first frame is preserved in the movie inspector.", systemImage: "arrow.right.to.line")
                    .font(.caption).foregroundStyle(.secondary)
                } else if clip.engine != .drawThings && clip.continuityMode == "motion" {
                  Label("Motion and audio continue from the accepted source clip.", systemImage: "film.stack")
                    .font(.caption).foregroundStyle(.secondary)
                }
                ForEach(clip.reviewAttachments) { a in AttachmentRow(attachment: a) }
                if clip.reviewMediaCount == 0 && clip.continuityMode != "motion" {
                  Text(
                    "No media conditioning. Add assets in the main window, then choose their role."
                  ).font(.caption).foregroundStyle(.secondary)
                }
                Divider().padding(.vertical, 8)
                SmallLabel(text: "Exact render prompt")
                Text(
                  store.preparedPrompt.isEmpty
                    ? "Prepare this clip to validate its model, inputs and task. The resolved prompt appears here."
                    : store.preparedPrompt
                ).font(.system(size: 11, design: .monospaced)).textSelection(.enabled)
                  .foregroundStyle(store.preparedPrompt.isEmpty ? .secondary : .primary)
                if !store.preparedReport.isEmpty {
                  RenderSettingsSummary(clip: clip, report: store.preparedReport)
                  DisclosureGroup("Validation details (JSON)") {
                    Text(store.preparedReport).font(.system(size: 11, design: .monospaced))
                      .textSelection(.enabled)
                  }.font(.caption)
                }
              }
            }
          }.padding(24).frame(minWidth: 310, idealWidth: 390, maxWidth: 490).background(Theme.panel)
        }
        Divider()
        PromptActions(bridge: store.bridge).environmentObject(store).padding(.horizontal, 28).frame(
          height: 68)
      } else {
        Spacer()
        Text("Select a generated clip first.")
        Spacer()
      }
    }.background(Theme.background).frame(maxWidth: .infinity, maxHeight: .infinity)
      .sheet(item: $assistant) { PromptAssistantView(context: $0).environmentObject(store) }
  }
}
struct PromptActions: View {
  @EnvironmentObject var store: StudioStore
  @ObservedObject var bridge: Bridge
  var body: some View {
    HStack {
      if store.operationBusy && !bridge.busy {
        ProgressView().controlSize(.small)
        Text("Checking render settings…").font(.caption)
      } else if bridge.busy {
        ProgressView().controlSize(.small)
        Text(bridge.message).font(.caption)
        Button("Cancel") { bridge.cancel() }
      } else {
        Image(systemName: !store.canGenerateSelected ? "checklist" : "checkmark.circle.fill")
          .foregroundStyle(!store.canGenerateSelected ? Color.secondary : Color.green)
        Text(
          !store.canGenerateSelected
            ? "Generate validates settings automatically" : "Preflight passed. Generate uses these settings."
        ).font(.caption).foregroundStyle(.secondary)
      }
      Spacer()
      Button("View log") { store.showLog = true }
      if store.pendingContinuousScene != nil {
        Button("Review scene") { store.showContinuousSceneReview = true }
      }
      Button(store.selectedContinuousScene.isEmpty ? "Prepare clip" : "Prepare scene") {
        Task { await store.prepareSelected() }
      }.disabled(store.operationBusy)
      Button {
        Task { await store.generateSelected() }
      } label: {
        Label(store.selectedContinuousScene.isEmpty ? "Generate clip" : "Generate scene", systemImage: "play.fill")
      }.buttonStyle(.borderedProminent).disabled(store.operationBusy || store.selectedClip?.prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty != false)
    }
  }
}
struct PathPicker: View {
  var label: String
  @Binding var value: String
  var directory = false
  var body: some View {
    VStack(alignment: .leading, spacing: 5) {
      Text(label).font(.caption).foregroundStyle(.secondary)
      HStack {
        TextField(label, text: $value).textFieldStyle(.roundedBorder)
        Button("Choose…") {
          let panel = NSOpenPanel()
          panel.canChooseDirectories = directory
          panel.canChooseFiles = !directory
          guard panel.runModal() == .OK, let url = panel.url else { return }
          value = url.path
        }
      }
    }
  }
}
struct RuntimeView: View {
  @EnvironmentObject var store: StudioStore
  @AppStorage("appearance") var appearance = "system"
  @StateObject private var installer = RuntimeInstaller()
  @StateObject private var modelSetup = ModelSetupState()
  var body: some View {
    VStack(alignment: .leading, spacing: 20) {
      HStack {
        Text("Studio Settings").font(.title2)
        Spacer()
        Button("Done") {
          store.saveRuntime()
          store.showRuntime = false
        }.keyboardShortcut(.defaultAction)
      }
      ScrollView {
        VStack(alignment: .leading, spacing: 16) {
          Picker("Appearance", selection: $appearance) {
            Text("System").tag("system")
            Text("Light").tag("light")
            Text("Dark").tag("dark")
          }.pickerStyle(.segmented)
          LoRAFolderSettings()
          Divider()
          VoiceModelSettingsView()
          Divider()
          Text("Performance and acceleration").font(.headline)
          Text("This Mac · \(Int(ProcessInfo.processInfo.physicalMemory / 1_073_741_824)) GiB unified memory")
            .font(.caption).foregroundStyle(.secondary)
          Picker("H3 sampling memory", selection: Binding(get: {
            store.runtime.acceleration?.h3MemoryPolicy ?? "automatic"
          }, set: { value in
            if store.runtime.acceleration == nil { store.runtime.acceleration = AccelerationSettings() }
            store.runtime.acceleration?.h3MemoryPolicy = value
          })) {
            Text("Automatic · recommended").tag("automatic")
            Text("Paged · lower memory").tag("paged")
            Text("Paged · larger workspace").tag("pagedNormal")
            Text("Resident · experimental high RAM").tag("resident")
          }
          Picker("H3 projection backend", selection: Binding(get: {
            store.runtime.acceleration?.h3ProjectionBackend ?? "auto"
          }, set: { value in
            if store.runtime.acceleration == nil { store.runtime.acceleration = AccelerationSettings() }
            store.runtime.acceleration?.h3ProjectionBackend = value
          })) {
            Text("Automatic · supported acceleration").tag("auto")
            Text("MLX").tag("mlx")
          }
          Text("Automatic uses hardware-aware memory policy and the verified projection backend. The resolved settings explain the actual choice. Paged · larger workspace retains checkpoint pagination with larger working buffers; fit on 36 GB Macs has not been qualified. Resident sampling is experimental and requires high RAM; it is not qualified on 36 GB Macs. Clips can override these defaults. Legacy custom recipes keep their saved behavior.")
            .font(.caption).foregroundStyle(.secondary)
          Button("Restore automatic acceleration") { store.runtime.acceleration = AccelerationSettings() }
          Divider()
          Text("Renderer").font(.headline)
          Text(
            "Studio can install a private native renderer with its own Python and verified dependencies. Existing model files stay shared. Advanced users can connect an existing environment."
          ).font(.caption).foregroundStyle(.secondary)
          HStack {
            Button(installer.busy ? "Setting up…" : "Set Up Managed Renderer") {
              Task {
                do {
                  let source = Bundle.main.resourceURL?.appendingPathComponent("RendererSource")
                  let root =
                    source.flatMap { FileManager.default.fileExists(atPath: $0.path) ? $0 : nil }
                    ?? URL(fileURLWithPath: store.runtime.root)
                  let result = try await installer.install(source: root)
                  if let root = result["root"], let python = result["pythonPath"] {
                    store.runtime.root = root
                    store.runtime.pythonPath = python
                    store.saveRuntime()
                  }
                } catch { store.error = error.localizedDescription }
              }
            }.disabled(installer.busy || store.bridge.busy)
            if installer.busy {
              ProgressView().controlSize(.small)
              Button("Cancel") { installer.cancel() }
            }
          }
          if !installer.message.isEmpty { Text(installer.message).font(.caption) }
          if !installer.log.isEmpty {
            DisclosureGroup("Setup log") {
              ScrollView {
                Text(installer.log).font(.system(size: 10, design: .monospaced)).textSelection(
                  .enabled)
              }.frame(height: 120)
            }
          }
          PathPicker(label: "WeeTodd Studio repository", value: $store.runtime.root, directory: true)
          PathPicker(label: "Python executable", value: $store.runtime.pythonPath)
          PathPicker(
            label: "Model recipes folder", value: $store.runtime.profilesDirectory, directory: true)
          HStack {
            Button("Import model recipes…") { store.importRecipes() }
            Button("Refresh") { store.saveRuntime() }
            Spacer()
            Text("\(store.profiles.count) recipes").foregroundStyle(.secondary)
          }
          Text(
            "A recipe identifies a compatible component set and sampling policy. Automatic selection matches the clip’s engine and selected task with compatible recipe contents."
          ).font(.caption).foregroundStyle(.secondary)
          Divider()
          ModelSetupCatalogView(
            state: modelSetup, bridge: store.bridge, rendererInstalling: installer.busy)
          Divider()
          Text("Finishing tools").font(.headline)
          PathPicker(
            label: "FFmpeg (blank uses installed executable)", value: $store.runtime.ffmpegPath)
          PathPicker(
            label: "FFprobe (blank uses installed executable)", value: $store.runtime.ffprobePath)
          PathPicker(label: "RIFE MLX executable", value: $store.runtime.rifePath)
          PathPicker(
            label: "RIFE weights folder", value: $store.runtime.rifeWeights, directory: true)
          PathPicker(label: "StudioMetal helper", value: $store.runtime.metalPath)
          Text(
            "MetalFX spatial upscaling uses the local GPU. Experimental MetalFX frame interpolation additionally requires matched depth and motion guides in the clip’s advanced settings."
          ).font(.caption).foregroundStyle(.secondary)
        }
      }
    }.padding(26).frame(width: 690, height: 730)
  }
}

struct RenderSettingsSummary: View {
  let clip: Clip
  let report: String
  private var details: [String: Any] {
    (try? JSONSerialization.jsonObject(with: Data(report.utf8))) as? [String: Any] ?? [:]
  }
  var sceneReport: ContinuousSceneRenderReport? {
    details["scene"].flatMap { try? ContinuousSceneRenderReport.decode($0) }
  }
  var requestedDuration: Double {
    guard sceneReport != nil,
      let plan = details["scenePlan"] as? [String: Any],
      let durations = plan["requested_durations"] as? [Double] else {
      return sceneReport?.duration ?? clip.duration
    }
    return durations.reduce(0, +)
  }
  var body: some View {
    let resolved = details
    let generation = resolved["generation"] as? [String: Any] ?? [:]
    let acceleration = generation["acceleration"] as? [String: Any] ?? [:]
    VStack(alignment: .leading, spacing: 7) {
      Text("Render settings").font(.headline)
      LabeledContent("Generation", value: clip.generationProvider.label)
      LabeledContent("Model", value: clip.engine.label)
      LabeledContent("Task", value: (resolved["task"] as? String)
        .map(GenerationSelection.taskLabel) ?? clip.displayTask)
      LabeledContent("Components", value: resolved["profile"] as? String ?? "See validation details")
      LabeledContent("Requested size", value: "\(clip.generationWidth) × \(clip.generationHeight)")
      LabeledContent("Requested duration", value: String(format: "%.2f s", requestedDuration))
      if let scene = sceneReport {
        LabeledContent("Resolved scene", value: String(format: "%d shots · %.2f s", scene.members.count, scene.duration))
      }
      if let guidance = resolved["sceneImageGuidance"] as? [String: Any] {
        let balanced = guidance["policy"] as? String == "balanced"
        let inherited = guidance["inherited_anchors"] as? [[String: Any]] ?? []
        LabeledContent("Boundary image guidance", value: balanced ? "Automatic" : "Strict")
        if !inherited.isEmpty {
          Text("Boundary images guide one window at their requested strength. Later windows inherit them through motion history.")
            .font(.caption).foregroundStyle(.secondary)
          DisclosureGroup("Boundary image routing") {
            ForEach(Array(inherited.enumerated()), id: \.offset) { _, item in
              if let frame = item["frame_index"] as? Int,
                let source = item["source_window"] as? Int,
                let window = item["window"] as? Int,
                let strength = item["strength"] as? Double {
                Text(String(format: "Frame %d · strength %.2f in window %d → inherited by window %d", frame, strength, source, window)).font(.caption)
              }
            }
          }
        }
      }
      ForEach(resolved["sceneAnchorWarnings"] as? [String] ?? [], id: \.self) { warning in
        Label(warning, systemImage: "exclamationmark.triangle").foregroundStyle(.orange)
      }
      if let fps = resolved["nativeFPS"] as? Double {
        LabeledContent("Generation frame rate", value: String(format: "%g fps", fps))
      }
      let conditioning = resolved["conditioning"] as? [String: Any]
      let contract = conditioning?["contract"] as? [String: Any]
      let inputs = contract?["inputs"] as? [[String: Any]]
      LabeledContent("Media inputs", value: String(inputs?.count ?? clip.reviewMediaCount))
      if clip.reviewLoRACount > 0 {
        LabeledContent("Active LoRAs", value: String(clip.reviewLoRACount))
      }
      LabeledContent(sceneReport == nil ? "Seed" : "Selected shot seed", value: String(clip.seed))
      if let policy = acceleration["memoryPolicy"] as? String {
        LabeledContent("Memory policy", value: AccelerationSettings.memoryPolicyLabel(policy))
      }
    }.font(.caption).textSelection(.enabled)
  }
}
