import StudioCore
import SwiftUI

struct ContinuityInspector: View {
  @EnvironmentObject var store: StudioStore
  var clip: Clip
  @State private var chooseSceneAnchors = false
  @State private var sceneAnchorClipID: UUID?
  private func update(_ mutate: (inout ClipContinuity) -> Void) {
    store.editClip {
      var settings = $0.continuity ?? ClipContinuity()
      mutate(&settings)
      $0.continuity = settings
    }
  }
  var body: some View {
    VStack(alignment: .leading, spacing: 8) {
      Text("CLIP CONTINUITY").font(.caption).foregroundStyle(.secondary)
      Picker("Connection", selection: Binding(get: { clip.continuityMode }, set: { value in
        if value == "scene" {
          if clip.continuityMode == "frame" {
            sceneAnchorClipID = clip.id
            chooseSceneAnchors = true
          } else {
            Task { await store.connectContinuousScene(clipID: clip.id, preserveFrameMatch: false) }
          }
        } else {
          update { $0.mode = value }
        }
      })) {
        Text("Independent").tag("independent")
        Text("Match previous frame").tag("frame")
        if clip.engine == .ltx25 {
          Text("Continue scene").tag("scene")
          Text("Extend previous take").tag("motion")
        } else {
          Text("Continue scene").tag("motion")
          if clip.continuityMode == "scene" {
            Text("Continue scene · incompatible LTX 2.5 group").tag("scene").disabled(true)
          }
        }
      }.labelsHidden()
      if clip.engine != .ltx25 && store.project.isContinuousSceneMember(clip) {
        Text("This shot is still connected to an LTX 2.5 scene. Switch it to LTX 2.5, or separate it to use this model. Images, takes and edit points will be preserved.")
          .foregroundStyle(.orange)
        Button("Separate this shot") { store.separateContinuousScene(clipID: clip.id) }
          .disabled(store.operationBusy)
      } else if clip.continuityMode == "scene" {
        Text("Experimental: connects this shot to the previous LTX 2.5 shot. Generate any member to render the complete scene with shared motion and sound.")
          .foregroundStyle(.secondary)
        ForEach(store.project.continuityIssues(for: clip), id: \.self) { issue in
          Label(issue, systemImage: "exclamationmark.triangle").foregroundStyle(.orange)
        }
      } else if clip.continuityMode != "independent" {
        Picker("Source", selection: Binding<UUID?>(get: { clip.continuity?.sourceClipID }, set: { value in
          update { $0.sourceClipID = value }
        })) {
          Text("Immediately previous clip").tag(nil as UUID?)
          ForEach(store.project.earlierContinuitySources(for: clip)) { source in
            Text(source.name).tag(Optional(source.id))
          }
        }
        if clip.continuityMode == "frame" {
          Text("Uses the accepted source take’s visible ending as the effective first frame. Your stored first-frame attachment and other settings are preserved.")
        } else if clip.engine == .h3 {
          Text("Experimental: continues an accepted H3 take with its saved motion and audio. Render and accept the source with motion context first. Its visible ending must be untrimmed; this generates only the new shot.")
        } else {
          Text("Experimental: extends an accepted take using its visible ending and audio. This generates only the new shot. Endpoint, audio, and reference attachments cannot be combined. Model and sampler compatibility is checked before rendering; choose Match previous frame for other controls.")
        }
        ForEach(store.project.continuityIssues(for: clip), id: \.self) { issue in
          Label(issue, systemImage: "exclamationmark.triangle").foregroundStyle(.orange)
        }
      }
      if let members = try? store.project.continuousSceneMembers(for: clip), let leader = members.first {
        Divider()
        Text("Continuous scene · \(members.count) shots · \(members.reduce(0) { $0 + $1.duration }, specifier: "%.2f") s")
          .fontWeight(.semibold)
        Text(members.map(\.name).joined(separator: " → ")).foregroundStyle(.secondary)
        Picker("Boundary image guidance", selection: Binding(
          get: { leader.continuity?.boundaryImagePolicy ?? "balanced" },
          set: { value in
            store.change { project in
              if let index = project.clips.firstIndex(where: { $0.id == leader.id }) {
                var settings = project.clips[index].continuity ?? ClipContinuity()
                settings.boundaryImagePolicy = value
                project.clips[index].continuity = settings
              }
            }
          }
        )) {
          Text("Automatic · smoother joins").tag("balanced")
          Text("Strict · repeat overlap images").tag("strict")
        }
        Text((leader.continuity?.boundaryImagePolicy ?? "balanced") == "balanced"
          ? "Each image guides one window at its requested strength. Following windows inherit it through motion history, avoiding duplicate image guidance."
          : "Reapplies images in every overlapping window. Competing image and motion guidance can produce a flash at a join.")
          .foregroundStyle(.secondary)
        TextField("Shared scene sound", text: Binding(get: { leader.soundscape }, set: { value in
          store.change { project in
            if let index = project.clips.firstIndex(where: { $0.id == leader.id }) {
              project.clips[index].soundscape = value
            }
          }
        }), axis: .vertical).lineLimit(2...5)
        TextField("Scene music (N/A for none)", text: Binding(get: { leader.music }, set: { value in
          store.change { project in
            if let index = project.clips.firstIndex(where: { $0.id == leader.id }) {
              project.clips[index].music = value
            }
          }
        }))
        Text("Generated sound follows these instructions but may still introduce music. For exact sound, mute the generated sound and add a soundtrack in the timeline.")
          .foregroundStyle(.secondary)
        Toggle("Use generated scene sound in movie", isOn: Binding(get: { members.contains { $0.volume > 0 } }, set: { enabled in
          let ids = Set(members.map(\.id))
          store.change { project in
            for index in project.clips.indices where ids.contains(project.clips[index].id) {
              project.clips[index].volume = enabled ? 1 : 0
            }
          }
        }))
        if store.pendingContinuousScene != nil {
          Button("Review generated scene") { store.showContinuousSceneReview = true }
        }
      }
      if clip.engine == .h3 {
        Toggle("Save motion context", isOn: Binding(get: { clip.continuity?.saveContext ?? false }, set: { value in
          update { $0.saveContext = value }
        }))
        if store.project.shouldSaveContinuityContext(for: clip), clip.continuity?.saveContext != true {
          Text("Context will be saved automatically because a later H3 clip continues this clip.")
            .foregroundStyle(.secondary)
        }
      }
    }.font(.caption)
      .confirmationDialog("Choose the scene's first image", isPresented: $chooseSceneAnchors, titleVisibility: .visible) {
        Button("Keep current frame match") {
          if let id = sceneAnchorClipID { Task { await store.connectContinuousScene(clipID: id, preserveFrameMatch: true) } }
        }
        Button("Use attached images") {
          if let id = sceneAnchorClipID { Task { await store.connectContinuousScene(clipID: id, preserveFrameMatch: false) } }
        }
        Button("Cancel", role: .cancel) {}
      } message: {
        Text("Keep current frame match freezes the previous take's visible ending as an image. Use attached images restores the stored first-frame image. The original assets remain available.")
      }
  }
}
