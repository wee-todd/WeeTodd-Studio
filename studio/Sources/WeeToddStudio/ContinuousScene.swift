import AppKit
import AVKit
import Foundation
import StudioCore
import SwiftUI

struct ContinuousSceneRenderReport: Codable, Equatable {
  var version: Int
  var members: [ContinuousSceneMember]
  var frameRate: Double
  var publicationMode: String
  enum CodingKeys: String, CodingKey {
    case version, members
    case frameRate = "frame_rate"
    case publicationMode = "publication_mode"
  }
  static func decode(_ value: Any) throws -> Self {
    let report = try JSONDecoder().decode(Self.self, from: JSONSerialization.data(withJSONObject: value))
    guard report.version == 1, report.frameRate.isFinite, report.frameRate > 0,
      report.publicationMode == "single_decode_native_latent_chain" else {
      throw StudioError.invalid("The result is not a supported native continuous scene.")
    }
    return report
  }
  var duration: Double { members.reduce(0) { $0 + $1.duration } }
}

struct PendingContinuousSceneTake: Identifiable {
  let id = UUID()
  var documentSessionID: UUID
  var projectID: UUID
  var requestKey: String
  var clips: [Clip]
  var report: ContinuousSceneRenderReport
  var video: String
  var recipePath: String
  var stats: RenderStats?
  var generation: GenerationDescriptor?
  var resolvedFingerprint: String?
}

@MainActor extension StudioStore {
  func connectContinuousScene(clipID: UUID, preserveFrameMatch: Bool) async {
    guard !operationBusy else { return }
    connectingContinuousScene = true
    defer { connectingContinuousScene = false }
    do {
      guard let index = project.clips.firstIndex(where: { $0.id == clipID }), index > 0,
        project.clips[index].engine == .ltx25, project.clips[index - 1].engine == .ltx25 else {
        throw StudioError.invalid("A continuous scene needs a preceding local LTX 2.5 shot.")
      }
      let snapshot = project
      let clip = snapshot.clips[index]
      let predecessor = snapshot.clips[index - 1].id
      let session = documentSessionID
      let settings = runtime
      let dependency = project.continuityDependencyFingerprint(for: clip)
      var anchor: MediaAsset?
      if preserveFrameMatch {
        guard clip.continuityMode == "frame" else { throw StudioError.invalid("Select Match previous frame before freezing its effective image.") }
        var body = try payload(); body["clipID"] = clipID.uuidString
        let destination = dataDirectory.appendingPathComponent("Anchors/\(UUID().uuidString)")
        let result = try await bridge.invoke("freeze-continuity-frame", runtime: settings, payload: body, output: destination)
        guard let filename = result["path"] as? String, FileManager.default.fileExists(atPath: filename) else {
          throw StudioError.invalid("The previous take did not produce a frozen first image.")
        }
        anchor = MediaAsset(name: clip.name + " · previous take ending", kind: .image,
          path: filename, scope: .clip, owner: clipID)
      }
      guard documentSessionID == session, project == snapshot,
        project.continuityDependencyFingerprint(for: clip) == dependency else {
        throw StudioError.invalid("The shot or its source changed while preparing the scene connection. Try again.")
      }
      change { project in
        if let anchor {
          project.assets.append(anchor)
          project.clips[index].attachments.removeAll { $0.role == .first }
          project.clips[index].attachments.append(Attachment(assetID: anchor.id, role: .first))
          let endpointTask = project.clips[index].attachments.contains {
            $0.role == .last || $0.role == .keyframe
          } ? "fflf" : "i2v"
          project.clips[index].selectGenerationTask(project.clips[index].attachments.contains { $0.role == .audioDriver } ? "a2v" : endpointTask)
        }
        project.clips[index].continuity = ClipContinuity(mode: "scene", sourceClipID: predecessor)
      }
      notice = preserveFrameMatch
        ? "Connected the shot and froze its previous frame match. The original image remains in Assets."
        : "Connected the shot using its attached images. Generate any member to render the complete scene."
    } catch { self.error = error.localizedDescription }
  }

  var selectedContinuousScene: [Clip] {
    guard let clip = selectedClip else { return [] }
    return (try? project.continuousSceneMembers(for: clip)) ?? []
  }

  /// Includes all scene inputs but never its accepted media or generation descriptions.
  /// Those change as a result is reviewed, without changing the requested generation.
  func continuousSceneDependencyKey(for clip: Clip) -> String {
    // Ordinary timeline shots have no scene inputs. Avoid serializing the model
    // library and probing every profile file on each redraw of those shots.
    do {
      guard !(try project.continuousSceneMembers(for: clip)).isEmpty else { return "" }
    } catch { return "invalid-scene:" + error.localizedDescription }
    let encoder = JSONEncoder(); encoder.outputFormatting = .sortedKeys
    let relevantProfiles = profiles.filter { $0.engine == "ltx25" }.sorted { $0.id < $1.id }
    let profileFiles = relevantProfiles.map { profile in
      let attributes = try? FileManager.default.attributesOfItem(atPath: profile.id)
      return profile.id + "|" + String(describing: (attributes?[.modificationDate] as? Date)?.timeIntervalSince1970)
        + "|" + String(describing: attributes?[.size])
    }.joined(separator: "\n")
    let runtimeKey = ((try? encoder.encode(runtime.generationSettings).base64EncodedString()) ?? "")
      + ((try? encoder.encode(relevantProfiles).base64EncodedString()) ?? "")
      + ((try? encoder.encode(loraGroups).base64EncodedString()) ?? "") + profileFiles
    do {
      return try project.continuousSceneInputFingerprint(for: clip, assets: allAssets, runtimeIdentity: runtimeKey)
    } catch { return "invalid-scene:" + error.localizedDescription }
  }

  func receiveContinuousScene(result: [String: Any], media: [String: Any],
    prepared: ContinuousSceneRenderReport, clips: [Clip], requestKey: String,
    projectID: UUID, session: UUID, recipePath: String, generation: GenerationDescriptor?,
    resolvedFingerprint: String?) throws {
    guard let video = result["video"] as? String else { throw StudioError.invalid("The scene returned no movie.") }
    guard documentSessionID == session, project.id == projectID else {
      throw StudioError.invalid("The destination project changed. The completed scene is saved at \(video)")
    }
    guard let object = result["scene"] else { throw StudioError.invalid("The result has no scene ranges. Movie saved at \(video)") }
    let report = try ContinuousSceneRenderReport.decode(object)
    guard report == prepared, report.members.map(\.clipID) == clips.map(\.id),
      let duration = media["duration"] as? Double, duration.isFinite,
      abs(duration - report.duration) <= 1 / report.frameRate,
      FileManager.default.fileExists(atPath: video) else {
      throw StudioError.invalid("The completed scene does not match the prepared members or duration. Movie saved at \(video)")
    }
    if let fps = media["fps"] as? Double, !fps.isFinite || abs(fps - report.frameRate) > 0.000001 {
      throw StudioError.invalid("The completed scene has a different frame rate. Movie saved at \(video)")
    }
    if let frames = media["num_frames"] as? Int, frames != Int((report.duration * report.frameRate).rounded()) {
      throw StudioError.invalid("The completed scene has a different frame count. Movie saved at \(video)")
    }
    pendingContinuousScene = PendingContinuousSceneTake(documentSessionID: session, projectID: projectID,
      requestKey: requestKey, clips: clips, report: report, video: video, recipePath: recipePath,
      stats: RenderStats(result: result), generation: generation, resolvedFingerprint: resolvedFingerprint)
    showContinuousSceneReview = true
    notice = continuousSceneMatchesCurrentInputs
      ? "Scene ready. Review the complete movie and accept all shots together."
      : "Scene saved for review. Its inputs changed during generation; it cannot replace the current shots."
  }

  var canAcceptContinuousScene: Bool {
    !operationBusy && continuousSceneMatchesCurrentInputs
  }

  private var continuousSceneMatchesCurrentInputs: Bool {
    guard let take = pendingContinuousScene, take.documentSessionID == documentSessionID,
      take.projectID == project.id, let firstID = take.clips.first?.id,
      let clip = project.clips.first(where: { $0.id == firstID }),
      FileManager.default.fileExists(atPath: take.video) else { return false }
    return continuousSceneDependencyKey(for: clip) == take.requestKey
      && ((try? project.continuousSceneMembers(for: clip).map(\.id)) == take.clips.map(\.id))
  }

  func acceptContinuousScene() async {
    do {
      guard let take = pendingContinuousScene, canAcceptContinuousScene else {
        throw StudioError.invalid("The scene inputs changed. Keep this movie for review and prepare the updated scene before replacing its shots.")
      }
      acceptingContinuousScene = true
      defer { acceptingContinuousScene = false }
      let before = project
      let settings = runtime
      var versions: [UUID: RenderVersion] = [:]
      for (clip, range) in zip(take.clips, take.report.members) {
        versions[clip.id] = RenderVersion(path: take.video, seed: clip.seed, prompt: clip.prompt,
          recipePath: take.recipePath, stats: take.stats, generationSettings: take.generation,
          resolvedFingerprint: take.resolvedFingerprint, usableSourceIn: range.sourceIn,
          usableDuration: range.duration, sceneMembers: take.report.members, sceneTakeID: take.id,
          sceneInputFingerprint: take.requestKey, sceneFrameRate: take.report.frameRate)
      }
      var accepted = project
      try accepted.acceptContinuousScene(versions: versions, members: take.report.members)
      var asset = MediaAsset(name: "Continuous scene · " + (take.clips.first?.name ?? "Movie"),
        kind: .video, path: take.video, scope: .project)
      asset.duration = take.report.duration
      asset.fps = take.report.frameRate
      accepted.assets.append(asset)
      // Resolve the accepted (possibly quantized) durations and every member's
      // actual dependency paths before stamping signatures. Reusing the pre-render
      // fingerprint here made an accepted scene immediately appear out of date.
      var descriptions: [UUID: [String: Any]] = [:]
      for clip in accepted.clips where versions[clip.id] != nil {
        let body: [String: Any] = ["project": try accepted.object(), "clipID": clip.id.uuidString,
          "globalAssets": try JSONSerialization.jsonObject(with: JSONEncoder().encode(globalAssets))]
        let description = try await descriptionBridge.invoke("describe-generation", runtime: settings, payload: body)
        guard documentSessionID == take.documentSessionID, project == before,
          pendingContinuousScene?.id == take.id,
          continuousSceneDependencyKey(for: before.clips.first(where: { $0.id == clip.id })!) == take.requestKey else {
          throw StudioError.invalid("The project changed during scene acceptance. The movie remains saved at \(take.video)")
        }
        if let issues = description["readinessErrors"] as? [String], !issues.isEmpty {
          throw StudioError.invalid("The accepted scene settings need review: " + issues.joined(separator: "\n"))
        }
        descriptions[clip.id] = description
      }
      change { $0 = accepted }
      for clip in project.clips where versions[clip.id] != nil {
        var description = descriptions[clip.id] ?? [:]
        description["studioInput"] = generationRequestKey(for: clip)
        description["studioEngine"] = clip.engine.rawValue
        description["studioTask"] = clip.inferredTask
        description["studioProfile"] = clip.profileID
        generationDescriptions[clip.id] = description
      }
      for index in project.clips.indices where versions[project.clips[index].id] != nil {
        let fingerprint = continuousSceneDependencyKey(for: project.clips[index])
        if let versionIndex = project.clips[index].versions.lastIndex(where: { $0.sceneTakeID == take.id }) {
          project.clips[index].versions[versionIndex].sceneInputFingerprint = fingerprint
        }
        project.clips[index].renderedSignature = signature(for: project.clips[index])
      }
      changed()
      pendingContinuousScene = nil; showContinuousSceneReview = false; showPrompt = false
      refreshPreview()
      notice = "Accepted the complete scene. All \(take.clips.count) shots use the same movie; prior takes remain in Versions."
    } catch { self.error = error.localizedDescription }
  }

  func activateRenderVersion(_ version: RenderVersion, for clip: Clip) {
    do {
      var updated = project
      if version.sceneMembers != nil {
        try updated.activateContinuousSceneVersion(selectedClipID: clip.id, version: version)
      } else {
        guard try project.continuousSceneMembers(for: clip).isEmpty else {
          throw StudioError.invalid("This is an individual shot take. Disconnect the shot from its continuous scene before using it alone.")
        }
        guard let index = updated.clips.firstIndex(where: { $0.id == clip.id }) else { return }
        try updated.clips[index].activateVersion(version)
      }
      change { $0 = updated }
      refreshPreview()
    } catch { self.error = error.localizedDescription }
  }
}

struct ContinuousSceneReviewView: View {
  @EnvironmentObject var store: StudioStore
  @State private var player: AVPlayer?
  var body: some View {
    VStack(alignment: .leading, spacing: 14) {
      if let take = store.pendingContinuousScene {
        HStack {
          Text("Review continuous scene").font(.title2)
          Spacer()
          Text("\(take.clips.count) shots · \(take.report.duration, specifier: "%.2f") seconds").foregroundStyle(.secondary)
        }
        VideoPlayer(player: player).frame(minWidth: 700, minHeight: 390)
        Text("Check motion and sound across every join. Accepting replaces all member shots together and preserves their previous takes.")
          .font(.callout).foregroundStyle(.secondary)
        HStack {
          ForEach(Array(zip(take.clips, take.report.members)), id: \.0.id) { clip, range in
            Button("\(clip.name) · \(range.sourceIn, specifier: "%.1f")s") {
              player?.seek(to: CMTime(seconds: max(0, range.sourceIn - 0.75), preferredTimescale: 600))
              player?.play()
            }.font(.caption)
          }
        }
        if !store.canAcceptContinuousScene && !store.operationBusy {
          Label("Scene inputs changed. The saved movie remains available; prepare the updated scene before accepting a replacement.", systemImage: "exclamationmark.triangle")
            .foregroundStyle(.orange).font(.caption)
        }
        HStack {
          Button("Show movie in Finder") { NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: take.video)]) }
          Spacer()
          Button("Review later") { store.showContinuousSceneReview = false }
          if store.acceptingContinuousScene { ProgressView().controlSize(.small) }
          Button("Accept entire scene") { Task { await store.acceptContinuousScene() } }
            .buttonStyle(.borderedProminent).disabled(!store.canAcceptContinuousScene)
        }
      } else {
        Text("No scene is waiting for review.")
        Button("Done") { store.showContinuousSceneReview = false }
      }
    }.padding(22).frame(minWidth: 820)
      .onChange(of: store.pendingContinuousScene?.id, initial: true) { _, _ in
        player?.pause()
        player = store.pendingContinuousScene.map { AVPlayer(url: URL(fileURLWithPath: $0.video)) }
      }.onDisappear { player?.pause(); player = nil }
  }
}
