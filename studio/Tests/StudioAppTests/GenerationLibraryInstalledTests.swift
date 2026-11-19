import AppKit
import Foundation
import ImageLayout
import StudioCore
import SwiftUI
import XCTest
@testable import WeeToddStudio

final class GenerationLibraryInstalledTests: XCTestCase {
  @MainActor func testInstalledLibraryFastPresetsReviewedOverrideAndRestart() async throws {
    guard let rootPath = ProcessInfo.processInfo.environment["WEETODD_PRESET_LIBRARY_CHECK_ROOT"],
      let profilesPath = ProcessInfo.processInfo.environment["WEETODD_PRESET_LIBRARY_CHECK_PROFILES"] else {
      throw XCTSkip("Opt-in read-only installed library and temporary-data preset qualification")
    }
    let temporary = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: temporary) }
    let store = StudioStore(dataDirectory: temporary, restoreSession: false)
    store.runtime.profilesDirectory = profilesPath
    let rows = try NativeLTXPreparation.catalog(directory: profilesPath) + NativeH3Preparation.catalog(directory: profilesPath)
    store.profiles = rows.compactMap { row in
      guard let id = row["id"] as? String, let name = row["name"] as? String, let engine = row["engine"] as? String, let task = row["task"] as? String else { return nil }
      let generation = (row["generation"] as? [String: Any]).flatMap { try? JSONDecoder().decode(GenerationDescriptor.self, from: JSONSerialization.data(withJSONObject: $0)) }
      return .init(id: id, name: name, engine: engine, task: task, generation: generation)
    }
    try store.setGenerationLibraryRoot(rootPath); await store.refreshGenerationLibrary(force: true)
    XCTAssertNil(store.generationLibraryError)
    XCTAssertTrue(store.generationLibrary.modelEntries.contains { $0.reference.backend == "ltx25" && $0.tasks.contains("video_upscale") && !($0.movieModes ?? []).isEmpty })
    if store.generationLibrary.modelEntries.contains(where: { $0.reference.backend == "ripple" && $0.availability == .installed }) {
      XCTAssertTrue(store.generationPresets.presets.contains { $0.snapshot.context.workspace == .ripple })
    }
    XCTAssertTrue(store.generationPresets.presets.contains { $0.snapshot.context.task == "video_upscale" && $0.snapshot.model?.backend == "ltx25" })
    if store.generationLibrary.modelEntries.contains(where: { $0.reference.family == "h3Regenerate" && $0.availability == .installed }) {
      XCTAssertTrue(store.generationPresets.presets.contains { $0.name.contains("H3 → Flash") })
    }
    XCTAssertTrue(store.generationPresets.presets.contains { $0.name.contains("Pixel + CQ → Flash") })
    XCTAssertTrue(store.generationLibrary.modelEntries.contains { $0.reference.precision == "q4" && $0.reference.backend == "nativeMOSS" })
    XCTAssertTrue(store.generationLibrary.modelEntries.contains { $0.reference.family == "relayVSR" && $0.reference.variant == "x2" })
    let guided = store.generationPresets.presets.filter { preset in
      preset.snapshot.context.workspace == .clip && preset.snapshot.context.task == "t2v" && store.generationLibrary.modelEntries.contains { $0.reference == preset.snapshot.model && $0.generation?.pipelineMode == "guided" }
    }
    for recipe in guided {
      var clip = Clip(engine: .ltx25); clip.prompt = ""; clip.generationSelection = .init(task: "t2v")
      store.project.clips = [clip]; store.selectedClipID = clip.id
      try await store.applyClipPreset(recipe, expectedTarget: store.clipPresetTargetToken())
      XCTAssertEqual(store.selectedClip?.generationSelection?.ltx25Guidance?.mode, .guided)
      XCTAssertEqual(store.selectedClip?.prompt, "")
    }
    XCTAssertFalse(guided.isEmpty)
    let preset = try XCTUnwrap(store.generationPresets.presets.first { $0.origin == .supplied && $0.name.contains("Fast 6 steps") && $0.snapshot.model?.precision == "bf16" })
    let model = try XCTUnwrap(preset.snapshot.model)
    var draft = DrawThingsImageDraft(destination: .init(scope: .global, projectID: store.project.id)); draft.selectProvider(.nativeMOSS)
    draft.prompt = "Preserve this creative prompt"; draft.width = 1200; draft.height = 1696
    store.restoringImageWorkspace = false; store.imageDraft = draft
    try await store.applyImagePreset(preset, expectedTarget: store.imagePresetTargetToken())
    XCTAssertEqual(store.imageDraft?.steps, 6); XCTAssertEqual(store.imageDraft?.width, 1200); XCTAssertEqual(store.imageDraft?.height, 1696)
    XCTAssertEqual(store.imageDraft?.prompt, draft.prompt); XCTAssertEqual(try store.captureImagePreset().model, model)
    store.imageDraft?.steps = 8
    let review = try store.beginPresetUpdate(for: .image)
    XCTAssertTrue(review.changes.contains { $0.path == "fields.steps" })
    XCTAssertEqual(store.generationPresets.preset(id: preset.id)?.snapshot.fields["steps"], .integer(6))
    try await store.confirmPresetUpdate(review)
    await store.refreshGenerationLibrary(force: true)
    XCTAssertEqual(store.generationPresets.preset(id: preset.id)?.snapshot.fields["steps"], .integer(8))
    let restored = StudioStore(dataDirectory: temporary, restoreSession: false); restored.restoreImageWorkspaces()
    XCTAssertEqual(restored.imageDraft?.steps, 8); XCTAssertEqual(restored.imagePresetSelection?.presetID, preset.id)
    XCTAssertEqual(restored.generationPresets.preset(id: preset.id)?.revision, 2)
    XCTAssertFalse(store.bridge.busy)
    if let output = ProcessInfo.processInfo.environment["WEETODD_PRESET_LIBRARY_CHECK_OUTPUT"] {
      let folder = URL(fileURLWithPath: output); try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
      store.runtime.pythonPath = "/unavailable/no-python"; store.saveRuntime(reloadProfiles: false)
      let qa = folder.appendingPathComponent("qa-data"); try FileManager.default.createDirectory(at: qa, withIntermediateDirectories: true)
      for file in try FileManager.default.contentsOfDirectory(at: temporary, includingPropertiesForKeys: [.isRegularFileKey, .fileSizeKey]) where file.pathExtension == "json" {
        let values = try file.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey])
        if values.isRegularFile == true, (values.fileSize ?? Int.max) <= 8 * 1024 * 1024 {
          let destination = qa.appendingPathComponent(file.lastPathComponent)
          if FileManager.default.fileExists(atPath: destination.path) { try FileManager.default.removeItem(at: destination) }
          try FileManager.default.copyItem(at: file, to: destination)
        }
      }
      _ = NSApplication.shared
      try snapshot(GenerationLibraryView(target: .image).environmentObject(store), size: .init(width: 940, height: 650), file: folder.appendingPathComponent("library.png"))
      try snapshot(GenerationPresetReviewView(review: review, onCancel: {}, onSaveAsNew: {}, onRefresh: { _ in }, onComplete: {}).environmentObject(store), size: .init(width: 790, height: 520), file: folder.appendingPathComponent("update-review.png"))
      let summary: [String: Any] = ["models": store.generationLibrary.modelEntries.count, "loras": store.generationLibrary.loraEntries.count,
        "guidedDefaultsAdmitted": guided.count,
        "suppliedPresets": store.generationPresets.presets.filter { $0.origin == .supplied }.count, "fast6Applied": true, "overrideSurvivedRefresh": true, "restartRestoredSelection": true]
      try JSONSerialization.data(withJSONObject: summary, options: [.prettyPrinted, .sortedKeys]).write(to: folder.appendingPathComponent("summary.json"))
    }
  }
  @MainActor private func snapshot<V: View>(_ view: V, size: NSSize, file: URL) throws {
    let host = NSHostingView(rootView: view.environment(\.colorScheme, .dark))
    let window = NSWindow(contentRect: .init(origin: .zero, size: size), styleMask: [.borderless], backing: .buffered, defer: false)
    window.isReleasedWhenClosed = false; window.appearance = NSAppearance(named: .darkAqua); window.contentView = host; defer { window.close() }
    host.frame = .init(origin: .zero, size: size); host.layoutSubtreeIfNeeded()
    RunLoop.main.run(until: Date().addingTimeInterval(0.1)); host.layoutSubtreeIfNeeded()
    let bitmap = try XCTUnwrap(host.bitmapImageRepForCachingDisplay(in: host.bounds)); host.cacheDisplay(in: host.bounds, to: bitmap)
    try XCTUnwrap(bitmap.representation(using: .png, properties: [:])).write(to: file)
  }
}
