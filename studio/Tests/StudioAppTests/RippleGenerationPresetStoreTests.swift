import Foundation
import XCTest
import StudioCore
@testable import WeeToddStudio

final class RippleGenerationPresetStoreTests: XCTestCase {
  @MainActor func testPerDraftModelReachesBothRoutesWithoutChangingGlobalRuntimeAndAssociationStaysOutsideDraft() async throws {
    for native in [false, true] {
      let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
      try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
      defer { try? FileManager.default.removeItem(at: root) }
      let profile = root.appendingPathComponent("profile.json"), adapter = root.appendingPathComponent("ripple.safetensors")
      try Data("{}".utf8).write(to: profile); try Data([0]).write(to: adapter)
      var commands: [String] = [], runtimes: [RuntimeSettings] = []
      let store = StudioStore(dataDirectory: root, restoreSession: false, invocation: { command, runtime, _, _ in
        commands.append(command); runtimes.append(runtime)
        if command == "ripple-native-prepare" { return ["recipePath": "/prepared.json"] }
        throw StudioError.invalid("Stop before inference")
      })
      store.runtime.nativeRippleEnabled = native; store.runtime.rippleProfileID = "/global/profile.json"; store.runtime.rippleAdapterPath = "/global/adapter.safetensors"
      let ref = GenerationModelRef(id: "paired-ripple", backend: "ripple", family: "ltx25", variant: "distilled")
      var index = GenerationLibraryIndex()
      try index.registerModel(.init(reference: ref, name: "Ripple", workspaces: [.ripple], tasks: ["ripple"], supportsLoRAs: false, availability: .installed,
        binding: .ripple(profileID: profile.path, adapterPath: adapter.path)), alias: "fixture")
      try store.persistGenerationLibrary(index)
      var clip = Clip(engine: .movie); clip.sourcePath = "/source.mov"; clip.rippleDraft = .init(clip: clip, frameRate: 24)
      clip.rippleDraft?.references[0].path = "/edited.png"; clip.rippleDraft?.libraryModel = ref
      store.project.clips = [clip]; store.selectedClipID = clip.id; store.rippleClipID = clip.id
      let draft = try XCTUnwrap(clip.rippleDraft), fingerprint = draft.inputFingerprint
      let preset = StudioGenerationPreset(name: "Ripple", snapshot: try store.captureRipplePreset())
      store.setRipplePresetSelection(.init(preset)); XCTAssertEqual(store.rippleClip?.rippleDraft?.inputFingerprint, fingerprint)
      store.setRipplePresetSelection(nil); XCTAssertEqual(store.rippleClip?.rippleDraft, draft)
      await store.generateRipple()
      XCTAssertEqual(commands, native ? ["ripple-native-prepare", "ltx-native-render"] : ["ripple-generate"])
      XCTAssertTrue(runtimes.allSatisfy { $0.rippleProfileID == profile.path && $0.rippleAdapterPath == adapter.path })
      XCTAssertEqual(store.runtime.rippleProfileID, "/global/profile.json"); XCTAssertEqual(store.runtime.rippleAdapterPath, "/global/adapter.safetensors")
      XCTAssertTrue(store.rippleClip?.rippleTakes?.isEmpty ?? true)
    }
  }
}
