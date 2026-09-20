import Foundation
import StudioCore
import XCTest
@testable import WeeToddStudio

final class RippleStoreTests: XCTestCase {
  @MainActor private func store(invocation: Bridge.Invocation? = nil) throws -> StudioStore {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    addTeardownBlock { try? FileManager.default.removeItem(at: directory) }
    let store = StudioStore(dataDirectory: directory, restoreSession: false, invocation: { command, runtime, payload, output in
      // Draft edits schedule a separate timeline preview mix. This fixture has no
      // playable media; keep that background command outside Ripple assertions.
      if command == "audio-mix" { throw CancellationError() }
      guard ["ripple-inspect", "ripple-frame", "ripple-generate"].contains(command),
        let invocation else {
        XCTFail("Unexpected bridge command in Ripple fixture: \(command)")
        throw StudioError.invalid("Unexpected test bridge command")
      }
      return try await invocation(command, runtime, payload, output)
    })
    addTeardownBlock {
      await MainActor.run { store.invalidateTimelinePlayback() }
    }
    var clip = Clip(name: "Source", engine: .movie)
    clip.sourcePath = "/source.mp4"; clip.sourceIn = 2.5; clip.duration = 4.8
    store.project.clips = [clip]; store.selectedClipID = clip.id
    store.openRipple()
    store.updateRipple { $0.references[0].path = "/edited.png" }
    return store
  }
  private static var result: [String: Any] {
    ["video_path": "/new-take.mp4", "duration": 4.8, "frames": 116,
      "frame_rate": 24.0, "width": 768, "height": 448, "has_audio": false,
      "receipt_path": "/take/receipt.json", "artifacts_directory": "/take",
      "frozen_references": [["frame": 0, "path": "/take/frozen-frame0.png", "strength": 1.0]],
      "source_sha256": String(repeating: "a", count: 64)]
  }
  @MainActor func testGenerationRetainsSourceUntilExplicitApplyAndRestoresOriginalInterval() async throws {
    let store = try store { command, _, payload, _ in
      XCTAssertEqual(command, "ripple-generate")
      let request = try XCTUnwrap(payload["ripple"] as? [String: Any])
      XCTAssertEqual(request["source_start"] as? Double, 2.5)
      XCTAssertEqual(request["audio_policy"] as? String, "preserve")
      return Self.result
    }
    await store.generateRipple()
    XCTAssertNil(store.error)
    XCTAssertEqual(store.selectedClip?.sourcePath, "/source.mp4")
    XCTAssertEqual(store.selectedClip?.sourceIn, 2.5)
    let take = try XCTUnwrap(store.selectedClip?.rippleTakes?.first)
    XCTAssertFalse(take.hasAudio, "Silent inputs must remain valid")
    XCTAssertEqual(take.draft.references[0].path, "/take/frozen-frame0.png")
    XCTAssertEqual(take.submittedDraftFingerprint, store.selectedClip?.rippleDraft?.inputFingerprint)
    XCTAssertTrue(store.canApplyRipple(take, to: store.selectedClip!))
    XCTAssertEqual(store.project.assets.last?.path, take.path)
    store.applyRipple(take)
    XCTAssertEqual(store.selectedClip?.sourcePath, "/new-take.mp4")
    XCTAssertEqual(store.selectedClip?.sourceIn, 0)
    XCTAssertEqual(store.selectedClip?.duration, 4.8)
    XCTAssertTrue(store.selectedClip?.hasReviewedReusedTake == true,
      "Explicit Ripple application must be a usable reviewed take for native as well as imported clips")
    XCTAssertEqual(store.selectedClip?.versions.first?.path, "/source.mp4")
    XCTAssertEqual(store.selectedClip?.versions.first?.usableSourceIn, 2.5)
    store.restoreRippleTakeInputs(take)
    XCTAssertEqual(store.selectedClip?.sourcePath, "/source.mp4")
    XCTAssertEqual(store.selectedClip?.sourceIn, 2.5)
    XCTAssertTrue(store.canApplyRipple(take, to: store.selectedClip!))
    XCTAssertEqual(store.selectedClip?.rippleDraft?.references[0].path, "/take/frozen-frame0.png")
    XCTAssertEqual(try store.selectedClip?.rippleDraft?.bridgeObject()["source_sha256"] as? String,
      String(repeating: "a", count: 64))
  }
  @MainActor func testChangedDraftRetainsTakeButCannotApplyIt() async throws {
    weak var active: StudioStore?
    let store = try store { _, _, _, _ in
      active?.updateRipple { $0.seed += 1 }
      return Self.result
    }
    active = store
    await store.generateRipple()
    let take = try XCTUnwrap(store.selectedClip?.rippleTakes?.first)
    XCTAssertNil(store.rippleSelectedTakeID)
    XCTAssertFalse(store.canApplyRipple(take, to: store.selectedClip!))
    store.applyRipple(take)
    XCTAssertEqual(store.selectedClip?.sourcePath, "/source.mp4")
  }
  @MainActor func testReopenedIdenticalDocumentCannotReceiveLateTake() async throws {
    weak var active: StudioStore?
    let store = try store { _, _, _, _ in
      if let active { try active.replaceDocument(active.project, url: nil, isDirty: false) }
      return Self.result
    }
    active = store
    await store.generateRipple()
    XCTAssertNil(store.selectedClip?.rippleTakes)
    XCTAssertTrue(store.project.assets.isEmpty)
    XCTAssertTrue(store.notice.contains("destination movie changed"))
  }
  @MainActor func testRestoredTakeUsesFrozenImageAfterExternalEditIsDeleted() async throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    let external = directory.appendingPathComponent("external-edit.png")
    let frozen = directory.appendingPathComponent("frozen-edit.png")
    try Data("edited image snapshot".utf8).write(to: external)
    try FileManager.default.copyItem(at: external, to: frozen)
    let store = try store { _, _, _, _ in
      var result = Self.result
      result["frozen_references"] = [["frame": 0, "path": frozen.path, "strength": 1.0]]
      return result
    }
    store.updateRipple { $0.references[0].path = external.path }
    await store.generateRipple()
    let take = try XCTUnwrap(store.selectedClip?.rippleTakes?.first)
    XCTAssertTrue(store.canApplyRipple(take, to: store.selectedClip!))
    try FileManager.default.removeItem(at: external)
    store.restoreRippleTakeInputs(take)
    let request = try XCTUnwrap(store.selectedClip?.rippleDraft).bridgeObject()
    let references = try XCTUnwrap(request["references"] as? [[String: Any]])
    XCTAssertEqual(references[0]["path"] as? String, frozen.path)
    XCTAssertTrue(FileManager.default.fileExists(atPath: frozen.path))
    XCTAssertFalse(FileManager.default.fileExists(atPath: external.path))
  }
  @MainActor func testMissingFrozenReferencesCannotCreateUnreplayableTake() async throws {
    let store = try store { _, _, _, _ in
      var result = Self.result; result.removeValue(forKey: "frozen_references"); return result
    }
    await store.generateRipple()
    XCTAssertNil(store.selectedClip?.rippleTakes)
    XCTAssertTrue(store.error?.contains("frozen replay") == true)
  }
  @MainActor func testChangedTrimPreventsApply() async throws {
    let store = try store { _, _, _, _ in Self.result }
    await store.generateRipple()
    let take = try XCTUnwrap(store.selectedClip?.rippleTakes?.first)
    store.project.clips[0].sourceIn += 1
    XCTAssertFalse(store.canApplyRipple(take, to: store.project.clips[0]))
    store.applyRipple(take)
    XCTAssertEqual(store.project.clips[0].sourceIn, 3.5)
  }
  @MainActor func testExtractionUsesExactRelativeFrameAndRejectsChangedDraft() async throws {
    weak var active: StudioStore?
    let store = try store { command, _, payload, _ in
      XCTAssertEqual(command, "ripple-frame")
      XCTAssertEqual((payload["ripple"] as? [String: Any])?["frame"] as? Int, 17)
      active?.updateRipple { $0.references[0].frame = 18 }
      return ["image_path": "/frame17.png"]
    }
    active = store
    store.updateRipple { $0.references[0].frame = 17 }
    await store.extractRippleFrame(referenceID: store.rippleClip!.rippleDraft!.references[0].id)
    XCTAssertEqual(store.rippleClip?.rippleDraft?.references[0].originalPath, "")
    XCTAssertTrue(store.notice.contains("inputs changed"))
  }
  @MainActor func testAddReferenceBindsPlayheadExactlyAndReusesExistingFrame() throws {
    let store = try store()
    let id = try store.addRippleReference(frame: 23)
    XCTAssertEqual(store.rippleClip?.rippleDraft?.references.last?.frame, 23)
    XCTAssertEqual(try store.addRippleReference(frame: 23), id)
    XCTAssertEqual(store.rippleClip?.rippleDraft?.references.count, 2)
    let firstID = store.rippleClip!.rippleDraft!.references[0].id
    store.removeRippleReference(firstID)
    XCTAssertEqual(store.rippleClip?.rippleDraft?.references.count, 2)
    for frame in 1...7 { try store.addRippleReference(frame: frame) }
    XCTAssertEqual(store.rippleClip?.rippleDraft?.references.count, 9)
    XCTAssertThrowsError(try store.addRippleReference(frame: 99))
    XCTAssertThrowsError(try store.addRippleReference(frame: -1))
  }
  @MainActor func testRequiredFirstFrameCannotMoveAndMovingOtherFrameClearsOldImages() throws {
    let store = try store()
    let first = store.rippleClip!.rippleDraft!.references[0].id
    XCTAssertThrowsError(try store.setRippleReferenceFrame(first, frame: 2))
    let second = try store.addRippleReference(frame: 12)
    store.updateRipple { $0.references[1].path = "/old-edit.png"; $0.references[1].originalPath = "/old-source.png" }
    XCTAssertThrowsError(try store.setRippleReferenceFrame(second, frame: 0))
    try store.setRippleReferenceFrame(second, frame: 18)
    XCTAssertEqual(store.rippleClip?.rippleDraft?.references[1].frame, 18)
    XCTAssertEqual(store.rippleClip?.rippleDraft?.references[1].path, "")
    XCTAssertEqual(store.rippleClip?.rippleDraft?.references[1].originalPath, "")
  }
  @MainActor func testInitialInspectAdoptsSourceCadenceAndAspectButAssignedImagesKeepTheirGrid() async throws {
    let store = try store { _, _, _, _ in
      ["source_frame_rate": 30.0, "source_preview_start": 2.5333333333,
       "source": ["width": 1080, "height": 1920], "has_audio": true]
    }
    store.updateRipple { $0.references[0].path = "" }
    await store.inspectRipple()
    XCTAssertEqual(store.rippleClip?.rippleDraft?.frameRate, 30)
    XCTAssertEqual(store.rippleClip?.rippleDraft?.sourcePreviewStart, 2.5333333333)
    XCTAssertEqual(store.rippleClip?.rippleDraft?.width, 1088)
    XCTAssertEqual(store.rippleClip?.rippleDraft?.height, 1920)
    store.updateRipple { $0.references[0].path = "/edited.png"; $0.frameRate = 24 }
    await store.inspectRipple()
    XCTAssertEqual(store.rippleClip?.rippleDraft?.frameRate, 24)
  }
  @MainActor func testRuntimePathsRemainOutOfClipAndOrdinaryGenerationSettings() throws {
    var settings = RuntimeSettings.defaults()
    settings.rippleAdapterPath = "/weights/ripple.safetensors"; settings.rippleProfileID = "/profiles/ltx25.json"
    let decoded = try JSONDecoder().decode(RuntimeSettings.self, from: JSONEncoder().encode(settings))
    XCTAssertEqual(decoded.rippleAdapterPath, settings.rippleAdapterPath)
    XCTAssertNil(settings.generationSettings.rippleAdapterPath)
    XCTAssertNil(settings.generationSettings.rippleProfileID)
  }
}
