import Foundation
import StudioCore
import XCTest
@testable import WeeToddStudio

final class ModelSwitchContinuityTests: XCTestCase {
  @MainActor private func fixture() throws -> StudioStore {
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    addTeardownBlock { try? FileManager.default.removeItem(at: folder) }
    let store = StudioStore(dataDirectory: folder, restoreSession: false)
    store.project.clips = (0..<6).map { index in
      var clip = Clip(name: "Shot \(index)", engine: .ltx25)
      clip.duration = 5; clip.sourceIn = Double(index * 5); clip.sourcePath = "/scene.mp4"
      clip.attachments = [Attachment(assetID: UUID(), role: .first), Attachment(assetID: UUID(), role: .last)]
      clip.versions = [RenderVersion(path: clip.sourcePath, seed: 42, prompt: "accepted", recipePath: "")]
      clip.continuity = ClipContinuity(mode: index == 0 ? "independent" : "scene")
      return clip
    }
    for index in 1..<6 { store.project.clips[index].continuity?.sourceClipID = store.project.clips[index - 1].id }
    store.project.clips[0].soundscape = "Shared forest ambience"
    store.project.clips[0].music = "No music"
    store.project.clips[0].continuity?.boundaryImagePolicy = "strict"
    return store
  }

  // Removing model-change reconciliation must leave a mixed-engine scene and fail this test.
  @MainActor func testChangingFirstMiddleOrLastSceneShotKeepsEveryRemainingGroupValidAndUndoable() throws {
    for index in 0..<6 {
      let store = try fixture()
      store.select(store.project.clips[index].id)
      let before = store.project
      store.preparedRecipe = "/prepared.json"
      store.editClip { $0.selectLocalModel(.h3) }
      let switched = try XCTUnwrap(store.selectedClip)
      XCTAssertEqual(switched.engine, .h3)
      XCTAssertEqual(switched.continuityMode, "independent")
      XCTAssertFalse(store.project.isContinuousSceneMember(switched))
      XCTAssertEqual(store.project.clips.map(\.attachments), before.clips.map(\.attachments))
      XCTAssertEqual(store.project.clips.map(\.versions), before.clips.map(\.versions))
      XCTAssertEqual(store.project.clips.map(\.sourcePath), before.clips.map(\.sourcePath))
      XCTAssertEqual(store.project.clips.map(\.sourceIn), before.clips.map(\.sourceIn))
      XCTAssertEqual(store.project.clips.map(\.duration), before.clips.map(\.duration))
      XCTAssertNil(store.preparedRecipe)
      for clip in store.project.clips { XCTAssertTrue(store.project.continuityIssues(for: clip).isEmpty) }
      if index < 5 {
        let rightLeader = store.project.clips[index + 1]
        XCTAssertEqual(rightLeader.continuityMode, "independent")
        XCTAssertNil(rightLeader.continuity?.sourceClipID)
        if index < 4 {
          XCTAssertEqual(rightLeader.soundscape, "Shared forest ambience")
          XCTAssertEqual(rightLeader.music, "No music")
          XCTAssertEqual(rightLeader.continuity?.boundaryImagePolicy, "strict")
        }
      }
      store.undo()
      XCTAssertEqual(store.project, before)
      store.redo()
      XCTAssertEqual(store.selectedClip?.engine, .h3)
      XCTAssertTrue(store.project.continuityIssues(for: store.selectedClip!).isEmpty)
    }
  }

  @MainActor func testSwitchingProviderSeparatesSceneButPreservesSavedLocalModel() throws {
    let store = try fixture()
    store.select(store.project.clips[2].id)
    store.editClip { $0.selectGenerationProvider(.drawThings) }
    XCTAssertFalse(store.project.isContinuousSceneMember(store.selectedClip!))
    XCTAssertEqual(store.selectedClip?.continuityMode, "independent")
    store.editClip { $0.selectGenerationProvider(.local) }
    XCTAssertEqual(store.selectedClip?.engine, .ltx25)
    XCTAssertEqual(store.selectedClip?.continuityMode, "independent")
    for clip in store.project.clips { XCTAssertTrue(store.project.continuityIssues(for: clip).isEmpty) }
  }

  @MainActor func testParameterEditsAndSelectingCurrentModelPreserveScene() throws {
    let store = try fixture()
    store.select(store.project.clips[2].id)
    let before = store.project
    store.editClip { $0.selectLocalModel(.ltx25) }
    XCTAssertEqual(store.project, before)
    store.editClip { $0.seed = 123 }
    XCTAssertEqual(try store.project.continuousSceneMembers(for: store.selectedClip!).count, 6)
    XCTAssertEqual(store.project.clips.map(\.continuity), before.clips.map(\.continuity))
  }

  @MainActor func testRepairSavedIncompatibleSceneIncludingItsIndependentLeader() throws {
    for index in [0, 2, 5] {
      let store = try fixture()
      store.project.clips[index].selectLocalModel(.h3)
      let before = store.project
      let clipID = before.clips[index].id
      XCTAssertFalse(store.project.continuityIssues(for: before.clips[index]).isEmpty)
      store.separateContinuousScene(clipID: clipID)
      XCTAssertFalse(store.project.isContinuousSceneMember(store.project.clips[index]))
      for clip in store.project.clips { XCTAssertTrue(store.project.continuityIssues(for: clip).isEmpty) }
      XCTAssertEqual(store.project.clips.map(\.versions), before.clips.map(\.versions))
      XCTAssertEqual(store.project.clips.map(\.attachments), before.clips.map(\.attachments))
      store.undo()
      XCTAssertEqual(store.project, before)
    }
  }

  @MainActor func testOrdinaryContinuityIsPreservedOnModelSwitch() throws {
    for mode in ["independent", "frame", "motion"] {
      let store = try fixture()
      store.project.clips = [store.project.clips[0]]
      store.project.clips[0].continuity = ClipContinuity(mode: mode, saveContext: true)
      store.select(store.project.clips[0].id)
      let continuity = store.selectedClip?.continuity
      store.editClip { $0.selectLocalModel(.h3) }
      XCTAssertEqual(store.selectedClip?.continuity, continuity)
    }
  }
}
