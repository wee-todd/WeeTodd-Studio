import Foundation
import StudioCore
import XCTest
@testable import WeeToddStudio

final class GuidedEndpointTests: XCTestCase {
  @MainActor func fixture() throws -> (StudioStore, Clip) {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    let store = StudioStore(dataDirectory: root, restoreSession: false)
    let generation = try JSONDecoder().decode(GenerationDescriptor.self, from: Data(#"{"supportedTasks":["control"],"controls":{"stepsEditable":false,"refinementStepsEditable":false,"cfgEditable":false,"shiftEditable":false,"stepsExplanation":"","cfgExplanation":"","shiftExplanation":""},"presets":[]}"#.utf8))
    store.profiles = [ModelProfile(id: "union", name: "Union", engine: "ltx25", task: "control", generation: generation)]
    var clip = Clip(engine: .ltx25)
    clip.profileID = "union"; clip.generationSelection = GenerationSelection(task: "control")
    clip.generationSelection?.ltx25SingleStage = .init(experimentalEnabled: true)
    clip.generationSelection?.ltx25Keyframes = .init(generatedCount: 1, experimentalEnabled: true)
    clip.attachments = [Attachment(assetID: UUID(), role: .control)]
    store.project.clips = [clip]
    store.generationDescriptions[clip.id] = ["studioEngine": "ltx25", "studioTask": "control",
      "studioProfile": "union", "studioInput": store.generationRequestKey(for: clip),
      "generation": ["supportedTasks": ["control"], "ordinaryKeyframesAvailable": true]]
    return (store, clip)
  }

  @MainActor func testResolvedSingleStageGuideAllowsEndpointsWithoutChangingControlTask() throws {
    let (store, clip) = try fixture()
    XCTAssertTrue(store.supportsEndpoint(.first, for: clip))
    XCTAssertTrue(store.supportsEndpoint(.last, for: clip))
    XCTAssertTrue(store.assignEndpoint(MediaAsset(name: "First", kind: .image), to: clip.id, role: .first))
    XCTAssertEqual(store.project.clips[0].inferredTask, "control")
    XCTAssertEqual(store.project.clips[0].attachments.filter { $0.role == .control }.count, 1)
    XCTAssertEqual(store.project.clips[0].generationSelection?.ltx25Keyframes?.generatedCount, 1)
  }

  @MainActor func testStaleDescriptionsAndTwoStageControlsDoNotEnableEndpoints() throws {
    let (store, clip) = try fixture()
    store.generationDescriptions[clip.id]?["studioProfile"] = "different-profile"
    XCTAssertFalse(store.supportsEndpoint(.first, for: clip))
    store.generationDescriptions[clip.id]?["studioProfile"] = "union"
    store.generationDescriptions[clip.id]?["generation"] = ["supportedTasks": ["control"], "ordinaryKeyframesAvailable": false]
    XCTAssertFalse(store.supportsEndpoint(.last, for: clip))
    store.runtime.nativeLTX25Enabled = false
    store.generationDescriptions[clip.id]?["generation"] = ["supportedTasks": ["control"], "ordinaryKeyframesAvailable": true]
    XCTAssertFalse(store.supportsEndpoint(.first, for: clip))
  }

  @MainActor func testChangedControlSettingsDoNotReuseOldEndpointCapabilities() throws {
    let (store, original) = try fixture()
    var changed = original
    changed.generationSelection?.ltx25SingleStage?.experimentalEnabled = false
    XCTAssertFalse(store.supportsEndpoint(.first, for: changed))
    XCTAssertFalse(store.supportsEndpoint(.last, for: changed))
  }

  @MainActor func testInheritedControlProfileKeepsNilSelectionAfterEndpointAssignment() throws {
    let (store, original) = try fixture()
    var clip = original
    clip.generationSelection = nil
    store.project.clips = [clip]
    store.generationDescriptions[clip.id]?["studioInput"] = store.generationRequestKey(for: clip)
    XCTAssertTrue(store.assignEndpoint(MediaAsset(name: "First", kind: .image), to: clip.id, role: .first))
    XCTAssertNil(store.project.clips[0].generationSelection)
    XCTAssertEqual(store.project.clips[0].inferredTask, "control")
    XCTAssertEqual(store.project.clips[0].attachments.filter { $0.role == .control }.count, 1)
  }
}
