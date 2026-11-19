import AppKit
import Foundation
import SwiftUI
import XCTest
import StudioCore
@testable import WeeToddStudio

final class GenerationPresetViewTests: XCTestCase {
  @MainActor func testSharedControlsAndReviewHostWithoutEditingOrGenerating() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let store = StudioStore(dataDirectory: root, restoreSession: false)
    let original = store.videoUpscaleSettings
    let controls = NSHostingView(rootView: GenerationPresetControls(target: .restoration(.flashVSR)).environmentObject(store))
    controls.frame = .init(x: 0, y: 0, width: 310, height: 220); controls.layoutSubtreeIfNeeded()
    let model = GenerationModelRef(id: "flash", backend: "restoration", family: "flashVSR")
    let before = GenerationPresetSnapshot(context: .init(workspace: .restoration, task: "flashVSR"), model: model, fields: ["steps": .integer(4)])
    var after = before; after.fields["steps"] = .integer(6)
    let review = GenerationPresetReviewToken(target: .restoration(.flashVSR), targetToken: try store.restorationPresetTargetToken(),
      presetID: "fixture", presetRevision: 1, name: "Example", before: before, after: after)
    let host = NSHostingView(rootView: GenerationPresetReviewView(review: review, onCancel: {}, onSaveAsNew: {}, onRefresh: { _ in }, onComplete: {}).environmentObject(store))
    host.frame = .init(x: 0, y: 0, width: 820, height: 500); host.layoutSubtreeIfNeeded()
    XCTAssertEqual(store.videoUpscaleSettings, original); XCTAssertFalse(store.bridge.busy); XCTAssertTrue(store.generationPresets.presets.isEmpty)
  }
}
