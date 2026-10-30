import Foundation
import XCTest
@testable import StudioCore
@testable import WeeToddStudio

@MainActor final class H3RegenerateInteractionTests: XCTestCase {
  func testModelChoiceAndH3ControlsSurviveReopen() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let store = StudioStore(dataDirectory: root, restoreSession: false)
    store.videoUpscaleTool = .h3Regenerate
    store.nativeUpscalerModelLibrary = "/shared/ai-models"
    store.editH3Regenerate { $0.targetWidth = 1920; $0.targetHeight = 1080; $0.seed = 37; $0.baseProfilePath = "/profiles/ref2va.json"; $0.modelLibraryRoot = "/shared/ai-models" }
    store.saveVideoUpscalePreferences()
    let reopened = StudioStore(dataDirectory: root, restoreSession: false); reopened.loadVideoUpscalePreferences()
    XCTAssertEqual(reopened.videoUpscaleTool, .h3Regenerate)
    XCTAssertEqual(reopened.nativeUpscalerModelLibrary, "/shared/ai-models")
    XCTAssertEqual(reopened.h3RegenerateSettings, store.h3RegenerateSettings)
    XCTAssertNil(reopened.h3RegeneratePlan); XCTAssertNil(reopened.h3RegenerateResult)
  }
  func testFinishingChoosesOnlyRequestedModelsAndPreservesGeneration() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let store = StudioStore(dataDirectory: root, restoreSession: false)
    var clip = Clip(engine: .h3); clip.sourcePath = "/original/movie.mov"
    store.project.clips = [clip]; store.select(clip.id)
    let generation = store.generationRequestKey(for: clip)
    store.setVideoProcessingEnabled(true)
    XCTAssertEqual(store.selectedClip?.videoProcessing?.steps.count, 1)
    store.setVideoProcessingEngine("h3-regenerate")
    var recipe = try XCTUnwrap(store.selectedClip?.videoProcessing)
    XCTAssertTrue(recipe.hasH3Regenerate); XCTAssertFalse(recipe.hasFlashVSR)
    store.editH3VideoProcessing { $0.seed = 89; $0.prompt = "Preserve the source." }
    store.setVideoProcessingEngine("h3-regenerate-flashvsr")
    recipe = try XCTUnwrap(store.selectedClip?.videoProcessing)
    XCTAssertTrue(recipe.isSequence)
    XCTAssertEqual(recipe.steps.map(\.kind), ["h3-regenerate", "flashvsr-turbo"])
    XCTAssertEqual(recipe.steps[0].h3Regenerate?.seed, 89)
    XCTAssertEqual(recipe.steps[1].input, recipe.steps[0].id)
    store.editVideoProcessing { $0.variant = .base; $0.decoder = .vae }
    XCTAssertNoThrow(try store.selectedClip?.videoProcessing?.validate())
    XCTAssertEqual(store.generationRequestKey(for: store.selectedClip!), generation)
    let reopened = try JSONDecoder().decode(StudioProject.self, from: JSONEncoder().encode(store.project))
    XCTAssertEqual(reopened.clips.first?.videoProcessing, store.selectedClip?.videoProcessing)
    store.setVideoProcessingEngine("flashvsr-turbo")
    XCTAssertFalse(try XCTUnwrap(store.selectedClip?.videoProcessing).hasH3Regenerate)
    XCTAssertEqual(store.project.clips[0].sourcePath, clip.sourcePath)
  }
}
