import Foundation
import XCTest
@testable import StudioCore

final class FinishingGenerationPresetTests: XCTestCase {
  func testFutureNestedPixelSchemaRejectsWholeChain() throws {
    let ref = GenerationModelRef(id: "pixel", backend: "restoration", family: "pixelCQ")
    let library = GenerationLibrarySnapshot(models: [.init(reference: ref, name: "Pixel", workspaces: [.finishing], tasks: ["pixelCQ"], availability: .installed, binding: .restoration(tool: "pixelCQ", components: [:]))])
    let recipe = VideoProcessingRecipe(pixelCQ: .init())
    var stage = GenerationPresetSnapshot(context: .init(workspace: .finishing, task: "pixelCQ"), model: ref, fields: [:]); stage.schemaVersion = 2
    let snapshot = GenerationPresetSnapshot(context: .init(workspace: .finishing, task: "finishing"), fields: ["stages": .object(["0": try .encoded(stage)])])
    XCTAssertThrowsError(try FinishingPresetRecipe.candidate(applying: snapshot, to: recipe, resolvedLibrary: library))
  }
  func testBasePresetPromotesLegacyTurboFormatWhileKeepingStageIdentityAndCanvas() throws {
    let ref = GenerationModelRef(id: "base", backend: "restoration", family: "flashVSR", variant: "base")
    let library = GenerationLibrarySnapshot(models: [.init(reference: ref, name: "Base", workspaces: [.finishing], tasks: ["flashVSR"], availability: .installed,
      binding: .restoration(tool: "flashVSR", components: ["modelRoot": "/models", "variant": "base", "decoder": "tiny"]))])
    var base = FlashVSRSettings(); base.variant = .base
    let preset = try FinishingPresetRecipe.capture(from: .init(steps: [.init(settings: base)]), models: [ref], resolvedLibrary: library)
    let original = VideoProcessingRecipe()
    let applied = try FinishingPresetRecipe.candidate(applying: preset, to: original, resolvedLibrary: library)
    XCTAssertEqual(applied.recipe.format, "weetodd-video-processing-v2"); XCTAssertEqual(applied.recipe.steps[0].settings.variant, .base)
    XCTAssertEqual(applied.recipe.steps[0].id, original.steps[0].id); XCTAssertEqual(applied.recipe.steps[0].settings.width, original.steps[0].settings.width)
  }
  func fixture() throws -> (VideoProcessingRecipe, [GenerationModelRef], GenerationLibrarySnapshot) {
    var h3 = H3RegenerateSettings(); h3.targetWidth = 1536; h3.targetHeight = 1024; h3.prompt = "Keep the source people"
    var first = VideoProcessingStep(h3Regenerate: h3); first.id = "my-first"
    var flash = FlashVSRSettings(); flash.width = 1872; flash.height = 1056; flash.fit = .fill
    var second = VideoProcessingStep(settings: flash); second.id = "my-second"; second.input = first.id
    let recipe = try VideoProcessingRecipe(steps: [first, second])
    let refs = [GenerationModelRef(id: "h3", backend: "restoration", family: "h3Regenerate"), .init(id: "flash", backend: "restoration", family: "flashVSR")]
    let library = GenerationLibrarySnapshot(models: zip(refs, ["h3Regenerate", "flashVSR"]).map { ref, tool in
      .init(reference: ref, name: tool, workspaces: [.finishing], tasks: [tool], supportsLoRAs: false, availability: .installed,
        binding: .restoration(tool: tool, components: ["modelRoot": "/models", "baseProfilePath": "/models/h3.json"]))
    })
    return (recipe, refs, library)
  }
  func testOrderedFinishingPresetsPreserveStageWiringSizeFramingAndCreativePrompt() throws {
    let (recipe, refs, library) = try fixture()
    let snapshot = try FinishingPresetRecipe.capture(from: recipe, models: refs, resolvedLibrary: library, sourceFPS: 24)
    let application = try FinishingPresetRecipe.candidate(applying: snapshot, to: recipe, resolvedLibrary: library, sourceFPS: 24)
    XCTAssertEqual(application.models, refs)
    XCTAssertEqual(application.recipe.steps.map(\.id), recipe.steps.map(\.id)); XCTAssertEqual(application.recipe.steps.map(\.input), recipe.steps.map(\.input))
    XCTAssertEqual(application.recipe.steps[0].h3Regenerate?.prompt, recipe.steps[0].h3Regenerate?.prompt)
    XCTAssertEqual(application.recipe.steps[0].h3Regenerate?.targetWidth, 1536)
    XCTAssertEqual(application.recipe.steps[1].settings.width, 1872); XCTAssertEqual(application.recipe.steps[1].settings.fit, .fill)
    let text = String(decoding: try JSONEncoder().encode(snapshot), as: UTF8.self)
    XCTAssertFalse(text.contains("/models")); XCTAssertFalse(text.contains("my-first")); XCTAssertFalse(text.contains("Keep the source people"))
  }
  func testDifferentStageLayoutAndUnsupportedSourceClockRejectAtomically() throws {
    let (recipe, refs, library) = try fixture()
    let snapshot = try FinishingPresetRecipe.capture(from: recipe, models: refs, resolvedLibrary: library, sourceFPS: 24)
    let another = VideoProcessingRecipe()
    XCTAssertThrowsError(try FinishingPresetRecipe.candidate(applying: snapshot, to: another, resolvedLibrary: library, sourceFPS: 24))
    XCTAssertThrowsError(try FinishingPresetRecipe.candidate(applying: snapshot, to: recipe, resolvedLibrary: library, sourceFPS: 30))
    XCTAssertEqual(recipe.steps.count, 2)
  }
  func testFinishingReviewListsEachChangedOwnedSetting() throws {
    let (recipe, refs, library) = try fixture()
    let saved = try FinishingPresetRecipe.capture(from: recipe, models: refs, resolvedLibrary: library, sourceFPS: 24)
    var changed = recipe; changed.steps[1].settings.temporalSigma = 1.5; changed.steps[1].settings.colorFix = .off
    let current = try FinishingPresetRecipe.capture(from: changed, models: refs, resolvedLibrary: library, sourceFPS: 24)
    let diff = GenerationPresetDiff.changes(from: saved, to: current)
    XCTAssertEqual(diff.count, 2); XCTAssertTrue(diff.contains { $0.path.hasSuffix("temporalSigma") }); XCTAssertTrue(diff.contains { $0.path.hasSuffix("colorFix") })
  }
}
