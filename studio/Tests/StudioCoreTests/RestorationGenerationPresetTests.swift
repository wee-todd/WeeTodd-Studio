import Foundation
import InferenceContracts
import XCTest
@testable import StudioCore

final class RestorationGenerationPresetTests: XCTestCase {
  func testCaptureRejectsARecipeWhosePrecisionDiffersFromItsModel() throws {
    let ref = GenerationModelRef(id: "reca-int8", backend: "restoration", family: "reCaVSR", precision: "int8ConvRot")
    let library = GenerationLibrarySnapshot(models: [.init(reference: ref, name: "ReCaVSR", workspaces: [.restoration], tasks: ["reCaVSR"], availability: .installed,
      binding: .restoration(tool: "reCaVSR", components: ["modelDirectory": "/models/reca", "precision": "int8ConvRot"]))])
    var state = RestorationPresetState(tool: .reCaVSR); state.model = ref; state.reCa.precision = .bf16
    XCTAssertThrowsError(try RestorationPresetRecipe.capture(from: state, resolvedLibrary: library))
  }
  func library(_ tool: RestorationPresetTool) -> GenerationLibrarySnapshot {
    .init(models: [.init(reference: .init(id: tool.rawValue, backend: "restoration", family: tool.rawValue), name: tool.rawValue,
      workspaces: [.restoration, .finishing], tasks: [tool.rawValue], supportsLoRAs: false, availability: .installed,
      binding: .restoration(tool: tool.rawValue, components: ["modelRoot": "/models", "baseProfilePath": "/models/h3.json"]))])
  }
  func testAllRestorationToolsPreserveCanvasFramingClockRuntimeAndCreativeInputs() throws {
    for tool in RestorationPresetTool.allCases {
      let library = library(tool)
      var state = RestorationPresetState(tool: tool); state.model = library.modelEntries[0].reference; state.sourceFPS = 24
      state.flash.width = 1872; state.flash.height = 1056; state.flash.fit = .fill; state.flash.preserveAudio = false
      state.h3.targetWidth = 1872; state.h3.targetHeight = 1056; state.h3.fit = .fill; state.h3.prompt = "My subjects"
      state.reCa.residency = .resident
      let snapshot = try RestorationPresetRecipe.capture(from: state, resolvedLibrary: library)
      let candidate = try RestorationPresetRecipe.candidate(applying: snapshot, to: state, resolvedLibrary: library)
      XCTAssertEqual(candidate.flash.width, state.flash.width); XCTAssertEqual(candidate.flash.height, state.flash.height)
      XCTAssertEqual(candidate.flash.fit, state.flash.fit); XCTAssertEqual(candidate.flash.preserveAudio, state.flash.preserveAudio)
      XCTAssertEqual(candidate.h3.targetWidth, state.h3.targetWidth); XCTAssertEqual(candidate.h3.targetHeight, state.h3.targetHeight)
      XCTAssertEqual(candidate.h3.fit, state.h3.fit); XCTAssertEqual(candidate.h3.prompt, state.h3.prompt)
      XCTAssertEqual(candidate.h3.sourceMode, state.h3.sourceMode); XCTAssertEqual(candidate.reCa.residency, .resident)
      XCTAssertEqual(candidate.sourceFPS, 24)
      let text = String(decoding: try JSONEncoder().encode(snapshot), as: UTF8.self)
      XCTAssertFalse(text.contains("/models")); XCTAssertFalse(text.contains("My subjects")); XCTAssertNil(snapshot.fields["width"])
    }
  }
  func testRelayReCaScaleAndH3ClockCannotBeChangedImplicitly() throws {
    for tool in [RestorationPresetTool.relayVSR, .reCaVSR] {
      let library = library(tool); var state = RestorationPresetState(tool: tool); state.model = library.modelEntries[0].reference
      var snapshot = try RestorationPresetRecipe.capture(from: state, resolvedLibrary: library)
      snapshot.fields["scale"] = tool == .relayVSR ? .integer(4) : .number(2)
      XCTAssertThrowsError(try RestorationPresetRecipe.candidate(applying: snapshot, to: state, resolvedLibrary: library))
    }
    let library = library(.h3Regenerate); var state = RestorationPresetState(tool: .h3Regenerate)
    state.model = library.modelEntries[0].reference; state.sourceFPS = 24
    let snapshot = try RestorationPresetRecipe.capture(from: state, resolvedLibrary: library)
    state.sourceFPS = 30
    XCTAssertThrowsError(try RestorationPresetRecipe.candidate(applying: snapshot, to: state, resolvedLibrary: library))
    XCTAssertEqual(state.sourceFPS, 30)
  }
  func testH3EncoderAndPosteriorSeedAreReviewedAndValidatedTogether() throws {
    let library = library(.h3Regenerate); var state = RestorationPresetState(tool: .h3Regenerate)
    state.model = library.modelEntries[0].reference; state.sourceFPS = 24
    var snapshot = try RestorationPresetRecipe.capture(from: state, resolvedLibrary: library)
    snapshot.fields["posteriorSeed"] = .automatic
    XCTAssertThrowsError(try RestorationPresetRecipe.candidate(applying: snapshot, to: state, resolvedLibrary: library))
    snapshot.fields["prompt"] = .string("Do not replace my creative prompt")
    XCTAssertThrowsError(try RestorationPresetRecipe.candidate(applying: snapshot, to: state, resolvedLibrary: library))
  }
  func testFullWidthRestorationSeedsRoundTripExactly() throws {
    let library = library(.h3Regenerate); var state = RestorationPresetState(tool: .h3Regenerate)
    state.model = library.modelEntries[0].reference; state.h3.seed = UInt64.max; state.h3.posteriorSeed = UInt64.max
    let snapshot = try RestorationPresetRecipe.capture(from: state, resolvedLibrary: library)
    let restored = try JSONDecoder().decode(GenerationPresetSnapshot.self, from: JSONEncoder().encode(snapshot))
    let applied = try RestorationPresetRecipe.candidate(applying: restored, to: state, resolvedLibrary: library)
    XCTAssertEqual(applied.h3.seed, UInt64.max); XCTAssertEqual(applied.h3.posteriorSeed, UInt64.max)
  }
}
