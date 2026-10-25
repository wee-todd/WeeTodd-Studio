import Foundation
import XCTest
@testable import StudioCore

final class H3TwoStageSettingsTests: XCTestCase {
  private func legacyStage() throws -> H3TwoStageSettings {
    try JSONDecoder().decode(H3TwoStageSettings.self,from:Data("{\"draftScale\":2,\"strength\":0.35,\"evaluations\":3,\"resizeMethod\":\"bilinear\"}".utf8))
  }
  private var recipe: [String:Any] {
    ["components":["task":"t2va"],"config":["width":1536,"height":864,"duration_seconds":3.0,"steps":21] as [String:Any],"prompt":"test"]
  }
  func testNewAndOldJointSettingsRoundTripWithoutChangingFinalCanvas() throws {
    let old = try JSONDecoder().decode(H3JointSettings.self, from: Data("{\"saveFullLatents\":true}".utf8))
    XCTAssertNil(old.twoStage)
    let settings = H3JointSettings(twoStage:try legacyStage())
    XCTAssertEqual(try JSONDecoder().decode(H3JointSettings.self,from:JSONEncoder().encode(settings)),settings)
    let output = try NativeH3JointPreparation.apply(settings,to:recipe,continuityMode:"independent")
    XCTAssertEqual((output["config"] as? [String:Any])?["width"] as? Int,1536)
    XCTAssertEqual((output["config"] as? [String:Any])?["height"] as? Int,864)
    let stage = try XCTUnwrap(output["two_stage"] as? [String:Any])
    XCTAssertEqual(stage["evaluations"] as? Int,3)
    XCTAssertEqual(stage["draft_scale"] as? Double,2)
    XCTAssertEqual(stage["refinement_adapter_policy"] as? String,"standard")
    let canvas = try H3TwoStageSettings().draftCanvas(width:1536,height:864,durationSeconds:3)
    XCTAssertEqual(canvas.width,768);XCTAssertEqual(canvas.height,448)
  }
  func testNewUnlinkedDefaultCannotSilentlyExportInterpolation() throws {
    XCTAssertThrowsError(try NativeH3JointPreparation.apply(.init(twoStage:H3TwoStageSettings()),to:recipe,continuityMode:"independent"))
  }
  func testInvalidFinalAndIncompatibleModesFailBeforeExport() throws {
    let settings = H3JointSettings(twoStage:try legacyStage())
    XCTAssertThrowsError(try NativeH3JointPreparation.apply(settings,to:recipe,continuityMode:"motion"))
    var invalid = recipe;invalid["config"]=["width":1920,"height":1080,"duration_seconds":3.0,"steps":21]
    XCTAssertThrowsError(try NativeH3JointPreparation.apply(settings,to:invalid,continuityMode:"independent"))
    invalid=recipe;invalid["fasth3"]=["version":1]
    XCTAssertThrowsError(try NativeH3JointPreparation.apply(settings,to:invalid,continuityMode:"independent"))
    XCTAssertThrowsError(try H3TwoStageSettings().draftCanvas(width:1920,height:1088,durationSeconds:5))
    var bad = try legacyStage();bad.evaluations=0
    XCTAssertThrowsError(try NativeH3JointPreparation.apply(.init(twoStage:bad),to:recipe,continuityMode:"independent"))
  }
  func testDefaultModelLinksSelectTaskAndFreezeIndependentSettings() throws {
    var links=H3TwoStageModelLinks();links.flTurboPath="/fl.safetensors";links.referenceTurboPath="/ref.safetensors"
    links.learnedUpscalerPath="/learned.safetensors";links.learnedUpscalerHeaderSHA256=String(repeating:"a",count:64)
    let fl=links.settings(task:"fflf"),ref=links.settings(task:"ref2va")
    XCTAssertEqual(H3TwoStageSettings.turboTask(for:"a2v"),"ref2va")
    XCTAssertEqual(links.settings(task:"a2v").draftTurboPath,"/ref.safetensors")
    XCTAssertNotEqual(H3TwoStageSettings.turboTask(for:"i2v"),H3TwoStageSettings.turboTask(for:"a2v"))
    XCTAssertEqual(fl.draftTurboPath,"/fl.safetensors");XCTAssertEqual(ref.draftTurboPath,"/ref.safetensors")
    links.setTurboPath("/new-ref.safetensors",task:"a2v")
    XCTAssertEqual(links.flTurboPath,"/fl.safetensors")
    XCTAssertEqual(links.referenceTurboPath,"/new-ref.safetensors")
    XCTAssertEqual(links.settings(task:"a2v").draftTurboPath,"/new-ref.safetensors")
    XCTAssertEqual(ref.draftTurboPath,"/ref.safetensors")
    XCTAssertTrue(fl.useTurboDraft);XCTAssertNil(fl.resizeMethod)
    XCTAssertEqual(try JSONDecoder().decode(H3TwoStageSettings.self,from:JSONEncoder().encode(fl)),fl)
    links.flTurboPath="/replacement.safetensors";XCTAssertEqual(fl.draftTurboPath,"/fl.safetensors")
    let old=try legacyStage();XCTAssertFalse(old.useTurboDraft);XCTAssertEqual(old.resizeMethod,.bilinear)
    let oldLearned=try JSONDecoder().decode(H3TwoStageSettings.self,from:Data("{\"learnedUpscalerPath\":\"/saved.safetensors\"}".utf8))
    XCTAssertFalse(oldLearned.useTurboDraft);XCTAssertNil(oldLearned.resizeMethod)
  }
  func testEditedSecondPassStrengthSurvivesReopeningAndExportsExactly() throws {
    var stage=try legacyStage();stage.strength=0.625
    let saved=try JSONEncoder().encode(H3JointSettings(twoStage:stage))
    let restored=try JSONDecoder().decode(H3JointSettings.self,from:saved)
    let output=try NativeH3JointPreparation.apply(restored,to:recipe,continuityMode:"independent")
    let refinement=try XCTUnwrap(output["two_stage"] as? [String:Any])
    XCTAssertEqual(refinement["strength"] as? Double,0.625)
    XCTAssertEqual(refinement["evaluations"] as? Int,3)
    let config=try XCTUnwrap(output["config"] as? [String:Any])
    XCTAssertEqual(config["width"] as? Int,1536);XCTAssertEqual(config["height"] as? Int,864)
    for value in [0.0,-0.1,1.01,Double.nan,Double.infinity] {
      stage.strength=value
      XCTAssertThrowsError(try NativeH3JointPreparation.apply(.init(twoStage:stage),to:recipe,continuityMode:"independent"))
    }
  }
  func testLearnedPathMapsForSavedProjects() {
    var settings = H3TwoStageSettings();settings.draftTurboPath="/old/turbo.safetensors";settings.learnedUpscalerPath="/old/model.safetensors"
    settings.learnedUpscalerHeaderSHA256=String(repeating:"a",count:64);settings.resizeMethod=nil
    var selection = GenerationSelection();selection.h3Joint = .init(twoStage:settings)
    selection.mapH3CreativePaths { $0.replacingOccurrences(of:"/old/",with:"/new/") }
    XCTAssertEqual(selection.h3Joint?.twoStage?.learnedUpscalerPath,"/new/model.safetensors")
    XCTAssertEqual(selection.h3Joint?.twoStage?.draftTurboPath,"/new/turbo.safetensors")
    selection.resetOverrides()
    XCTAssertNil(selection.h3Joint)
  }
}
