import Foundation
import XCTest
@testable import StudioCore

final class NativeLTXKeyframePreparationTests:XCTestCase {
  private func fixture() throws -> (URL,StudioProject,[String:Any]) {
    let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
    addTeardownBlock { try? FileManager.default.removeItem(at:root) }
    let recipe:[String:Any]=["format":"weetodd-headless-v2","engine":"ltx25","prompt":"old",
      "components":["transformer_path":"/models/transformer"],
      "config":["pipeline_mode":"distilled","stage1_steps":8,"stage2_steps":3,"frame_rate":24,
        "width":768,"height":448,"seed":1,"duration_seconds":5],
      "conditioning":["version":1,"task":"t2v","inputs":[]]]
    try JSONSerialization.data(withJSONObject:recipe).write(to:root.appendingPathComponent("model.json"))
    var project=StudioProject(),clip=Clip()
    clip.engine = .ltx25;clip.prompt="Two people wave.";clip.duration=5;clip.generationWidth=768;clip.generationHeight=448
    clip.generationSelection=GenerationSelection(task:"t2v")
    project.clips=[clip]
    return(root,project,["profilesDirectory":root.path,"ffmpegPath":"/usr/bin/true"])
  }
  private func compose(_ project:StudioProject,_ runtime:[String:Any]) throws -> [String:Any] {
    try NativeLTXPreparation.compose(request:["project":JSONSerialization.jsonObject(with:JSONEncoder().encode(project)),
      "clipID":project.clips[0].id.uuidString,"runtime":runtime])
  }
  private func append(_ frame:Double,role:MediaRole = .keyframe,root:URL,project:inout StudioProject) throws {
    let file=root.appendingPathComponent(UUID().uuidString+".png");try Data([1]).write(to:file)
    let asset=MediaAsset(name:"Frame",kind:.image,path:file.path)
    var attachment=Attachment(assetID:asset.id,role:role);attachment.time=frame/24;attachment.strength=0.75
    project.assets.append(asset);project.clips[0].attachments.append(attachment)
  }
  func testAllCountsAndOrderedArbitraryAnchorsFreezeIDsSettingsAndNoPython() throws {
    let(root,original,runtime)=try fixture()
    for count in 0...8 {
      var project=original;project.clips[0].generationSelection?.ltx25Keyframes = .init(generatedCount:count,experimentalEnabled:true)
      let output=try compose(project,runtime),recipe=output["recipe"] as! [String:Any]
      XCTAssertEqual((recipe["config"] as! [String:Any])["generated_keyframes"] as? Int,count)
    }
    for count in 1...8 {
      var project=original;project.clips[0].generationSelection?.task="fflf"
      project.clips[0].generationSelection?.ltx25Keyframes = .init(generatedCount:2,experimentalEnabled:true)
      for frame in (1...count).reversed() { try append(Double(frame),root:root,project:&project) }
      let snapshot=project,encoder=JSONEncoder();encoder.outputFormatting = [.sortedKeys]
      let before=try encoder.encode(project),output=try compose(project,runtime)
      let recipe=output["recipe"] as! [String:Any],contract=recipe["conditioning"] as! [String:Any],inputs=contract["inputs"] as! [[String:Any]]
      XCTAssertEqual(contract["task"] as? String,"fflf")
      XCTAssertEqual(inputs.map { $0["frame_index"] as! Int },Array((1...count).reversed()))
      XCTAssertEqual(inputs.map { $0["id"] as! String },project.clips[0].attachments.map { $0.id.uuidString })
      XCTAssertEqual(project,snapshot)
      XCTAssertEqual(try encoder.encode(project),before)
      XCTAssertEqual((output["report"] as! [String:Any])["productionQualified"] as? Bool,false)
      XCTAssertEqual((output["report"] as! [String:Any])["nativePreparation"] as? String,"swift")
    }
  }
  func testNilDefaultsPreserveEndpointRecipeAndSavedSelectionReset() throws {
    let(root,original,runtime)=try fixture();var project=original
    project.clips[0].generationSelection?.task="fflf"
    try append(0,role:.first,root:root,project:&project);try append(120,role:.last,root:root,project:&project)
    let before=try compose(project,runtime),recipe=before["recipe"] as! [String:Any]
    XCTAssertNil((recipe["config"] as! [String:Any])["generated_keyframes"])
    let decoded=try JSONDecoder().decode(StudioProject.self,from:JSONEncoder().encode(project))
    XCTAssertNil(decoded.clips[0].generationSelection?.ltx25Keyframes)
    let after=try compose(decoded,runtime)
    XCTAssertEqual(try JSONSerialization.data(withJSONObject:before["recipe"]!,options:.sortedKeys),try JSONSerialization.data(withJSONObject:after["recipe"]!,options:.sortedKeys))
    var selection=GenerationSelection();selection.ltx25Keyframes = .init(generatedCount:3,experimentalEnabled:true)
    XCTAssertTrue(selection.isModified)
    XCTAssertEqual(try JSONDecoder().decode(GenerationSelection.self,from:JSONEncoder().encode(selection)),selection)
    selection.resetOverrides();XCTAssertNil(selection.ltx25Keyframes)
  }
  func testOptInDuplicateNearestEvenAndInputBoundsFailClosed() throws {
    let(root,original,runtime)=try fixture();var project=original
    project.clips[0].generationSelection?.ltx25Keyframes = .init(generatedCount:1)
    XCTAssertThrowsError(try compose(project,runtime))
    project.clips[0].generationSelection?.ltx25Keyframes?.experimentalEnabled=true
    for count in [-1,9] { project.clips[0].generationSelection?.ltx25Keyframes?.generatedCount=count;XCTAssertThrowsError(try compose(project,runtime)) }
    project.clips[0].generationSelection?.ltx25Keyframes?.generatedCount=0;project.clips[0].generationSelection?.task="fflf"
    try append(6.5,root:root,project:&project);try append(6,root:root,project:&project)
    XCTAssertThrowsError(try compose(project,runtime)) // nearest-even 6.5 collides with 6.
    project.clips[0].attachments.removeLast();try append(121,root:root,project:&project)
    XCTAssertThrowsError(try compose(project,runtime))
    project=original;project.clips[0].generationSelection?.task="fflf"
    project.clips[0].generationSelection?.ltx25Keyframes = .init(experimentalEnabled:true)
    for frame in 1...9 { try append(Double(frame),root:root,project:&project) }
    XCTAssertThrowsError(try compose(project,runtime))
  }
  func testQualifiedAutomaticTimedGeneratedCombinationFreezesSymbolicLastForPrediction() throws {
    let(root,original,runtime)=try fixture(),file=root.appendingPathComponent("model.json")
    var recipe=try JSONSerialization.jsonObject(with:Data(contentsOf:file)) as! [String:Any]
    var components=recipe["components"] as! [String:Any]
    components["duration_head_path"]=try NativeLTXAutomaticDurationTests.headFixture(at:root).path
    recipe["components"]=components
    try JSONSerialization.data(withJSONObject:recipe).write(to:file)
    var project=original;project.clips[0].generationSelection?.task="fflf"
    project.clips[0].generationSelection?.ltx25Keyframes = .init(generatedCount:2,experimentalEnabled:true)
    project.clips[0].generationSelection?.ltx25AutomaticDuration = .init(experimentalEnabled:true,minimumSeconds:1,maximumSeconds:3)
    try append(0,role:.first,root:root,project:&project)
    try append(7,root:root,project:&project)
    try append(0,role:.last,root:root,project:&project)
    let originalProject=project,encoder=JSONEncoder();encoder.outputFormatting = [.sortedKeys]
    let before=try encoder.encode(project),output=try compose(project,runtime)
    let content=output["recipe"] as! [String:Any],config=content["config"] as! [String:Any]
    XCTAssertEqual(config["generated_keyframes"] as? Int,2)
    XCTAssertEqual(config["duration_mode"] as? String,"automatic")
    let inputs=(content["conditioning"] as! [String:Any])["inputs"] as! [[String:Any]]
    XCTAssertEqual(inputs[0]["frame_index"] as? Int,0);XCTAssertEqual(inputs[1]["frame_index"] as? Int,7)
    XCTAssertEqual(inputs[2]["frame_index"] as? String,"last")
    let body:[String:Any]=["project":try JSONSerialization.jsonObject(with:JSONEncoder().encode(project)),
      "clipID":project.clips[0].id.uuidString,"runtime":runtime]
    let prepared=try NativeLTXPreparation.prepare(request:body,destination:root.appendingPathComponent("combined-export"))
    let frozen=try JSONSerialization.jsonObject(with:Data(contentsOf:URL(fileURLWithPath:prepared["recipePath"] as! String))) as! [String:Any]
    XCTAssertEqual(((frozen["conditioning"] as! [String:Any])["inputs"] as! [[String:Any]])[2]["frame_index"] as? String,"last")
    XCTAssertEqual(project,originalProject)
    XCTAssertEqual(try encoder.encode(project),before)
    project.clips[0].attachments[1].time=4
    XCTAssertThrowsError(try compose(project,runtime)) // Beyond the admitted prediction maximum.
  }
  func testInheritedCountCannotBypassOptInAndUnsupportedMixturesAreExplicit() throws {
    let(root,original,runtime)=try fixture(),file=root.appendingPathComponent("model.json")
    var recipe=try JSONSerialization.jsonObject(with:Data(contentsOf:file)) as! [String:Any]
    var config=recipe["config"] as! [String:Any];config["generated_keyframes"]=2;recipe["config"]=config
    try JSONSerialization.data(withJSONObject:recipe).write(to:file)
    XCTAssertThrowsError(try compose(original,runtime))
    var project=original;project.clips[0].generationSelection?.ltx25Keyframes = .init(experimentalEnabled:true)
    project.clips[0].generationSelection?.ltx25AutomaticDuration = .init(experimentalEnabled:true)
    XCTAssertThrowsError(try compose(project,runtime)) // This profile deliberately has no duration head.
    project=original;project.clips[0].generationSelection?.ltx25Keyframes = .init(generatedCount:8,experimentalEnabled:true)
    project.clips[0].generationWidth=4096;project.clips[0].generationHeight=4096
    XCTAssertThrowsError(try compose(project,runtime))
  }
}
