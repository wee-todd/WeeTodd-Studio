import Foundation
import XCTest
@testable import StudioCore

final class NativeLTXSceneImagePreparationTests:XCTestCase {
  private func fixture() throws -> (URL,StudioProject,[String:Any]) {
    let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
    addTeardownBlock { try? FileManager.default.removeItem(at:root) }
    let recipe:[String:Any]=["format":"weetodd-headless-v2","engine":"ltx25","prompt":"old",
      "components":["transformer_path":"/models/transformer"],
      "config":["pipeline_mode":"distilled","stage1_steps":8,"stage2_steps":3,"frame_rate":24,
        "width":512,"height":256,"seed":1,"duration_seconds":5],
      "conditioning":["version":1,"task":"t2v","inputs":[]]]
    try JSONSerialization.data(withJSONObject:recipe).write(to:root.appendingPathComponent("model.json"))
    var project=StudioProject(),first=Clip()
    first.engine = .ltx25;first.prompt="A warrior waits.";first.duration=2
    first.generationWidth=512;first.generationHeight=256;first.seed=43
    first.generationSelection=GenerationSelection(task:"t2v")
    var second=first;second.id=UUID();second.duration=3;second.seed=44;second.prompt="The warrior turns."
    second.continuity=ClipContinuity(mode:"scene",sourceClipID:first.id)
    project.clips=[first,second]
    return(root,project,["profilesDirectory":root.path,"ffmpegPath":"/usr/bin/true"])
  }
  private func body(_ project:StudioProject,_ runtime:[String:Any]) throws -> [String:Any] {
    ["project":try JSONSerialization.jsonObject(with:JSONEncoder().encode(project)),
      "clipID":project.clips[0].id.uuidString,"runtime":runtime]
  }
  private func append(_ frame:Double,role:MediaRole = .keyframe,clip:Int,root:URL,project:inout StudioProject) throws {
    let file=root.appendingPathComponent(UUID().uuidString+".png");try Data([1]).write(to:file)
    let asset=MediaAsset(name:"Anchor",kind:.image,path:file.path)
    var attachment=Attachment(assetID:asset.id,role:role);attachment.time=frame/24;attachment.strength=0.75
    project.assets.append(asset);project.clips[clip].attachments.append(attachment)
    project.clips[clip].generationSelection?.task="fflf"
  }
  private func images(_ result:[String:Any]) -> [[String:Any]] {
    ((result["recipe"] as! [String:Any])["conditioning"] as! [String:Any])["inputs"] as! [[String:Any]]
  }
  func testVersionTwoGlobalOrderSymbolicLastSourceIdentityAndProjectAreFrozen() throws {
    let(root,original,runtime)=try fixture();var project=original
    try append(0,role:.last,clip:1,root:root,project:&project)
    try append(7,clip:1,root:root,project:&project)
    try append(6.5,clip:0,root:root,project:&project)
    try append(0,role:.last,clip:0,root:root,project:&project)
    let snapshot=project,encoder=JSONEncoder();encoder.outputFormatting = [.sortedKeys]
    let before=try encoder.encode(project),request=try body(project,runtime)
    let output=try NativeLTXPreparation.compose(request:request),recipe=output["recipe"] as! [String:Any]
    let scene=recipe["scene"] as! [String:Any],inputs=images(output)
    XCTAssertEqual(scene["version"] as? Int,2)
    XCTAssertEqual(inputs.map { $0["frame_index"] as! Int },[6,47,55,119])
    let expected=[project.clips[0].attachments[0],project.clips[0].attachments[1],project.clips[1].attachments[1],project.clips[1].attachments[0]]
    XCTAssertEqual(inputs.map { $0["id"] as! String },expected.enumerated().map { index,attachment in
      project.clips[index<2 ? 0:1].id.uuidString+":"+attachment.id.uuidString
    })
    XCTAssertTrue((scene["segments"] as! [[String:Any]]).allSatisfy { $0["image_input"] == nil })
    XCTAssertEqual((output["report"] as! [String:Any])["productionQualified"] as? Bool,false)
    let described=try NativeLTXPreparation.describe(request:request)
    XCTAssertTrue((described["readinessErrors"] as? [String] ?? []).isEmpty)
    for asset in project.assets { XCTAssertTrue((described["sourcePaths"] as? [String] ?? []).contains(asset.path)) }
    let prepared=try NativeLTXPreparation.prepare(request:request,destination:root.appendingPathComponent("prepared"))
    let exported=try JSONSerialization.jsonObject(with:Data(contentsOf:URL(fileURLWithPath:prepared["recipePath"] as! String))) as! [String:Any]
    XCTAssertEqual((exported["scene"] as! [String:Any])["version"] as? Int,2)
    XCTAssertEqual(project,snapshot)
    XCTAssertEqual(try encoder.encode(project),before)
  }
  func testThirtyTwoSceneImagesWithoutExpandingOrdinaryEightAndOldFirstOnlyVersion() throws {
    let(root,original,runtime)=try fixture();var project=original
    try append(0,role:.first,clip:0,root:root,project:&project)
    project.clips[0].generationSelection?.task="i2v"
    let legacy=try NativeLTXPreparation.compose(request:body(project,runtime))
    XCTAssertEqual(((legacy["recipe"] as! [String:Any])["scene"] as! [String:Any])["version"] as? Int,1)
    project=original
    for frame in 0..<32 { try append(Double(frame),clip:0,root:root,project:&project) }
    XCTAssertEqual(images(try NativeLTXPreparation.compose(request:body(project,runtime))).count,32)
    try append(32,clip:0,root:root,project:&project)
    XCTAssertThrowsError(try NativeLTXPreparation.compose(request:body(project,runtime)))
  }
  func testAdmissionRejectsForeignAssetsConflictingFramesCausalTailAndUnsupportedModes() throws {
    let(root,original,runtime)=try fixture();var project=original
    try append(7,clip:0,root:root,project:&project)
    let valid=project
    try append(7.5,clip:0,root:root,project:&project) // rounds to eight, remains distinct.
    XCTAssertNoThrow(try NativeLTXPreparation.compose(request:body(project,runtime)))
    try append(7,clip:0,root:root,project:&project)
    XCTAssertThrowsError(try NativeLTXPreparation.compose(request:body(project,runtime)))
    project=valid;project.clips[0].attachments[0].assetID=UUID()
    XCTAssertThrowsError(try NativeLTXPreparation.compose(request:body(project,runtime)))
    project=valid;project.assets[0].kind = .video
    XCTAssertThrowsError(try NativeLTXPreparation.compose(request:body(project,runtime)))
    project=valid;project.assets[0].path=root.appendingPathComponent("missing.png").path
    XCTAssertThrowsError(try NativeLTXPreparation.compose(request:body(project,runtime)))
    project=valid;try append(72,clip:1,root:root,project:&project)
    XCTAssertThrowsError(try NativeLTXPreparation.compose(request:body(project,runtime)))
    project=valid;project.clips[0].generationSelection?.ltx25Keyframes = .init(generatedCount:1,experimentalEnabled:true)
    XCTAssertThrowsError(try NativeLTXPreparation.compose(request:body(project,runtime)))
    project=valid;project.clips[0].generationSelection?.ltx25AutomaticDuration = .init(experimentalEnabled:true)
    XCTAssertThrowsError(try NativeLTXPreparation.compose(request:body(project,runtime)))
    project=valid;project.clips[0].generationSelection?.ltx25Guidance = .init(mode:.guided,experimentalEnabled:true)
    XCTAssertThrowsError(try NativeLTXPreparation.compose(request:body(project,runtime)))
    XCTAssertThrowsError(try NativeLTXSceneImageInputs.make(members:project.clips,assets:project.assets,
      segmentStarts:[0,Int.max],segmentFrames:[48,72],fps:24))
  }
  func testIdenticalDuplicateCollapsesAndLibraryImagesRemainLinked() throws {
    let(root,original,runtime)=try fixture();var project=original
    try append(7,clip:0,root:root,project:&project)
    var duplicate=project.clips[0].attachments[0];duplicate.id=UUID();project.clips[0].attachments.append(duplicate)
    let asset=project.assets.removeLast()
    var request=try body(project,runtime);request["globalAssets"]=try JSONSerialization.jsonObject(with:JSONEncoder().encode([asset]))
    let output=try NativeLTXPreparation.compose(request:request)
    XCTAssertEqual(images(output).count,1)
    XCTAssertEqual(images(output)[0]["path"] as? String,asset.path)
    project.clips[0].attachments[1].strength=0.5
    request=try body(project,runtime);request["globalAssets"]=try JSONSerialization.jsonObject(with:JSONEncoder().encode([asset]))
    XCTAssertThrowsError(try NativeLTXPreparation.compose(request:request))
  }
}
