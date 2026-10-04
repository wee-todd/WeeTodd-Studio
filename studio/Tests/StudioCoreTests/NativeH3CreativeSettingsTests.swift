import CryptoKit
import Foundation
import XCTest
@testable import StudioCore

final class NativeH3CreativeSettingsTests:XCTestCase {
  private func sha(_ bytes:Data) -> String { SHA256.hash(data:bytes).map { String(format:"%02x",$0) }.joined() }
  private func artifact() throws -> (URL,H3JointLatentArtifact,[String:Any]) {
    let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at:root,withIntermediateDirectories:false)
    addTeardownBlock { try? FileManager.default.removeItem(at:root) }
    let frames=73,video=22*96,audio=2*Int((Double(frames)/24*40).rounded(.toNearestOrEven))*32
    let payload=Data(count:(video+audio)*4);try payload.write(to:root.appendingPathComponent("joint-latents.f32"))
    let fields:[String:Any]=["format":"weetodd-h3-swift-joint-latents-v1","task":"t2va","width":32,"height":32,
      "generatedFrames":frames,"componentIdentity":String(repeating:"b",count:64),"videoFloats":video,"audioFloats":audio,
      "payloadBytes":payload.count,"payloadSHA256":sha(payload)]
    let bytes=try JSONSerialization.data(withJSONObject:fields,options:[.sortedKeys]),url=root.appendingPathComponent("joint-manifest.json")
    try bytes.write(to:url)
    let metadata:[String:Any]=["jointLatentManifest":url.path,"jointLatentManifestSHA256":sha(bytes),"jointLatentPayloadSHA256":sha(payload)]
    return(root,try XCTUnwrap(H3JointLatentArtifact.adopt(metadata:metadata)),metadata)
  }
  func testBoundedFullArtifactAdoptionAndPayloadMutationReject() throws {
    let(root,source,metadata)=try artifact();try source.verify()
    XCTAssertEqual(source.task,"t2va");XCTAssertEqual(source.generatedFrames,73)
    XCTAssertEqual(try JSONDecoder().decode(H3JointLatentArtifact.self,from:JSONEncoder().encode(source)),source)
    XCTAssertNil(try H3JointLatentArtifact.adopt(metadata:[:]))
    XCTAssertThrowsError(try H3JointLatentArtifact.adopt(metadata:["jointLatentManifest":source.manifest]))
    var payload=try Data(contentsOf:root.appendingPathComponent(source.payloadFilename));payload[0]=1
    try payload.write(to:root.appendingPathComponent(source.payloadFilename))
    XCTAssertThrowsError(try H3JointLatentArtifact.adopt(metadata:metadata))
  }
  func testContinuationTailCannotBeAdoptedAsFullArtifact() throws {
    let(_,source,metadata)=try artifact()
    var fields=try JSONSerialization.jsonObject(with:Data(contentsOf:URL(fileURLWithPath:source.manifest))) as! [String:Any]
    fields["format"]="weetodd-h3-swift-continuation-v2"
    let bytes=try JSONSerialization.data(withJSONObject:fields);try bytes.write(to:URL(fileURLWithPath:source.manifest))
    var changed=metadata;changed["jointLatentManifestSHA256"]=sha(bytes)
    XCTAssertThrowsError(try H3JointLatentArtifact.adopt(metadata:changed))
  }
  func testInitializedAndSpatialRefinementDoNotChangeOrdinaryNilRecipe() throws {
    let(_,source,_)=try artifact()
    let ordinary:[String:Any]=["components":["task":"t2va"],"config":["width":32,"height":32,"duration_seconds":Double(73)/24,"steps":20],"prompt":"same"]
    XCTAssertEqual(try JSONSerialization.data(withJSONObject:NativeH3JointPreparation.apply(nil,to:ordinary,continuityMode:"independent"),options:[.sortedKeys]),try JSONSerialization.data(withJSONObject:ordinary,options:[.sortedKeys]))
    let initialized=H3JointSettings(saveFullLatents:true,refinement:.init(mode:.initialized,source:source))
    let applied=try NativeH3JointPreparation.apply(initialized,to:ordinary,continuityMode:"independent")
    XCTAssertEqual((applied["refinement"] as? [String:Any])?["source_manifest_sha256"] as? String,source.manifestSHA256)
    XCTAssertThrowsError(try NativeH3JointPreparation.apply(initialized,to:ordinary,continuityMode:"motion"))
    var spatial=ordinary;spatial["config"]=["width":64,"height":64,"duration_seconds":Double(73)/24,"steps":20]
    let selected=H3JointSettings(refinement:.init(mode:.spatial,source:source))
    XCTAssertEqual((try NativeH3JointPreparation.apply(selected,to:spatial,continuityMode:"independent")["refinement"] as? [String:Any])?["resize_method"] as? String,"bilinear")
    spatial["config"]=["width":64,"height":32,"duration_seconds":Double(73)/24,"steps":20]
    XCTAssertThrowsError(try NativeH3JointPreparation.apply(selected,to:spatial,continuityMode:"independent"))
  }
  func testSignedDeferredLoRAAndTurboScheduleContracts() throws {
    try H3LoRASettings(profile:.standard,qkvLayout:.nativeInterleaved,startAfterEvaluations:3).validate(strength:-10,evaluations:4,samplingMethod:"res_multistep")
    XCTAssertThrowsError(try H3LoRASettings(startAfterEvaluations:4).validate(strength:1,evaluations:4,samplingMethod:"euler"))
    try H3LoRASettings(profile:.turbo).validate(strength:1,evaluations:4,samplingMethod:"euler")
    XCTAssertThrowsError(try H3LoRASettings(profile:.turbo).validate(strength:1,evaluations:5,samplingMethod:"euler"))
    XCTAssertThrowsError(try H3LoRASettings(profile:.turbo,startAfterEvaluations:1).validate(strength:1,evaluations:4,samplingMethod:"euler"))
  }
  func testTimingAndGlobalReferenceStrengthBounds() throws {
    XCTAssertEqual(try H3ReferenceFrame.last.wire(visibleFrames:120) as? String,"last")
    XCTAssertThrowsError(try H3ReferenceFrame.index(120).wire(visibleFrames:120))
    XCTAssertThrowsError(try JSONDecoder().decode(H3ReferenceFrame.self,from:Data("true".utf8)))
    try H3ReferenceSettings(visualConditionStrength:0,audioConditionStrength:1).validate(task:"ref2va")
    XCTAssertThrowsError(try H3ReferenceSettings(visualConditionStrength:.nan).validate(task:"ref2va"))
    XCTAssertThrowsError(try H3ReferenceSettings(audioConditionStrength:0.5).validate(task:"t2v"))
  }
  func testCollectionRetainsSiblingPayloadAndRejectsChangedSourceOrExistingDestination() throws {
    let(root,source,_)=try artifact();let destination=root.appendingPathComponent("collected")
    let copied=try ProjectStorage.collectJointLatentArtifact(source,to:destination)
    try copied.verify();XCTAssertEqual(copied.payloadSHA256,source.payloadSHA256)
    XCTAssertEqual(URL(fileURLWithPath:copied.payloadPath).deletingLastPathComponent().path,destination.path)
    XCTAssertThrowsError(try ProjectStorage.collectJointLatentArtifact(source,to:destination))
    try Data([1]).write(to:URL(fileURLWithPath:source.payloadPath))
    let invalid=root.appendingPathComponent("invalid")
    XCTAssertThrowsError(try ProjectStorage.collectJointLatentArtifact(source,to:invalid))
    XCTAssertFalse(FileManager.default.fileExists(atPath:invalid.path))
  }

}
