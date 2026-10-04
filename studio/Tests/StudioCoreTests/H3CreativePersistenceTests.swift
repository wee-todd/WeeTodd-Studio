import Foundation
import XCTest
@testable import StudioCore

final class H3CreativePersistenceTests:XCTestCase {
  private var artifact:H3JointLatentArtifact {
    .init(manifest:"latent/joint-manifest.json",manifestSHA256:String(repeating:"a",count:64),payloadSHA256:String(repeating:"b",count:64),task:"ref2va",componentIdentity:String(repeating:"c",count:64),width:768,height:448,generatedFrames:124)
  }
  func testLegacyNilControlsDoNotAddSerializedFields() throws {
    let selection=try JSONDecoder().decode(GenerationSelection.self,from:Data(#"{"task":"t2v","preset":"balanced"}"#.utf8))
    XCTAssertNil(selection.h3Reference);XCTAssertNil(selection.h3Joint)
    XCTAssertEqual(Set((try JSONSerialization.jsonObject(with:JSONEncoder().encode(selection)) as! [String:Any]).keys),["task","preset"])
    let attachment=Attachment(assetID:UUID(),role:.reference)
    let fields=try JSONSerialization.jsonObject(with:JSONEncoder().encode(attachment)) as! [String:Any]
    XCTAssertNil(fields["h3LoRA"]);XCTAssertNil(fields["h3ReferencePlacement"])
    let take=RenderVersion(path:"movie.mp4",seed:42,prompt:"same",recipePath:"recipe.json")
    XCTAssertNil((try JSONSerialization.jsonObject(with:JSONEncoder().encode(take)) as! [String:Any])["jointLatentArtifact"])
  }
  func testFullH3ControlsPersistReopenAndMapAllOwnedPaths() throws {
    let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at:root,withIntermediateDirectories:false)
    defer { try? FileManager.default.removeItem(at:root) }
    var project=StudioProject(),clip=Clip(engine:.h3)
    clip.generationSelection = .init(task:"ref2va")
    clip.generationSelection?.h3Reference = .init(visualConditionStrength:0.4,audioConditionStrength:0.7)
    clip.generationSelection?.h3Joint = .init(saveFullLatents:true,refinement:.init(mode:.initialized,source:artifact))
    clip.savedNativeGenerations=["h3":SavedNativeGenerationSettings(profileID:"auto",selection:clip.generationSelection)]
    let asset=MediaAsset(name:"Reference",kind:.video,path:"source.mov");project.assets=[asset]
    var attachment=Attachment(assetID:asset.id,role:.reference)
    attachment.h3ReferencePlacement = .init(frame:.index(48),soundtrackPath:"sound/voice.wav")
    clip.attachments=[attachment]
    clip.versions=[RenderVersion(path:"take.mp4",seed:42,prompt:"same",recipePath:"recipe.json",jointLatentArtifact:artifact)]
    project.clips=[clip]
    let url=root.appendingPathComponent("project.json");try ProjectStorage.write(project,to:url)
    let loaded=try ProjectStorage.read(url),actual=loaded.clips[0]
    let expected=root.appendingPathComponent("latent/joint-manifest.json").path
    XCTAssertEqual(actual.versions[0].jointLatentArtifact?.manifest,expected)
    XCTAssertEqual(actual.versions[0].jointLatentArtifact?.payloadPath,root.appendingPathComponent("latent/joint-latents.f32").path)
    XCTAssertEqual(actual.generationSelection?.h3Joint?.refinement?.source.manifest,expected)
    XCTAssertEqual(actual.savedNativeGenerations?["h3"]?.selection?.h3Joint?.refinement?.source.manifest,expected)
    XCTAssertEqual(actual.attachments[0].h3ReferencePlacement?.soundtrackPath,root.appendingPathComponent("sound/voice.wav").path)
    XCTAssertEqual(actual.attachments[0].h3ReferencePlacement?.frame,.index(48))
    XCTAssertEqual(actual.versions[0].jointLatentArtifact?.manifestSHA256,artifact.manifestSHA256)
    var reset=try XCTUnwrap(actual.generationSelection);XCTAssertTrue(reset.isModified)
    reset.resetOverrides();XCTAssertNil(reset.h3Reference);XCTAssertNil(reset.h3Joint)
  }
  func testAdapterControlsRoundTripWithoutChangingAssetTrainingMetadata() throws {
    var attachment=Attachment(assetID:UUID(),role:.lora);attachment.strength = -1.5
    attachment.h3LoRA = .init(profile:.standard,qkvLayout:.contiguousQKV,startAfterEvaluations:3)
    XCTAssertEqual(try JSONDecoder().decode(Attachment.self,from:JSONEncoder().encode(attachment)),attachment)
  }
}
