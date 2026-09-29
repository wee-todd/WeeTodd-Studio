import XCTest
import Foundation
import InferenceMedia
@testable import LTX25MLX

final class MLXReferenceImageTests:XCTestCase {
  private func reference(_ role:String,_ image:URL) throws -> MLXImageReference {
    try JSONDecoder().decode(MLXImageReference.self,from:JSONSerialization.data(withJSONObject:
      ["role":role,"path":image.path,"strength":1,"crf":0]))
  }
  func testNativePreparationCenterCropsAndBoundsImageDecode() throws {
    let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at:root,withIntermediateDirectories:false)
    defer { try? FileManager.default.removeItem(at:root) }
    let image=root.appendingPathComponent("source.png")
    var rgb:[Float]=[]
    for _ in 0..<32 { for x in 0..<96 { rgb += x<32 ? [1,-1,-1] : x<64 ? [-1,1,-1] : [-1,-1,1] } }
    try MediaOutput.writePNG(rgb,width:96,height:32,to:image)
    let output=try MLXReferenceImage.prepare(image,width:32,height:32,crf:0,ffmpeg:URL(fileURLWithPath:"/unused"),temporaryParent:root)
    XCTAssertEqual(output.count,32*32*3)
    let center=(16*32+16)*3
    XCTAssertLessThan(output[center],-0.9);XCTAssertGreaterThan(output[center+1],0.9)
    XCTAssertThrowsError(try MLXReferenceImage.prepare(image,width:31,height:32,crf:0,ffmpeg:URL(fileURLWithPath:"/unused"),temporaryParent:root))
    XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath:root.path),["source.png"])
  }
  func testPreparedStagesRetainPixelsAndDeduplicateBeforeWeightedWork() throws {
    let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at:root,withIntermediateDirectories:false)
    defer { try? FileManager.default.removeItem(at:root) }
    let image=root.appendingPathComponent("source.png")
    try MediaOutput.writePNG([Float](repeating:0.25,count:64*32*3),width:64,height:32,to:image)
    let refs=try ["first","last"].map { try reference($0,image) }
    let prepared=try MLXReferenceImage.prepareStages(refs,sizes:[(32,32),(64,32)],ffmpeg:URL(fileURLWithPath:"/unused"),directory:root)
    XCTAssertEqual(prepared.count,2)
    XCTAssertEqual(prepared[0][0].file,prepared[0][1].file)
    XCTAssertEqual(prepared[1][0].file,prepared[1][1].file)
    XCTAssertNotEqual(prepared[0][0].file,prepared[1][0].file)
    XCTAssertEqual(try prepared[0][0].pixels().count,32*32*3)
    XCTAssertEqual(try prepared[1][0].pixels().first!,0.25,accuracy:1.0/127)
    try Data().write(to:prepared[0][0].file)
    XCTAssertThrowsError(try prepared[0][0].pixels())
  }
  func testExtremeAspectRatioFailsInUnweightedStagePreparation() throws {
    let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at:root,withIntermediateDirectories:false)
    defer { try? FileManager.default.removeItem(at:root) }
    let image=root.appendingPathComponent("source.png")
    try MediaOutput.writePNG([Float](repeating:0,count:8192*3),width:8192,height:1,to:image)
    let ref=try reference("first",image)
    XCTAssertThrowsError(try MLXReferenceImage.prepareStages([ref],sizes:[(224,128),(448,256)],
      ffmpeg:URL(fileURLWithPath:"/unused"),directory:root)) { error in
      XCTAssertTrue(String(describing:error).contains("aspect ratio"))
    }
    XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath:root.path),["source.png"])
  }
}
