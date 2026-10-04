import XCTest
@testable import StudioCore

final class NativeLTXAutomaticDurationTests: XCTestCase {
  static func headFixture(at directory:URL, wrongShape:Bool=false) throws -> URL {
    var shapes = NativeLTXAutomaticDuration.headShapes
    if wrongShape { shapes["video_input_proj.weight"] = [256,4095] }
    let config = try JSONSerialization.data(withJSONObject:["transformer":["cross_attention_dim":4096,"audio_cross_attention_dim":2048],"duration_head":[:]] as [String:Any])
    var header: [String:Any] = ["__metadata__":["model_version":"2.5.0","config":String(data:config,encoding:.utf8)!]]
    var count = 0
    for name in shapes.keys.sorted() {
      let shape=shapes[name]!, size=shape.reduce(1,*)*2
      header["duration_head."+name] = ["dtype":"BF16","shape":shape,"data_offsets":[count,count+size]]
      count += size
    }
    let json=try JSONSerialization.data(withJSONObject:header,options:.sortedKeys)
    var length=UInt64(json.count).littleEndian
    var bytes=withUnsafeBytes(of:&length) { Data($0) };bytes.append(json);bytes.append(Data(repeating:0,count:count))
    let file=directory.appendingPathComponent(UUID().uuidString+".safetensors")
    try bytes.write(to:file);return file
  }
  func testCompatibleHeadAndCompletePayloadAdmitHeaderOnlyButMutationAndBadShapeReject() throws {
    let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
    defer { try? FileManager.default.removeItem(at:root) }
    let file=try Self.headFixture(at:root)
    XCTAssertEqual(try NativeLTXAutomaticDuration.validateHead(at:file).count,64)
    XCTAssertThrowsError(try NativeLTXAutomaticDuration.validateHead(at:Self.headFixture(at:root,wrongShape:true)))
    let handle=try FileHandle(forWritingTo:file);try handle.seekToEnd();try handle.write(contentsOf:Data([0]));try handle.close()
    XCTAssertThrowsError(try NativeLTXAutomaticDuration.validateHead(at:file))
  }
  func testConfigurablePythonRangeKeepsDefaultsAndRejectsTightInvalidGrid() throws {
    XCTAssertEqual(try NativeLTXAutomaticDuration.maximumFrames(minimumSeconds:1,maximumSeconds:20,fps:24),473)
    XCTAssertEqual(try NativeLTXAutomaticDuration.maximumFrames(minimumSeconds:0.25,maximumSeconds:30,fps:24),713)
    XCTAssertThrowsError(try NativeLTXAutomaticDuration.maximumFrames(minimumSeconds:2.4,maximumSeconds:2.5,fps:24))
    for (minimum,maximum) in [(0.24,20.0),(1.0,30.01),(3.0,2.0),(Double.nan,20.0)] {
      XCTAssertThrowsError(try NativeLTXAutomaticDuration.maximumFrames(minimumSeconds:minimum,maximumSeconds:maximum,fps:24))
    }
  }
}
