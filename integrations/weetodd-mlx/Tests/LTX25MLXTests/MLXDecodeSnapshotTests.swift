import XCTest
import Foundation
import TensorIO
import LTX25Engine
@testable import LTX25MLX

final class MLXDecodeSnapshotTests:XCTestCase {
  func testDecoderLayoutsRoundTripAndExistingFilesAreProtected() throws {
    let g=try AVGeometry(width:32,height:64,frames:9,fps:24)
    let latents=AVLatents(video:(0..<g.videoTokens*128).map(Float.init),audio:(0..<g.audioFrames*128).map { -Float($0) })
    let url=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString+".safetensors")
    defer { try? FileManager.default.removeItem(at:url) }
    try MLXDecodeSnapshot.write(latents,geometry:g,to:url)
    let file=try SafeTensorFile(url:url)
    XCTAssertEqual(file.metadata["format"],"weetodd-decoder-latents-v1")
    XCTAssertEqual(file.tensors["video"]?.shape,g.videoShape.map(UInt64.init))
    XCTAssertEqual(file.tensors["audio"]?.shape,[8,UInt64(g.audioFrames),16])
    XCTAssertEqual(try file.readFloat32(named:"video"),try g.unpackVideo(latents.video))
    XCTAssertEqual(try file.readFloat32(named:"audio"),try g.unpackAudio(latents.audio))
    XCTAssertThrowsError(try MLXDecodeSnapshot.write(latents,geometry:g,to:url))
  }
  func testInvalidPayloadCreatesNoSnapshot() throws {
    let g=try AVGeometry(width:32,height:32,frames:1,fps:24)
    let url=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString+".safetensors")
    XCTAssertThrowsError(try MLXDecodeSnapshot.write(AVLatents(video:[],audio:[]),geometry:g,to:url))
    XCTAssertFalse(FileManager.default.fileExists(atPath:url.path))
  }
}
