import XCTest
import Foundation
import MLX
import InferenceMedia
@testable import LTX25MLX

final class RawVideoWriterTests:XCTestCase {
  private func directory() throws -> URL {
    let url=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at:url,withIntermediateDirectories:false)
    return url
  }
  func testGPUByteConversionPreservesReferenceTruncationAndClipping() {
    let values:[Float]=[-2,-1,-0.5,0,0.5,1,2,0.12345,-0.65432]
    let actual=MLXVideoDecoder.rgb8(MLXArray(values,[1,3,3]))
    XCTAssertEqual(Array(actual),values.map { UInt8((min(1,max(-1,$0))+1)*127.5) })
  }
  func testEncoderFailureAndIncompleteStreamNeverPublish() throws {
    let dir=try directory();defer { try? FileManager.default.removeItem(at:dir) }
    let output=dir.appendingPathComponent("video.mp4")
    let writer=try RawVideoWriter(ffmpeg:URL(fileURLWithPath:"/usr/bin/false"),output:output,width:4,height:4,frames:2,fps:24)
    defer { writer.cancel() }
    XCTAssertThrowsError(try writer.append(Data(repeating:0,count:48),frame:1))
    XCTAssertThrowsError(try writer.append(Data(repeating:0,count:47),frame:0))
    XCTAssertThrowsError(try writer.finish())
    XCTAssertFalse(FileManager.default.fileExists(atPath:output.path))
  }
  @MainActor func testBlockedEncoderWriteRespondsToCancellation() async throws {
    let dir=try directory();defer { try? FileManager.default.removeItem(at:dir) }
    let executable=dir.appendingPathComponent("blocked-encoder")
    try Data("#!/bin/sh\nexec /bin/sleep 30\n".utf8).write(to:executable)
    try FileManager.default.setAttributes([.posixPermissions:0o700],ofItemAtPath:executable.path)
    let output=dir.appendingPathComponent("video.mp4")
    let task=Task.detached {
      let writer=try RawVideoWriter(ffmpeg:executable,output:output,width:512,height:512,frames:1,fps:24)
      defer { writer.cancel() }
      do { try writer.append(Data(repeating:0,count:512*512*3),frame:0);return false }
      catch is CancellationError { return true }
    }
    try await Task.sleep(for:.milliseconds(100));let start=Date();task.cancel()
    let canceled=try await task.value
    XCTAssertTrue(canceled);XCTAssertLessThan(Date().timeIntervalSince(start),2)
    XCTAssertFalse(FileManager.default.fileExists(atPath:output.path))
  }
  func testInstalledFFmpegPublishesExactlyTheRequestedFrames() throws {
    guard let binary=ProcessInfo.processInfo.environment["WEETODD_FFMPEG"] else { throw XCTSkip("Installed FFmpeg qualification is opt-in.") }
    let dir=try directory();defer { try? FileManager.default.removeItem(at:dir) }
    let output=dir.appendingPathComponent("video.mp4")
    let writer=try RawVideoWriter(ffmpeg:URL(fileURLWithPath:binary),output:output,width:4,height:4,frames:3,fps:24)
    defer { writer.cancel() }
    for frame in 0..<3 { try writer.append(Data(repeating:UInt8(frame*100),count:48),frame:frame) }
    try writer.finish()
    XCTAssertGreaterThan(try Data(contentsOf:output).count,0)
    XCTAssertThrowsError(try writer.append(Data(repeating:0,count:48),frame:3))
    let process=Process(),pipe=Pipe()
    process.executableURL=URL(fileURLWithPath:binary)
    process.arguments=["-v","error","-i",output.path,"-f","rawvideo","-pix_fmt","rgb24","-"]
    process.standardOutput=pipe;try process.run()
    let decoded=pipe.fileHandleForReading.readDataToEndOfFile();process.waitUntilExit()
    XCTAssertEqual(process.terminationStatus,0);XCTAssertEqual(decoded.count,3*48)
  }
}
