import XCTest
import Foundation
import CryptoKit
import Darwin
@testable import LTX25MLX

final class MLXMovieDigestTests: XCTestCase {
  private func physicalFootprint() throws -> UInt64 {
    var info=task_vm_info_data_t(),count=mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size/MemoryLayout<integer_t>.size)
    let status=withUnsafeMutablePointer(to:&info) { pointer in
      pointer.withMemoryRebound(to:integer_t.self,capacity:Int(count)) { task_info(mach_task_self_,task_flavor_t(TASK_VM_INFO),$0,&count) }
    }
    XCTAssertEqual(status,KERN_SUCCESS);return info.phys_footprint
  }
  func test256MiBHashReleasesReadBuffersBeforeCallerAutoreleasePoolDrains() throws {
    let file=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at:file) }
    try Data().write(to:file)
    let writer=try FileHandle(forWritingTo:file);try writer.truncate(atOffset:256*1024*1024);try writer.close()
    var oracle=SHA256();let zeros=Data(repeating:0,count:1024*1024)
    for _ in 0..<256 { oracle.update(data:zeros) }
    let expected=oracle.finalize().map { String(format:"%02x",$0) }.joined()
    try autoreleasepool {
      let before=try physicalFootprint(),actual=try MLXMovieFiles.digest(file),after=try physicalFootprint()
      XCTAssertEqual(actual,expected)
      print("chunk-digest physical before=\(before) after=\(after) payload=268435456")
      XCTAssertLessThan(after,before+64*1024*1024,"Movie hashing must release each read before returning to a caller's long-lived autorelease pool.")
    }
  }
  func testCancelledHashRejectsBeforeReadingAllMedia() async throws {
    let file=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at:file) }
    try Data([1,2,3]).write(to:file)
    let task=Task { () throws -> Bool in
      withUnsafeCurrentTask { $0?.cancel() }
      do { _ = try MLXMovieFiles.digest(file);return false } catch is CancellationError { return true }
    }
    let cancelled=try await task.value;XCTAssertTrue(cancelled)
  }
  func testOnlyNonemptyRegularMediaCanBeHashed() throws {
    let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at:root,withIntermediateDirectories:false)
    defer { try? FileManager.default.removeItem(at:root) }
    let file=root.appendingPathComponent("movie"),link=root.appendingPathComponent("link")
    try Data().write(to:file);XCTAssertThrowsError(try MLXMovieFiles.digest(file))
    try Data([1,2,3]).write(to:file)
    try FileManager.default.createSymbolicLink(at:link,withDestinationURL:file)
    XCTAssertThrowsError(try MLXMovieFiles.digest(link))
  }
}
