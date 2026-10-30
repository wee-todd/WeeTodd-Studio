import Foundation
import XCTest
@testable import InferenceContracts

final class NativeMLXMetalLibraryTests:XCTestCase {
  func testPackagedSharedLibraryWinsOverShadowSibling() throws {
    let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at:root) }
    let executable=root.appendingPathComponent("Test.app/Contents/MacOS/Worker")
    let shared=root.appendingPathComponent("Test.app/Contents/Resources/LTXNative/mlx.metallib")
    try write(executable);try write(shared);try write(executable.deletingLastPathComponent().appendingPathComponent("mlx.metallib"))
    XCTAssertEqual(try NativeMLXMetalLibrary.resolve(executable:executable),shared.resolvingSymlinksInPath())
    try FileManager.default.removeItem(at:shared)
    XCTAssertThrowsError(try NativeMLXMetalLibrary.resolve(executable:executable))
  }
  func testExecutableSymlinksResolveBeforeChoosingLibrary() throws {
    let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at:root) }
    let executable=root.appendingPathComponent("Test.app/Contents/MacOS/Worker")
    let shared=root.appendingPathComponent("Test.app/Contents/Resources/LTXNative/mlx.metallib")
    try write(executable);try write(shared)
    let link=root.appendingPathComponent("linked-worker")
    try FileManager.default.createSymbolicLink(at:link,withDestinationURL:executable)
    XCTAssertEqual(try NativeMLXMetalLibrary.resolve(executable:link),shared.resolvingSymlinksInPath())
  }
  func testStandaloneRequiresAdjacentRegularLibrary() throws {
    let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at:root) }
    let executable=root.appendingPathComponent("Worker"),library=root.appendingPathComponent("mlx.metallib")
    try write(executable)
    XCTAssertThrowsError(try NativeMLXMetalLibrary.resolve(executable:executable))
    try FileManager.default.createDirectory(at:library,withIntermediateDirectories:false)
    XCTAssertThrowsError(try NativeMLXMetalLibrary.resolve(executable:executable))
    try FileManager.default.removeItem(at:library);try write(library)
    XCTAssertEqual(try NativeMLXMetalLibrary.resolve(executable:executable),library.resolvingSymlinksInPath())
    XCTAssertThrowsError(try NativeMLXMetalLibrary.resolve(executable:URL(string:"https://example.com/Worker")!))
  }
  private func write(_ url:URL)throws {
    try FileManager.default.createDirectory(at:url.deletingLastPathComponent(),withIntermediateDirectories:true)
    try Data("fixture".utf8).write(to:url)
  }
}
