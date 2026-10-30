import Darwin
import Foundation
import XCTest
@testable import StudioCore

final class ModelDownloadExistingTokenTests:XCTestCase {
  func testKeychainThenEnvironmentThenConfiguredCachePrecedence() throws {
    let home=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at:home,withIntermediateDirectories:false)
    defer{try?FileManager.default.removeItem(at:home)}
    let token=home.appendingPathComponent("token")
    try Data("  hf_cache_fixture\n".utf8).write(to:token)
    let environment=["HF_TOKEN":"hf_environment_fixture","HF_TOKEN_PATH":token.path]
    XCTAssertEqual(try ModelDownloadToken.resolveExisting(environment:environment,home:home,keychainReader:{"hf_keychain_fixture"}),"hf_keychain_fixture")
    XCTAssertEqual(try ModelDownloadToken.resolveExisting(environment:environment,home:home,keychainReader:{nil}),"hf_environment_fixture")
    XCTAssertEqual(try ModelDownloadToken.resolveExisting(environment:["HF_TOKEN_PATH":token.path],home:home,keychainReader:{nil}),"hf_cache_fixture")
    XCTAssertNil(try ModelDownloadToken.resolveExisting(environment:["HF_TOKEN_PATH":home.appendingPathComponent("missing").path],home:home,keychainReader:{nil}))
  }
  func testStandardHomeAndXDGLocationsAreReadWithoutPersisting() throws {
    let home=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer{try?FileManager.default.removeItem(at:home)}
    for (environment,relative) in [([String:String](),".cache/huggingface/token"),(["HF_HOME":home.appendingPathComponent("hf").path],"hf/token"),(["XDG_CACHE_HOME":home.appendingPathComponent("xdg").path],"xdg/huggingface/token")] {
      let file=home.appendingPathComponent(relative)
      try FileManager.default.createDirectory(at:file.deletingLastPathComponent(),withIntermediateDirectories:true)
      try Data("hf_existing_fixture".utf8).write(to:file)
      let before=try Data(contentsOf:file)
      XCTAssertEqual(try ModelDownloadToken.resolveExisting(environment:environment,home:home,keychainReader:{nil}),"hf_existing_fixture")
      XCTAssertEqual(try Data(contentsOf:file),before)
    }
  }
  func testOversizedNonRegularAndInvalidCredentialsRejectWithoutBlocking() throws {
    let home=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at:home,withIntermediateDirectories:false)
    defer{try?FileManager.default.removeItem(at:home)}
    let file=home.appendingPathComponent("token")
    for data in [Data(repeating:65,count:16385),Data("two tokens".utf8),Data([0xff])] {
      try data.write(to:file)
      XCTAssertThrowsError(try ModelDownloadToken.resolveExisting(environment:["HF_TOKEN_PATH":file.path],home:home,keychainReader:{nil}))
    }
    try FileManager.default.removeItem(at:file)
    XCTAssertEqual(mkfifo(file.path,0o600),0)
    XCTAssertThrowsError(try ModelDownloadToken.resolveExisting(environment:["HF_TOKEN_PATH":file.path],home:home,keychainReader:{nil}))
    for key in ["HF_TOKEN_PATH","HF_HOME","XDG_CACHE_HOME"] {
      XCTAssertThrowsError(try ModelDownloadToken.resolveExisting(environment:[key:"relative/token"],home:home,keychainReader:{nil}))
    }
  }
}
