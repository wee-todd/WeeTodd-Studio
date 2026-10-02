import CryptoKit
import XCTest
@testable import StudioCore

private final class ModelResponseProtocol: URLProtocol, @unchecked Sendable {
  static var response: (URLRequest) -> (Int, [String: String], Data) = { _ in (500, [:], Data()) }
  override class func canInit(with request: URLRequest) -> Bool { true }
  override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
  override func startLoading() {
    let (status, headers, data) = Self.response(request)
    client?.urlProtocol(self, didReceive: HTTPURLResponse(url: request.url!, statusCode: status,
      httpVersion: "HTTP/1.1", headerFields: headers)!, cacheStoragePolicy: .notAllowed)
    client?.urlProtocol(self, didLoad: data)
    client?.urlProtocolDidFinishLoading(self)
  }
  override func stopLoading() {}
}

final class NativeModelDownloadsTests: XCTestCase {
  let bytes = Data("fixture-model".utf8)
  func fixture(_ root: URL) throws -> (URL, NativeModelDownloadFile) {
    let digest = SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined()
    let file: [String: Any] = ["repo": "fixture/model", "revision": String(repeating: "a", count: 40),
      "filename": "model.safetensors", "target": "pages/model.safetensors", "size": bytes.count, "sha256": digest]
    let descriptor: [String: Any] = ["id": "test-model", "name": "Test", "description": "Test",
      "downloadBytes": bytes.count, "requiredDiskBytes": bytes.count, "sourceURL": "https://huggingface.co/fixture/model",
      "licenseURL": "https://huggingface.co/fixture/model/LICENSE", "outputKind": "directory", "engines": ["h3"]]
    let catalog = root.appendingPathComponent("catalog.json")
    try JSONSerialization.data(withJSONObject: [["descriptor": descriptor, "kind": "h3-qwen", "files": [file]]])
      .write(to: catalog)
    return (catalog, try JSONDecoder().decode(NativeModelDownloadFile.self, from: JSONSerialization.data(withJSONObject: file)))
  }
  func directory() throws -> URL {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    return root
  }
  func testExistingWeightsAreVerifiedAndLinkedWithoutDownloadingOrCopying() async throws {
    let root = try directory(); defer { try? FileManager.default.removeItem(at: root) }
    let (catalog, _) = try fixture(root)
    let source = root.appendingPathComponent("existing.safetensors"); try bytes.write(to: source)
    let final = try await NativeModelDownloads.prepare(id: "test-model", catalog: catalog,
      destination: root.appendingPathComponent("library"), existingRoots: [source.path], token: nil,
      progress: { _, _ in }, transfer: { _, _, _, _ in XCTFail("Must reuse verified weights"); throw CancellationError() })
    let installed = final.appendingPathComponent("pages/model.safetensors")
    XCTAssertEqual(try Data(contentsOf: installed), bytes)
    let first = try FileManager.default.attributesOfItem(atPath: source.path)
    let second = try FileManager.default.attributesOfItem(atPath: installed.path)
    XCTAssertEqual(first[.systemFileNumber] as? NSNumber, second[.systemFileNumber] as? NSNumber)
    let provenance = try String(contentsOf: final.appendingPathComponent("setup_provenance.json"), encoding: .utf8)
    XCTAssertTrue(provenance.contains("swift")); XCTAssertFalse(provenance.contains("Bearer"))
    do {
      _ = try await NativeModelDownloads.prepare(id: "test-model", catalog: catalog,
        destination: root.appendingPathComponent("library"), existingRoots: [], progress: { _, _ in })
      XCTFail("Never replace an installed package")
    } catch { XCTAssertTrue(error.localizedDescription.contains("already exists")) }
  }
  func testCancellationRetainsPartialAndRetryPublishesOnlyVerifiedBytes() async throws {
    let root = try directory(); defer { try? FileManager.default.removeItem(at: root) }
    let (catalog, _) = try fixture(root), library = root.appendingPathComponent("library")
    do {
      _ = try await NativeModelDownloads.prepare(id: "test-model", catalog: catalog, destination: library,
        existingRoots: [], token: nil, progress: { _, _ in }, transfer: { _, partial, _, _ in
          try Data("fixture-".utf8).write(to: partial); throw CancellationError()
        })
      XCTFail("Cancellation must not publish")
    } catch is CancellationError {}
    XCTAssertFalse(FileManager.default.fileExists(atPath: library.appendingPathComponent("test-model").path))
    XCTAssertFalse(FileManager.default.fileExists(atPath: library.appendingPathComponent(".test-model.lock").path))
    let bytes = self.bytes
    let final = try await NativeModelDownloads.prepare(id: "test-model", catalog: catalog, destination: library,
      existingRoots: [], token: nil, progress: { _, _ in }, transfer: { _, partial, _, _ in
        XCTAssertEqual(try Data(contentsOf: partial), Data("fixture-".utf8)); try bytes.write(to: partial)
      })
    XCTAssertEqual(try Data(contentsOf: final.appendingPathComponent("pages/model.safetensors")), bytes)
  }
  func testCorruptPayloadAndCatalogTraversalAreRejected() async throws {
    let root = try directory(); defer { try? FileManager.default.removeItem(at: root) }
    let (catalog, _) = try fixture(root), library = root.appendingPathComponent("library")
    let wrong = Data(repeating: 0, count: bytes.count)
    do {
      _ = try await NativeModelDownloads.prepare(id: "test-model", catalog: catalog, destination: library,
        existingRoots: [], token: nil, progress: { _, _ in }, transfer: { _, partial, _, _ in try wrong.write(to: partial) })
      XCTFail("Reject corrupt content")
    } catch { XCTAssertTrue(error.localizedDescription.contains("checksum")) }
    XCTAssertFalse(FileManager.default.fileExists(atPath: library.appendingPathComponent("test-model").path))
    var records = try JSONSerialization.jsonObject(with: Data(contentsOf: catalog)) as! [[String: Any]]
    var files = records[0]["files"] as! [[String: Any]]; files[0]["target"] = "../escape"; records[0]["files"] = files
    try JSONSerialization.data(withJSONObject: records).write(to: catalog)
    XCTAssertThrowsError(try NativeModelDownloads.catalog(at: catalog))
  }
  func testHTTPRangeResumeAndIgnoredRangeBothProduceExactPayload() async throws {
    let root = try directory(); defer { try? FileManager.default.removeItem(at: root) }
    let (_, file) = try fixture(root), partial = root.appendingPathComponent("partial")
    let configuration = URLSessionConfiguration.ephemeral; configuration.protocolClasses = [ModelResponseProtocol.self]
    for status in [206, 200] {
      try bytes.prefix(4).write(to: partial)
      let bytes = self.bytes
      ModelResponseProtocol.response = { request in
        XCTAssertEqual(request.value(forHTTPHeaderField: "Range"), "bytes=4-")
        XCTAssertEqual(request.value(forHTTPHeaderField: "Authorization"), "Bearer test-token")
        let body = status == 206 ? bytes.dropFirst(4) : bytes
        return (status, ["Content-Length": "\(body.count)", "Content-Range": "bytes 4-\(bytes.count-1)/\(bytes.count)"], Data(body))
      }
      try await NativeModelHTTPTransfer(file: file, partial: partial, token: "test-token",
        configuration: configuration, progress: { _, _ in }).run()
      XCTAssertTrue(try NativeModelDownloads.verified(partial, file: file))
    }
  }
  func testHTTPRejectsInvalidRangeBeforeReplacingPartialBytes() async throws {
    let root = try directory(); defer { try? FileManager.default.removeItem(at: root) }
    let (_, file) = try fixture(root), partial = root.appendingPathComponent("partial")
    try bytes.prefix(4).write(to: partial)
    let configuration = URLSessionConfiguration.ephemeral; configuration.protocolClasses = [ModelResponseProtocol.self]
    ModelResponseProtocol.response = { _ in (206, ["Content-Length": "9", "Content-Range": "bytes 0-8/13"], Data(repeating: 0, count: 9)) }
    do {
      try await NativeModelHTTPTransfer(file: file, partial: partial, token: nil,
        configuration: configuration, progress: { _, _ in }).run()
      XCTFail("Reject mismatched resume range")
    } catch { XCTAssertTrue(error.localizedDescription.contains("resume range")) }
    XCTAssertEqual(try Data(contentsOf: partial), bytes.prefix(4))
  }
  func testCancelledHTTPTaskPreservesExistingPartial() async throws {
    let root = try directory(); defer { try? FileManager.default.removeItem(at: root) }
    let (_, file) = try fixture(root), partial = root.appendingPathComponent("partial")
    try bytes.prefix(4).write(to: partial)
    let configuration = URLSessionConfiguration.ephemeral; configuration.protocolClasses = [ModelResponseProtocol.self]
    let task = Task {
      withUnsafeCurrentTask { $0?.cancel() }
      try await NativeModelHTTPTransfer(file: file, partial: partial, token: nil,
        configuration: configuration, progress: { _, _ in }).run()
    }
    do { try await task.value; XCTFail("Cancellation must stop the transfer") }
    catch is CancellationError {}
    XCTAssertEqual(try Data(contentsOf: partial), bytes.prefix(4))
  }
  func testCanonicalCatalogAndPinnedNetworkFileWhenRequested() async throws {
    let source = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
      .deletingLastPathComponent().deletingLastPathComponent().appendingPathComponent("src/wee_todd_mlx/model_download_catalog.json")
    let catalog = try NativeModelDownloads.catalog(at: source)
    XCTAssertEqual(catalog.count, 4)
    XCTAssertFalse(catalog.contains { $0.kind.hasPrefix("h3-transformer") || $0.kind.hasPrefix("h3-support") })
    guard ProcessInfo.processInfo.environment["WEETODD_NATIVE_DOWNLOAD_NETWORK"] == "1" else {
      throw XCTSkip("Opt-in real pinned-file HTTP transfer")
    }
    let root = try directory(); defer { try? FileManager.default.removeItem(at: root) }
    let file = try XCTUnwrap(catalog.flatMap(\.files).first { $0.filename == "LICENSE" && $0.size < 100_000 })
    let partial = root.appendingPathComponent("license.partial")
    try await NativeModelHTTPTransfer(file: file, partial: partial, token: nil, progress: { _, _ in }).run()
    XCTAssertTrue(try NativeModelDownloads.verified(partial, file: file))
  }
}
