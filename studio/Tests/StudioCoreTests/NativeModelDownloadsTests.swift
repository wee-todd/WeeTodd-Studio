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
  func testGatedAccessFailureExplainsSourcePermissionAndPreservesPartial() async throws {
    let root = try directory();defer { try? FileManager.default.removeItem(at:root) }
    let (_,file) = try fixture(root), partial = root.appendingPathComponent("partial")
    let configuration = URLSessionConfiguration.ephemeral;configuration.protocolClasses = [ModelResponseProtocol.self]
    for status in [401,403] {
      try bytes.prefix(4).write(to:partial)
      ModelResponseProtocol.response = { _ in (status,[:],Data("denied".utf8)) }
      do {
        try await NativeModelHTTPTransfer(file:file,partial:partial,token:"fixture-read-token",
          configuration:configuration,progress:{ _,_ in }).run()
        XCTFail("Gated access must not publish denied response bytes")
      } catch {
        XCTAssertTrue(error.localizedDescription.contains("source access"))
        XCTAssertTrue(error.localizedDescription.contains("read token"))
      }
      XCTAssertEqual(try Data(contentsOf:partial),bytes.prefix(4))
    }
  }
  func testCanonicalCatalogAndPinnedNetworkFileWhenRequested() async throws {
    let source = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
      .deletingLastPathComponent().deletingLastPathComponent().appendingPathComponent("src/wee_todd_mlx/model_download_catalog.json")
    let catalog = try NativeModelDownloads.catalog(at: source)
    XCTAssertEqual(catalog.count, 19)
    let audio=try XCTUnwrap(catalog.first { $0.kind == "h3-audio-vae" })
    XCTAssertEqual(audio.descriptor.components,["audio_vae"])
    XCTAssertTrue(audio.files.contains { $0.filename == "audio_vae.safetensors" && $0.size == 605254808 })
    XCTAssertFalse(catalog.contains { $0.kind.hasPrefix("h3-transformer") || $0.kind.hasPrefix("h3-support") })
    XCTAssertEqual(catalog.filter { $0.kind == "h3-direct-transformer" }.count, 2)
    XCTAssertEqual(catalog.filter { $0.kind == "h3-native-support" }.count, 2)
    XCTAssertEqual(catalog.filter { $0.kind == "h3-native-control" }.count, 2)
    XCTAssertEqual(catalog.filter { $0.kind == "ltx25-adapter" }.count, 8)
    for preset in NativeModelSetup.catalog() {
      for component in preset.components {
        XCTAssertTrue(catalog.contains { $0.descriptor.supports(engine:preset.engine, task:preset.task, component:component.key) },
          "Missing Python-free download for \(preset.id): \(component.key)")
      }
    }
    guard ProcessInfo.processInfo.environment["WEETODD_NATIVE_DOWNLOAD_NETWORK"] == "1" else {
      throw XCTSkip("Opt-in real pinned-file HTTP transfer")
    }
    let root = try directory(); defer { try? FileManager.default.removeItem(at: root) }
    let file = try XCTUnwrap(catalog.flatMap(\.files).first { $0.filename == "LICENSE" && $0.size < 100_000 })
    let partial = root.appendingPathComponent("license.partial")
    try await NativeModelHTTPTransfer(file: file, partial: partial, token: nil, progress: { _, _ in }).run()
    XCTAssertTrue(try NativeModelDownloads.verified(partial, file: file))
  }
  func testRepeatedPackagePayloadIsDownloadedOnceAndLinkedToBothTargets() async throws {
    let root = try directory(); defer { try? FileManager.default.removeItem(at:root) }
    let (catalog, _) = try fixture(root)
    var records = try JSONSerialization.jsonObject(with:Data(contentsOf:catalog)) as! [[String:Any]]
    var files = records[0]["files"] as! [[String:Any]], second = files[0]
    second["target"] = "processor/model.safetensors"; files.append(second); records[0]["files"] = files
    var descriptor = records[0]["descriptor"] as! [String:Any]
    descriptor["downloadBytes"] = bytes.count * 2; records[0]["descriptor"] = descriptor
    try JSONSerialization.data(withJSONObject:records).write(to:catalog)
    let bytes = self.bytes
    let final = try await NativeModelDownloads.prepare(id:"test-model",catalog:catalog,
      destination:root.appendingPathComponent("library"),existingRoots:[],token:nil,progress:{ _,_ in },
      transfer:{ file,partial,_,_ in
        XCTAssertEqual(file.target,"pages/model.safetensors","Duplicate content must reuse the first verified file")
        try bytes.write(to:partial)
      })
    let first = try FileManager.default.attributesOfItem(atPath:final.appendingPathComponent("pages/model.safetensors").path)
    let secondInfo = try FileManager.default.attributesOfItem(atPath:final.appendingPathComponent("processor/model.safetensors").path)
    XCTAssertEqual(first[.systemFileNumber] as? NSNumber,secondInfo[.systemFileNumber] as? NSNumber)
  }
  func testPinnedSwiftTaskManifestsAndProcessorDiscoverWithoutPythonWhenRequested() async throws {
    guard ProcessInfo.processInfo.environment["WEETODD_NATIVE_DOWNLOAD_NETWORK"] == "1" else {
      throw XCTSkip("Opt-in pinned task/processor HTTP transfers")
    }
    let catalogURL = URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent()
      .deletingLastPathComponent().deletingLastPathComponent().appendingPathComponent("src/wee_todd_mlx/model_download_catalog.json")
    let catalog = try NativeModelDownloads.catalog(at:catalogURL)
    let root = try directory(); defer { try? FileManager.default.removeItem(at:root) }
    for (id,preset) in [("h3-swift-fl2va-support","swift-h3-text"),("h3-swift-ref2va-support","swift-h3-reference")] {
      let package = try XCTUnwrap(catalog.first { $0.descriptor.id == id })
      let destination = root.appendingPathComponent(id)
      for target in ["model_index.json","processor/preprocessor_config.json"] {
        let file = try XCTUnwrap(package.files.first { $0.target == target })
        let output = destination.appendingPathComponent(target)
        try FileManager.default.createDirectory(at:output.deletingLastPathComponent(),withIntermediateDirectories:true)
        try await NativeModelHTTPTransfer(file:file,partial:output,token:nil,progress:{ _,_ in }).run()
        XCTAssertTrue(try NativeModelDownloads.verified(output,file:file))
      }
      let scan = try NativeModelSetup.scan(presetID:preset,roots:[destination.path])
      XCTAssertEqual(scan.candidates["checkpoint"],[destination.path])
      XCTAssertEqual(scan.candidates["processor"],[destination.appendingPathComponent("processor").path])
    }
  }
  func testInstalledCompatibleTransformerIsVerifiedAndLinkedWithoutLargeTransferWhenRequested() async throws {
    guard let installed = ProcessInfo.processInfo.environment["WEETODD_NATIVE_TRANSFORMER_SOURCE"] else {
      throw XCTSkip("Opt-in installed direct H3 transformer qualification")
    }
    let catalogURL = URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent()
      .deletingLastPathComponent().deletingLastPathComponent().appendingPathComponent("src/wee_todd_mlx/model_download_catalog.json")
    let root = try directory(); defer { try? FileManager.default.removeItem(at:root) }
    let result = try await NativeModelDownloads.prepare(id:"h3-singularity-full-comfy-int8",catalog:catalogURL,
      destination:root,existingRoots:[installed],token:nil,progress:{ _,_ in },transfer:{ file,partial,token,progress in
        guard file.size < 100_000 else { XCTFail("Must reuse installed transformer"); throw CancellationError() }
        try await NativeModelHTTPTransfer(file:file,partial:partial,token:token,progress:progress).run()
      })
    let linked = result.appendingPathComponent("transformer.safetensors")
    let canonical = linked.resolvingSymlinksInPath()
    let original = try FileManager.default.attributesOfItem(atPath:URL(fileURLWithPath:installed).resolvingSymlinksInPath().path)
    let prepared = try FileManager.default.attributesOfItem(atPath:canonical.path)
    XCTAssertEqual(original[.systemNumber] as? NSNumber,prepared[.systemNumber] as? NSNumber)
    XCTAssertEqual(original[.systemFileNumber] as? NSNumber,prepared[.systemFileNumber] as? NSNumber)
    XCTAssertEqual(try NativeModelSetup.scan(presetID:"swift-h3-reference",roots:[result.path]).candidates["transformer"],[canonical.path])
    XCTAssertEqual(try NativeModelSetup.scan(presetID:"swift-h3-fun-control",roots:[result.path]).candidates["transformer"],[canonical.path])
  }
  func testInstalledFoldedAudioPackageReusesWeightsWhenRequested() async throws {
    guard let installed=ProcessInfo.processInfo.environment["WEETODD_NATIVE_AUDIO_SOURCE"] else {
      throw XCTSkip("Opt-in installed folded H3 audio package qualification")
    }
    let source=URL(fileURLWithPath:installed)
    let output=ProcessInfo.processInfo.environment["WEETODD_NATIVE_AUDIO_PACKAGE_OUTPUT"]
    let root=try output.map { URL(fileURLWithPath:$0) } ?? directory()
    defer { if output == nil { try? FileManager.default.removeItem(at:root) } }
    let catalog=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent()
      .deletingLastPathComponent().deletingLastPathComponent().appendingPathComponent("src/wee_todd_mlx/model_download_catalog.json")
    let package=try await NativeModelDownloads.prepare(id:"h3-folded-audio-vae-preconverted",
      catalog:catalog,destination:root,existingRoots:[source.deletingLastPathComponent().path],progress:{ _,_ in })
    let linked=package.appendingPathComponent("audio_vae.safetensors")
    let original=try FileManager.default.attributesOfItem(atPath:source.path)
    let prepared=try FileManager.default.attributesOfItem(atPath:linked.path)
    XCTAssertEqual(original[.systemFileNumber] as? NSNumber,prepared[.systemFileNumber] as? NSNumber)
    let preset=try XCTUnwrap(NativeModelSetup.catalog().first { $0.engine == "h3" && $0.task == "t2v" })
    let scan=try NativeModelSetup.scan(presetID:preset.id,roots:[package.path])
    XCTAssertEqual(scan.candidates["audio_vae"],[linked.resolvingSymlinksInPath().path])
    for notice in ["LICENSE","NOTICE","MODIFICATIONS.md"] {
      XCTAssertTrue(FileManager.default.isReadableFile(atPath:package.appendingPathComponent(notice).path))
    }
  }
  func testInstalledPublishedFL2VACurveReusesWeightsAndSupportsImageSetupWhenRequested() async throws {
    guard let installed=ProcessInfo.processInfo.environment["WEETODD_NATIVE_FL2VA_SOURCE"] else {
      throw XCTSkip("Opt-in installed published FL2VA curve qualification")
    }
    let requestedOutput=ProcessInfo.processInfo.environment["WEETODD_NATIVE_FL2VA_PACKAGE_OUTPUT"]
    let root=try requestedOutput.map { URL(fileURLWithPath:$0) } ?? directory()
    defer { if requestedOutput == nil { try? FileManager.default.removeItem(at:root) } }
    let catalog=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent()
      .deletingLastPathComponent().deletingLastPathComponent().appendingPathComponent("src/wee_todd_mlx/model_download_catalog.json")
    let result=try await NativeModelDownloads.prepare(id:"h3-fl2va-deepbeep-curve64-bf16",catalog:catalog,
      destination:root,existingRoots:[installed],token:nil,progress:{ _,_ in },transfer:{ file,partial,token,progress in
        guard file.size<100_000 else { XCTFail("Published FL2VA weights must be reused");throw CancellationError() }
        try await NativeModelHTTPTransfer(file:file,partial:partial,token:token,progress:progress).run()
      })
    let linked=result.appendingPathComponent("transformer.safetensors").resolvingSymlinksInPath()
    let original=try FileManager.default.attributesOfItem(atPath:URL(fileURLWithPath:installed).resolvingSymlinksInPath().path)
    let prepared=try FileManager.default.attributesOfItem(atPath:linked.path)
    XCTAssertEqual(original[.systemNumber] as? NSNumber,prepared[.systemNumber] as? NSNumber)
    XCTAssertEqual(original[.systemFileNumber] as? NSNumber,prepared[.systemFileNumber] as? NSNumber)
    XCTAssertEqual(try NativeModelSetup.scan(presetID:"swift-h3-image",roots:[result.path]).candidates["transformer"],[linked.path])
    XCTAssertTrue(try NativeModelSetup.scan(presetID:"swift-h3-reference",roots:[result.path]).candidates["transformer"]!.isEmpty)
    for notice in ["SOURCE_README.md","LICENSE"] { XCTAssertTrue(FileManager.default.isReadableFile(atPath:result.appendingPathComponent(notice).path)) }
  }
  func testInstalledPublishedFL2VASupportReusesTokenizerAndMatchesTaskWhenRequested() async throws {
    guard let support=ProcessInfo.processInfo.environment["WEETODD_NATIVE_FL2VA_SUPPORT_SOURCE"] else {
      throw XCTSkip("Opt-in installed official FL2VA metadata qualification")
    }
    let requestedOutput=ProcessInfo.processInfo.environment["WEETODD_NATIVE_FL2VA_PACKAGE_OUTPUT"]
    let root=try requestedOutput.map { URL(fileURLWithPath:$0) } ?? directory()
    defer { if requestedOutput == nil { try? FileManager.default.removeItem(at:root) } }
    let catalog=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent()
      .deletingLastPathComponent().deletingLastPathComponent().appendingPathComponent("src/wee_todd_mlx/model_download_catalog.json")
      let metadata=try await NativeModelDownloads.prepare(id:"h3-swift-fl2va-support",catalog:catalog,
        destination:root,existingRoots:[support],token:nil,progress:{ _,_ in },transfer:{ file,partial,token,progress in
          guard file.size<100_000 else { XCTFail("Installed tokenizer payloads must be reused");throw CancellationError() }
          try await NativeModelHTTPTransfer(file:file,partial:partial,token:token,progress:progress).run()
        })
      let scan=try NativeModelSetup.scan(presetID:"swift-h3-image",roots:[metadata.path])
      XCTAssertEqual(scan.candidates["checkpoint"],[metadata.path])
      XCTAssertEqual(scan.candidates["processor"],[metadata.appendingPathComponent("processor").path])
      XCTAssertTrue(scan.candidates["tokenizer"]!.contains(metadata.appendingPathComponent("tokenizer").path))
  }
}
