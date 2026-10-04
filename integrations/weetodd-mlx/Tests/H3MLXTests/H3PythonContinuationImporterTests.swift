import CryptoKit
import Darwin
import Foundation
import XCTest
@testable import H3MLX

final class H3PythonContinuationImporterTests: XCTestCase {
  private struct Fixture: Sendable {
    let folder: URL
    let manifest: URL
    let identity: Data
    let request: H3T2VARequest
    let video: [Float]
    let audio: [Float]
    func load() throws -> H3PythonContinuationImporter.Artifact {
      try H3PythonContinuationImporter.load(manifestURL: manifest,
        expectedSHA256: H3PythonContinuationImporterTests.hash(Data(contentsOf: manifest)),
        expectedIdentityJSON: identity, request: request, task: "t2va", contextFrames: 22)
    }
  }
  private static func hash(_ data: Data) -> String {
    SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
  }
  private static func encoded(_ object: Any) throws -> Data {
    try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
  }
  private static func fixture(dtype: String = "F32") throws -> Fixture {
    let temporary = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: temporary, withIntermediateDirectories: true)
    let resolved = try XCTUnwrap(realpath(temporary.path, nil))
    defer { free(resolved) }
    let folder = URL(fileURLWithPath: String(cString: resolved))
    let checkpoint = folder.appendingPathComponent("model_index.json")
    try Data("{}".utf8).write(to: checkpoint)
    func file(_ name: String, _ content: String) throws -> URL {
      let url = folder.appendingPathComponent(name)
      try Data(content.utf8).write(to: url); return url
    }
    func record(_ url: URL) throws -> [String: Any] {
      let bytes = try Data(contentsOf: url)
      return ["bytes": bytes.count, "sha256": hash(bytes)]
    }
    let transformer = try file("transformer.safetensors", "fixture-transformer")
    let qwen = folder.appendingPathComponent("qwen")
    let processor = folder.appendingPathComponent("processor")
    try FileManager.default.createDirectory(at: qwen, withIntermediateDirectories: true)
    try FileManager.default.createDirectory(at: processor, withIntermediateDirectories: true)
    let qwenFile = try file("qwen/model.bin", "fixture-text-weights")
    let tokenizer = try file("processor/tokenizer.json", "{}")
    let videoVAE = try file("video.safetensors", "fixture-video-weights")
    let audioVAE = try file("audio.safetensors", "fixture-audio-weights")
    let components: [String: Any] = [
      "transformer": ["path": transformer.path, "files": [transformer.lastPathComponent: try record(transformer)]],
      "text_encoder": ["path": qwen.path, "files": ["model.bin": try record(qwenFile)], "referenced_files": [:]],
      "processor": ["path": processor.path, "files": ["tokenizer.json": try record(tokenizer)], "referenced_files": [:]],
      "tokenizer": ["path": processor.path, "files": ["tokenizer.json": try record(tokenizer)], "referenced_files": [:]],
      "video_vae": ["path": videoVAE.path, "files": [videoVAE.lastPathComponent: try record(videoVAE)]],
      "audio_vae": ["path": audioVAE.path, "files": [audioVAE.lastPathComponent: try record(audioVAE)]],
    ]
    let sampling: [String: Any] = ["width": 64, "height": 32, "steps": 5,
      "drop_adaln": true, "memory_mode": "normal", "attention_chunk_size": "automatic",
      "attention_head_chunk_size": "automatic", "ffn_row_chunk_size": "automatic",
      "projection_backend": "auto", "transformer_backend": "mlx",
      "sampling_method": "euler", "inference_optimization": "off", "paging_cache_gb": 0.0]
    let identity: [String: Any] = ["version": 1, "engine": "h3", "task": "t2va",
      "fps": 24, "sample_rate": 32_000, "audio_channels": 2, "width": 64, "height": 32,
      "checkpoint": folder.path, "model_index": try record(checkpoint),
      "text_architecture_config": NSNull(), "components": components, "sampling": sampling,
      "schedule": "native_h3_shifted_flow_v1", "block_residency": "checkpoint_default", "loras": []]
    let video = (0..<1344).map { Float($0) / 4 }
    let audio = (0..<2368).map { -Float($0) / 8 }
    var payload = Data()
    func append(_ values: [Float]) -> [Float] {
      values.map { value in
        if dtype == "F32" {
          var bits = value.bitPattern.littleEndian
          withUnsafeBytes(of: &bits) { payload.append(contentsOf: $0) }
          return value
        }
        var bits = dtype == "BF16" ? UInt16(value.bitPattern >> 16) : Float16(value).bitPattern
        let result = dtype == "BF16" ? Float(bitPattern: UInt32(bits) << 16) : Float(Float16(bitPattern: bits))
        bits = bits.littleEndian
        withUnsafeBytes(of: &bits) { payload.append(contentsOf: $0) }
        return result
      }
    }
    let decodedVideo = append(video), videoBytes = payload.count
    let decodedAudio = append(audio)
    let tensors: [String: Any] = [
      "video": ["dtype": dtype, "shape": [1, 24, 7, 2, 4], "data_offsets": [0, videoBytes]],
      "audio": ["dtype": dtype, "shape": [2, 32, 37], "data_offsets": [videoBytes, payload.count]]]
    let header = try encoded(tensors)
    var length = UInt64(header.count).littleEndian
    var blob = withUnsafeBytes(of: &length) { Data($0) }
    blob.append(header); blob.append(payload)
    try blob.write(to: folder.appendingPathComponent("latents.safetensors"))
    let manifest: [String: Any] = ["format": "weetodd-h3-continuation-v1", "context_frames": 22,
      "identity": identity, "provenance": ["generated_frames": 124,
        "published_frames": 124, "tail_trim_frames": 0, "output_take_id": "fixture"],
      "payload": ["file": "latents.safetensors", "bytes": blob.count,
        "sha256": hash(blob), "tensors": tensors]]
    let manifestURL = folder.appendingPathComponent("manifest.json")
    try encoded(manifest).write(to: manifestURL)
    let request = try H3T2VARequest(prompt: "Continue walking", width: 64, height: 32,
      durationSeconds: 3, seed: 9, requestedSteps: 5,
      transformer: transformer, qwenPages: qwen, tokenizer: tokenizer,
      videoVAE: videoVAE, audioVAE: audioVAE)
    return Fixture(folder: folder, manifest: manifestURL, identity: try encoded(identity),
      request: request, video: decodedVideo, audio: decodedAudio)
  }

  func testAllReleasedFloatDtypesImportExactChannelAndPatchOrderWithoutNormalization() throws {
    for dtype in ["F32", "F16", "BF16"] {
      let fixture = try Self.fixture(dtype: dtype)
      defer { try? FileManager.default.removeItem(at: fixture.folder) }
      let imported = try fixture.load()
      var expectedVideo: [Float] = []
      for frame in 0..<7 { for x in stride(from: 0, to: 4, by: 2) {
        for channel in 0..<24 { for y in 0..<2 { for dx in 0..<2 {
          expectedVideo.append(fixture.video[((channel * 7 + frame) * 2 + y) * 4 + x + dx])
        } } }
      } }
      var expectedAudio: [Float] = []
      for channel in 0..<2 { for time in 0..<37 { for latent in 0..<32 {
        expectedAudio.append(fixture.audio[(channel * 32 + latent) * 37 + time])
      } } }
      // Independently generated NumPy reshape/transpose float32-byte oracles.
      let golden: [String: (String, String)] = [
        "F32": ("475d32a0c5ff5017d6c25643447d699140434475f3745309695e27d17fb80cdf", "16f681fe208cea773c8f87b948a72c4eb4740bb901124c2afd473157fc34831c"),
        "F16": ("475d32a0c5ff5017d6c25643447d699140434475f3745309695e27d17fb80cdf", "c17a27685f0ff9c3e150454e3193365c537525008ada2cc2472d9118eb26b352"),
        "BF16": ("5ff45838ce086ccf43e3b337f2ba929f5c3718b605019c41db9f13c284713bbb", "6e73e58b7f2420a97051b1075ff46aee69cca1622478db36bbb9c8cb0438880a")]
      XCTAssertEqual(imported.rows.video.withUnsafeBytes { Self.hash(Data($0)) }, golden[dtype]?.0)
      XCTAssertEqual(imported.rows.audio.withUnsafeBytes { Self.hash(Data($0)) }, golden[dtype]?.1)
      XCTAssertEqual(imported.rows.video.map(\.bitPattern), expectedVideo.map(\.bitPattern))
      XCTAssertEqual(imported.rows.audio.map(\.bitPattern), expectedAudio.map(\.bitPattern))
      XCTAssertEqual(imported.rows.video.count, 1344)
      XCTAssertEqual(imported.rows.audio.count, 2368)
      XCTAssertEqual(imported.rows.audio[0].bitPattern, Float(-0.0).bitPattern)
    }
  }

  func testChangedComponentAndPayloadContentAreRejectedDespiteValidManifest() throws {
    for changed in ["transformer.safetensors", "latents.safetensors"] {
      let fixture = try Self.fixture()
      defer { try? FileManager.default.removeItem(at: fixture.folder) }
      let file = fixture.folder.appendingPathComponent(changed)
      var bytes = try Data(contentsOf: file); bytes[bytes.count - 1] ^= 1
      try bytes.write(to: file)
      XCTAssertThrowsError(try fixture.load(), "Changed \(changed)")
    }
  }

  func testUntrimmedProvenanceContextAndPayloadPathFailClosed() throws {
    let changes: [(inout [String: Any]) -> Void] = [
      { $0["context_frames"] = true },
      { $0["format"] = "weetodd-h3-swift-continuation-v2" },
      { $0["provenance"] = ["generated_frames": 124, "published_frames": 123, "tail_trim_frames": 1] },
      { value in var payload = value["payload"] as! [String: Any]; payload["file"] = "../latents.safetensors"; value["payload"] = payload },
      { value in
        var payload = value["payload"] as! [String: Any]
        var tensors = payload["tensors"] as! [String: Any]
        tensors["unrecorded"] = tensors["video"]; payload["tensors"] = tensors
        value["payload"] = payload
      },
    ]
    for change in changes {
      let fixture = try Self.fixture()
      defer { try? FileManager.default.removeItem(at: fixture.folder) }
      var manifest = try JSONSerialization.jsonObject(with: Data(contentsOf: fixture.manifest)) as! [String: Any]
      change(&manifest); try Self.encoded(manifest).write(to: fixture.manifest)
      XCTAssertThrowsError(try fixture.load())
    }
  }

  func testSameContentHashCannotOverrideRequestSampling() throws {
    let fixture = try Self.fixture()
    defer { try? FileManager.default.removeItem(at: fixture.folder) }
    var manifest = try JSONSerialization.jsonObject(with: Data(contentsOf: fixture.manifest)) as! [String: Any]
    var identity = manifest["identity"] as! [String: Any]
    var sampling = identity["sampling"] as! [String: Any]
    sampling["sampling_method"] = "res_multistep"; identity["sampling"] = sampling
    manifest["identity"] = identity; try Self.encoded(manifest).write(to: fixture.manifest)
    XCTAssertThrowsError(try H3PythonContinuationImporter.load(manifestURL: fixture.manifest,
      expectedSHA256: Self.hash(Data(contentsOf: fixture.manifest)),
      expectedIdentityJSON: Self.encoded(identity), request: fixture.request,
      task: "t2va", contextFrames: 22))
  }

  func testExplicitNativePublicationIsVerifiedByActualLoaderAndKeepsLegacyUntouched() throws {
    let fixture = try Self.fixture()
    defer { try? FileManager.default.removeItem(at: fixture.folder) }
    let manifestBefore = try Data(contentsOf: fixture.manifest)
    let legacyPayload = fixture.folder.appendingPathComponent("latents.safetensors")
    let payloadBefore = try Data(contentsOf: legacyPayload)
    let output = fixture.folder.appendingPathComponent("native-import")
    let publication = try H3PythonContinuationImporter.importToNative(
      manifestURL: fixture.manifest, expectedSHA256: Self.hash(manifestBefore),
      expectedIdentityJSON: fixture.identity, request: fixture.request, task: "t2va",
      contextFrames: 22, output: output)
    let rows = try XCTUnwrap(H3Continuation.load(manifestURL: publication.manifestURL,
      expectedSHA256: publication.manifestSHA256, contextFrames: 22,
      width: 64, height: 32, identity: H3Continuation.fingerprint(fixture.request)))
    XCTAssertEqual(rows.video.map(\.bitPattern), try fixture.load().rows.video.map(\.bitPattern))
    XCTAssertEqual(rows.audio.map(\.bitPattern), try fixture.load().rows.audio.map(\.bitPattern))
    XCTAssertEqual(try Data(contentsOf: fixture.manifest), manifestBefore)
    XCTAssertEqual(try Data(contentsOf: legacyPayload), payloadBefore)
    let origin = try Data(contentsOf: output.appendingPathComponent("python-v1-origin.json"))
    XCTAssertEqual(Self.hash(origin), publication.originReceiptSHA256)
    let manifest = try JSONSerialization.jsonObject(with: Data(contentsOf: publication.manifestURL)) as! [String: Any]
    XCTAssertEqual(manifest["pythonV1OriginReceiptSHA256"] as? String, publication.originReceiptSHA256)
    XCTAssertEqual(manifest["generatedFrames"] as? Int, 124)
    XCTAssertEqual(manifest["publishedFrames"] as? Int, 124)
    XCTAssertEqual(manifest["overlapFrames"] as? Int, 0)
    XCTAssertThrowsError(try H3PythonContinuationImporter.importToNative(
      manifestURL: fixture.manifest, expectedSHA256: Self.hash(manifestBefore),
      expectedIdentityJSON: fixture.identity, request: fixture.request, task: "t2va",
      contextFrames: 22, output: output))
  }

  func testPublicationPreservesOverlapDifferentFromNewTailContextCount() throws {
    let fixture = try Self.fixture()
    defer { try? FileManager.default.removeItem(at: fixture.folder) }
    var manifest = try JSONSerialization.jsonObject(with: Data(contentsOf: fixture.manifest)) as! [String: Any]
    manifest["provenance"] = ["generated_frames": 124, "published_frames": 119,
      "overlap_frames": 5, "tail_trim_frames": 0]
    try Self.encoded(manifest).write(to: fixture.manifest)
    let result = try H3PythonContinuationImporter.importToNative(manifestURL: fixture.manifest,
      expectedSHA256: Self.hash(Data(contentsOf: fixture.manifest)),
      expectedIdentityJSON: fixture.identity, request: fixture.request, task: "t2va",
      contextFrames: 22, output: fixture.folder.appendingPathComponent("overlap-import"))
    let native = try JSONSerialization.jsonObject(with: Data(contentsOf: result.manifestURL)) as! [String: Any]
    XCTAssertEqual(native["generatedFrames"] as? Int, 124)
    XCTAssertEqual(native["publishedFrames"] as? Int, 119)
    XCTAssertEqual(native["overlapFrames"] as? Int, 5)
    XCTAssertEqual(native["contextFrames"] as? Int, 22)
  }

  func testCancellationPrecedesContentHashingOrRowAllocation() async throws {
    let fixture = try Self.fixture()
    defer { try? FileManager.default.removeItem(at: fixture.folder) }
    let task = Task {
      withUnsafeCurrentTask { $0?.cancel() }
      _ = try fixture.load()
    }
    do { try await task.value; XCTFail("Cancelled import completed") }
    catch is CancellationError {} catch { XCTFail("Unexpected error: \(error)") }
  }
  func testFLPublicationBindsTaskAndVisionInventoryWithoutRemappingComponents() throws {
    let fixture = try Self.fixture()
    defer { try? FileManager.default.removeItem(at: fixture.folder) }
    var identity = try XCTUnwrap(JSONSerialization.jsonObject(with: fixture.identity) as? [String: Any])
    identity["task"] = "fl2va"
    let identityData = try Self.encoded(identity)
    var manifest = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: fixture.manifest)) as? [String: Any])
    manifest["identity"] = identity
    try Self.encoded(manifest).write(to: fixture.manifest)
    let image = H3StillReference(rgb8: Data(repeating: 0, count: 64 * 32 * 3), width: 64, height: 32)
    let frames = try H3FL2VARequest(base: fixture.request, vision: fixture.request.qwenPages,
      images: [image], anchors: [.first])
    let result = try H3PythonContinuationImporter.importToNative(manifestURL: fixture.manifest,
      expectedSHA256: Self.hash(Data(contentsOf: fixture.manifest)),
      expectedIdentityJSON: identityData, request: frames, contextFrames: 22,
      output: fixture.folder.appendingPathComponent("native-fl"))
    let checked = try XCTUnwrap(H3Continuation.load(manifestURL: result.manifestURL,
      expectedSHA256: result.manifestSHA256, contextFrames: 22, width: 64, height: 32,
      identity: H3Continuation.fingerprint(frames), task: "fl2va"))
    let raw = try H3PythonContinuationImporter.load(manifestURL: fixture.manifest,
      expectedSHA256: Self.hash(Data(contentsOf: fixture.manifest)), expectedIdentityJSON: identityData,
      request: fixture.request, task: "fl2va", contextFrames: 22)
    XCTAssertEqual(checked.video.map(\.bitPattern), raw.rows.video.map(\.bitPattern))
    XCTAssertEqual(checked.audio.map(\.bitPattern), raw.rows.audio.map(\.bitPattern))
    let wrongVision = try H3FL2VARequest(base: fixture.request, vision: fixture.request.videoVAE,
      images: [image], anchors: [.first])
    XCTAssertThrowsError(try H3PythonContinuationImporter.importToNative(manifestURL: fixture.manifest,
      expectedSHA256: Self.hash(Data(contentsOf: fixture.manifest)), expectedIdentityJSON: identityData,
      request: wrongVision, contextFrames: 22, output: fixture.folder.appendingPathComponent("wrong-vision")))
  }

  func testHistoricalTwelveKeySamplingDictionaryDefaultsOnlyMissingMLXBackend() throws {
    let fixture = try Self.fixture()
    defer { try? FileManager.default.removeItem(at: fixture.folder) }
    let priorRows = try fixture.load().rows.video.map(\.bitPattern)
    var manifest = try JSONSerialization.jsonObject(with: Data(contentsOf: fixture.manifest)) as! [String: Any]
    var identity = manifest["identity"] as! [String: Any]
    var sampling = identity["sampling"] as! [String: Any]
    sampling.removeValue(forKey: "transformer_backend"); identity["sampling"] = sampling
    func load() throws -> H3PythonContinuationImporter.Artifact {
      manifest["identity"] = identity
      try Self.encoded(manifest).write(to: fixture.manifest)
      return try H3PythonContinuationImporter.load(manifestURL: fixture.manifest,
        expectedSHA256: Self.hash(Data(contentsOf: fixture.manifest)),
        expectedIdentityJSON: Self.encoded(identity), request: fixture.request,
        task: "t2va", contextFrames: 22)
    }
    XCTAssertEqual(try load().rows.video.map(\.bitPattern), priorRows)
    sampling["transformer_backend"] = "nnc_experimental"; identity["sampling"] = sampling
    XCTAssertThrowsError(try load())
    sampling["transformer_backend"] = NSNull(); identity["sampling"] = sampling
    XCTAssertThrowsError(try load())
    sampling.removeValue(forKey: "transformer_backend"); sampling.removeValue(forKey: "steps")
    identity["sampling"] = sampling; XCTAssertThrowsError(try load())
  }

  func testExplicitImportSessionReusesOnlyIdenticalVerifiedFilesAndRejectsMutation() throws {
    let fixture = try Self.fixture()
    defer { try? FileManager.default.removeItem(at: fixture.folder) }
    let session = H3PythonContinuationImporter.ImportSession()
    func load() throws -> H3PythonContinuationImporter.Artifact {
      try H3PythonContinuationImporter.load(manifestURL: fixture.manifest,
        expectedSHA256: Self.hash(Data(contentsOf: fixture.manifest)),
        expectedIdentityJSON: fixture.identity, request: fixture.request,
        task: "t2va", contextFrames: 22, session: session)
    }
    let first = try load(), bytes = session.bytesHashed, files = session.filesHashed
    XCTAssertGreaterThan(bytes, 0); XCTAssertGreaterThan(files, 0)
    let second = try load()
    XCTAssertEqual(first.rows.video.map(\.bitPattern), second.rows.video.map(\.bitPattern))
    XCTAssertEqual(first.rows.audio.map(\.bitPattern), second.rows.audio.map(\.bitPattern))
    XCTAssertEqual(session.bytesHashed, bytes); XCTAssertEqual(session.filesHashed, files)
    XCTAssertGreaterThan(session.hashReuses, 0); XCTAssertGreaterThan(session.bytesReuseValidated, bytes)
    try session.checkUnchanged()
    let transformer = fixture.request.transformer
    let prior = try FileManager.default.attributesOfItem(atPath: transformer.path)
    let count = try Data(contentsOf: transformer).count
    try Data(repeating: 90, count: count).write(to: transformer)
    try FileManager.default.setAttributes([.modificationDate: prior[.modificationDate]!], ofItemAtPath: transformer.path)
    XCTAssertThrowsError(try session.checkUnchanged(), "Same size and restored mtime cannot conceal mutation")
    XCTAssertThrowsError(try load())
  }

  func testImportSessionRejectsConflictingContentDeclarations() throws {
    let fixture = try Self.fixture()
    defer { try? FileManager.default.removeItem(at: fixture.folder) }
    let session = H3PythonContinuationImporter.ImportSession()
    _ = try H3PythonContinuationImporter.load(manifestURL: fixture.manifest,
      expectedSHA256: Self.hash(Data(contentsOf: fixture.manifest)), expectedIdentityJSON: fixture.identity,
      request: fixture.request, task: "t2va", contextFrames: 22, session: session)
    var manifest = try JSONSerialization.jsonObject(with: Data(contentsOf: fixture.manifest)) as! [String: Any]
    var identity = manifest["identity"] as! [String: Any]
    var components = identity["components"] as! [String: Any]
    var component = components["transformer"] as! [String: Any]
    var files = component["files"] as! [String: Any]
    var record = files[fixture.request.transformer.lastPathComponent] as! [String: Any]
    record["sha256"] = String(repeating: "f", count: 64)
    files[fixture.request.transformer.lastPathComponent] = record; component["files"] = files
    components["transformer"] = component; identity["components"] = components; manifest["identity"] = identity
    try Self.encoded(manifest).write(to: fixture.manifest)
    XCTAssertThrowsError(try H3PythonContinuationImporter.load(manifestURL: fixture.manifest,
      expectedSHA256: Self.hash(Data(contentsOf: fixture.manifest)), expectedIdentityJSON: Self.encoded(identity),
      request: fixture.request, task: "t2va", contextFrames: 22, session: session))
  }

  func testHistoricalModelIndexAliasBindsTargetAndRejectsRetargeting() throws {
    let fixture = try Self.fixture()
    defer { try? FileManager.default.removeItem(at: fixture.folder) }
    let alias = fixture.folder.appendingPathComponent("model_index.json")
    let target = fixture.folder.appendingPathComponent("upstream-model-index.json")
    try FileManager.default.moveItem(at: alias, to: target)
    try FileManager.default.createSymbolicLink(at: alias, withDestinationURL: target)
    let session = H3PythonContinuationImporter.ImportSession()
    _ = try H3PythonContinuationImporter.load(manifestURL: fixture.manifest,
      expectedSHA256: Self.hash(Data(contentsOf: fixture.manifest)), expectedIdentityJSON: fixture.identity,
      request: fixture.request, task: "t2va", contextFrames: 22, session: session)
    try session.checkUnchanged()
    let other = fixture.folder.appendingPathComponent("other-model-index.json")
    try Data(contentsOf: target).write(to: other)
    try FileManager.default.removeItem(at: alias)
    try FileManager.default.createSymbolicLink(at: alias, withDestinationURL: other)
    XCTAssertThrowsError(try session.checkUnchanged(), "Even identical bytes cannot hide a retargeted source alias")
    XCTAssertThrowsError(try H3PythonContinuationImporter.load(manifestURL: fixture.manifest,
      expectedSHA256: Self.hash(Data(contentsOf: fixture.manifest)), expectedIdentityJSON: fixture.identity,
      request: fixture.request, task: "t2va", contextFrames: 22, session: session))
  }

  func testModelIndexAliasDoesNotPermitTargetMutationOrWeightAliases() throws {
    let fixture = try Self.fixture()
    defer { try? FileManager.default.removeItem(at: fixture.folder) }
    let alias = fixture.folder.appendingPathComponent("model_index.json")
    let target = fixture.folder.appendingPathComponent("upstream-model-index.json")
    try FileManager.default.moveItem(at: alias, to: target)
    try FileManager.default.createSymbolicLink(at: alias, withDestinationURL: target)
    let session = H3PythonContinuationImporter.ImportSession()
    _ = try H3PythonContinuationImporter.load(manifestURL: fixture.manifest,
      expectedSHA256: Self.hash(Data(contentsOf: fixture.manifest)), expectedIdentityJSON: fixture.identity,
      request: fixture.request, task: "t2va", contextFrames: 22, session: session)
    try Data("[]".utf8).write(to: target)
    XCTAssertThrowsError(try session.checkUnchanged())
    try Data("{}".utf8).write(to: target)
    let weight = fixture.request.transformer, moved = fixture.folder.appendingPathComponent("weight-target")
    try FileManager.default.moveItem(at: weight, to: moved)
    try FileManager.default.createSymbolicLink(at: weight, withDestinationURL: moved)
    XCTAssertThrowsError(try fixture.load(), "Model-index exception never extends to a weight path")
  }

}
