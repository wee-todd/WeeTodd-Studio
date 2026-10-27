import CryptoKit
import Foundation
import MLX
import TensorIO
import XCTest
@testable import H3MLX

final class H3OmniCheckpointTests: XCTestCase {
  private let metadata = ["modified_from": "FastVideo/FastVideo-FastH3-Omni-8-Step-V1",
    "modified_by": "mlx-serve tests/convert_minimax_h3_weights.py",
    "quantization": "affine 8-bit g64", "quantized_tensors": "310", "dense_tensors": "275"]

  private func fixture() -> [String: H3TensorInfo] {
    var result: [String: H3TensorInfo] = [:]
    func dense(_ name: String, _ shape: [UInt64], _ dtype: String = "BF16") {
      result[name] = .init(dtype: dtype, shape: shape)
    }
    func affine(_ name: String, _ rows: UInt64, _ columns: UInt64) {
      dense(name + ".weight", [rows, columns / 4], "U32")
      dense(name + ".scales", [rows, columns / 64])
      dense(name + ".biases", [rows, columns / 64])
    }
    for (name, rows, columns) in [("video_patch_proj", 5376, 96),
      ("audio_patch_proj", 5376, 32), ("time_embedder.proj_in", 5376, 256),
      ("time_embedder.proj_out", 2688, 5376), ("final_layer.video_out", 3072, 5376),
      ("final_layer.audio_out", 1024, 5376)] {
      dense(name + ".weight", [UInt64(rows), UInt64(columns)])
      dense(name + ".bias", [UInt64(rows)])
    }
    affine("condition_proj", 5376, 5120); dense("condition_proj.bias", [5376])
    affine("final_layer.adaln_proj.linear", 10752, 2688)
    dense("final_layer.adaln_proj.linear.bias", [10752])
    dense("final_layer.norm.weight", [5376]); dense("token_refiner.final_norm.weight", [5376])
    dense("rope.inv_freq", [16], "F32")
    for body in [false, true] {
      for index in 0..<(body ? 50 : 2) {
        let root = (body ? "blocks." : "token_refiner.blocks.") + "\(index)."
        for name in ["norm1.weight", "norm2.weight"] { dense(root + name, [5376]) }
        for name in ["attn.q_norm.weight", "attn.k_norm.weight"] { dense(root + name, [128]) }
        affine(root + "attn.qkv_proj", 21504, 5376)
        affine(root + "attn.out_proj", 5376, 7168)
        affine(root + "mlp.fc1", 28672, 5376)
        affine(root + "mlp.fc2", 5376, 14336)
        if body {
          affine(root + "attn.to_gate_compress", 7168, 5376)
          affine(root + "adaln_proj.linear", 96768, 2688)
          dense(root + "adaln_proj.linear.bias", [96768])
        }
      }
    }
    return result
  }

  func testIntrinsicHeaderAdmitsOnlyCompleteOmniAndRejectsOrdinaryAndPreview() throws {
    let tensors = fixture()
    XCTAssertEqual(tensors.count, 1205)
    let layout = try H3OmniCheckpoint.validate(tensors: tensors, metadata: metadata)
    XCTAssertEqual(layout.omniVariant, .affineQ8V1)
    XCTAssertNil(layout.fastVariant)
    XCTAssertEqual(layout.quantizedProjections, 310)
    XCTAssertThrowsError(try H3CheckpointLayout(tensors: tensors))
    for variant in [H3FastVariant.denseV1, .vsaV1] {
      XCTAssertThrowsError(try H3CheckpointLayout(tensors: tensors, pagedAffine: true,
        computedRotary: true, fastVariant: variant))
    }
  }

  func testMissingGatesFactorsWrongHeadsAndFalseSourceIdentityFailHeaderAdmission() throws {
    let tensors = fixture()
    for name in ["blocks.0.attn.to_gate_compress.weight", "blocks.49.attn.to_gate_compress.biases",
      "condition_proj.scales", "final_layer.adaln_proj.linear.biases", "token_refiner.blocks.1.attn.qkv_proj.scales"] {
      var broken = tensors; broken.removeValue(forKey: name)
      XCTAssertThrowsError(try H3OmniCheckpoint.validate(tensors: broken, metadata: metadata), name)
    }
    for (name, shape) in [("final_layer.video_out.weight", [UInt64(96), 5376]),
      ("final_layer.audio_out.bias", [UInt64(32)]), ("blocks.2.attn.to_gate_compress.weight", [UInt64.max, 1344])] {
      var broken = tensors; broken[name] = .init(dtype: "BF16", shape: shape)
      XCTAssertThrowsError(try H3OmniCheckpoint.validate(tensors: broken, metadata: metadata), name)
    }
    var altered = metadata; altered["modified_from"] = "unrelated/model"
    XCTAssertThrowsError(try H3OmniCheckpoint.validate(tensors: tensors, metadata: altered))
    altered = metadata; altered["quantization"] = "affine 4-bit g64"
    XCTAssertThrowsError(try H3OmniCheckpoint.validate(tensors: tensors, metadata: altered))
    var extra = tensors; extra["untrained.weight"] = .init(dtype: "BF16", shape: [1])
    XCTAssertThrowsError(try H3OmniCheckpoint.validate(tensors: extra, metadata: metadata))
  }

  // Six source rows: Q0,Q1,K0,K1,V0,V1. Distinct affine offsets detect
  // factor-only reordering, absent reordering and a second reorder.
  func testBodyAndRefinerQ8ProjectionMatchIndependentlyReorderedDenseOracle() throws {
    for stem in ["blocks.0.attn.qkv_proj", "token_refiner.blocks.0.attn.qkv_proj"] {
      let url = try tinyFile(stem: stem)
      defer { try? FileManager.default.removeItem(at: url) }
      let scope = H3OmniParameterScope(file: try SafeTensorFile(url: url), url: url)
      defer { scope.close() }
      let input = MLXArray([Float](repeating: 1, count: 64), [1, 1, 64])
      let result = try scope.project(stem, input: input, qkvHeads: 2, headWidth: 1)
      XCTAssertEqual(result.asArray(Float.self), [64, 208, 352, 136, 280, 424])
      XCTAssertEqual(scope.storageBytes, 408)
      scope.close()
      XCTAssertEqual(scope.storageBytes, 0)
      XCTAssertThrowsError(try scope.project(stem, input: input, qkvHeads: 2, headWidth: 1))
    }
  }

  func testBoundedReadsAndIdentityRejectBeforeLoadingAndScopeClosesOnFailure() throws {
    let url = try tinyFile(stem: "gate")
    defer { try? FileManager.default.removeItem(at: url) }
    let scope = H3OmniParameterScope(file: try SafeTensorFile(url: url), url: url)
    defer { scope.close() }
    XCTAssertThrowsError(try scope.read("gate.weight", maximumBytes: 383))
    XCTAssertEqual(scope.storageBytes, 0)
    XCTAssertThrowsError(try scope.read("gate.scales", rows: 0..<7))
    let handle = try FileHandle(forWritingTo: url)
    try handle.seekToEnd(); try handle.write(contentsOf: Data([0])); try handle.close()
    XCTAssertThrowsError(try scope.read("gate.weight"))
    XCTAssertEqual(scope.storageBytes, 0)
  }

  func testInstalledHeaderAndDigestAdmissionAndTinyGateProjection() throws {
    guard let path = ProcessInfo.processInfo.environment["WEETODD_H3_OMNI_CHECKPOINT"] else {
      throw XCTSkip("Set WEETODD_H3_OMNI_CHECKPOINT for pinned source admission and tiny loaded projection.")
    }
    let url = URL(fileURLWithPath: path)
    XCTAssertEqual(try H3OmniCheckpoint.inspect(url: url).omniVariant, .affineQ8V1)
    XCTAssertThrowsError(try H3CheckpointLayout(url: url))
    let source = try H3OmniCheckpoint.admit(url: url)
    XCTAssertThrowsError(try H3CheckpointSource.layout(url.appendingPathExtension("different"), omni: source))
    XCTAssertTrue(source === (try H3OmniCheckpoint.admit(url: url)))
    var retired: H3OmniParameterScope?
    try source.withParameters { scope in
      retired = scope
      let projection = try scope.projection("blocks.0.attn.to_gate_compress", rows: 0..<2)
      XCTAssertEqual(projection.storageBytes, 11424)
      let input = MLXArray.ones([1, 5376], dtype: .float32)
      let actual = try projection.project(input).asArray(Float.self)
      let factors = projection.parametersToMaterialize
      let packed = factors[0].asArray(UInt32.self)
      let scales = factors[1].asType(.float32).asArray(Float.self)
      let biases = factors[2].asType(.float32).asArray(Float.self)
      for row in 0..<2 {
        var oracle: Float = 0
        for col in 0..<5376 {
          let byte = Float((packed[row * 1344 + col / 4] >> UInt32((col % 4) * 8)) & 255)
          oracle += byte * scales[row * 84 + col / 64] + biases[row * 84 + col / 64]
        }
        XCTAssertEqual(actual[row], oracle, accuracy: 0.001)
      }
      XCTAssertEqual(scope.storageBytes, 11424)
    }
    XCTAssertEqual(retired?.storageBytes, 0)
    XCTAssertTrue(retired?.isClosed == true)
    enum Stop: Error { case requested }
    XCTAssertThrowsError(try source.withParameters { scope in
      retired = scope
      _ = try scope.read("final_layer.video_out.bias", rows: 0..<384)
      throw Stop.requested
    })
    XCTAssertEqual(retired?.storageBytes, 0)
    XCTAssertTrue(retired?.isClosed == true)
  }

  func testInstalledExplicitTrunkReadersPreservePrecisionQKVAndRelease() throws {
    guard let path = ProcessInfo.processInfo.environment["WEETODD_H3_OMNI_CHECKPOINT"] else {
      throw XCTSkip("Set WEETODD_H3_OMNI_CHECKPOINT for explicit common trunk reader qualification.")
    }
    let url = URL(fileURLWithPath: path)
    let source = try H3OmniCheckpoint.admit(url: url)
    let time = try H3TimeEmbedding.evaluate(checkpointURL: url,
      timesteps: MLXArray([Float(0.001), 1]), omni: source)
    XCTAssertEqual(time.dtype, .float32)
    XCTAssertEqual(time.shape, [2, 2688])
    for kind in [H3InputProjection.Kind.video, .audio, .condition] {
      let input = MLXArray.ones([1, 1, kind.columns], dtype: .float32)
      let result = try H3InputProjection.evaluate(checkpointURL: url, kind: kind, input: input, omni: source)
      XCTAssertEqual(result.dtype, kind == .condition ? .bfloat16 : .float32)
      XCTAssertTrue(result.asType(.float32).asArray(Float.self).allSatisfy(\.isFinite))
    }
    let modulation = try H3AdaLNProjection.evaluate(checkpointURL: url, blockIndex: 0,
      timeEmbeddings: time, omni: source)
    XCTAssertEqual(modulation.shape, [2, 96768])
    XCTAssertEqual(modulation.dtype, .bfloat16)
    var qkvObserved = false
    _ = try H3TokenRefiner.evaluateBlock(checkpointURL: url, index: 0,
      input: MLXArray.ones([1, 1, 5376], dtype: .bfloat16), omni: source) { name, value in
      if name == "qkv" {
        qkvObserved = true
        XCTAssertEqual(value.shape, [1, 1, 56, 3, 128])
        XCTAssertTrue(value.asType(.float32).asArray(Float.self).allSatisfy(\.isFinite))
      }
    }
    XCTAssertTrue(qkvObserved)
    let block = try H3PreparedBlock(checkpointURL: url, index: 0,
      projectionMode: .weightDecoded, omni: source)
    defer { block.close() }
    XCTAssertFalse(block.usesGroupedFastQ8Loading)
    let activation = MLXArray.ones([1, 1, 5376], dtype: .bfloat16)
    enum UnexpectedExecution: Error { case ordinaryAttention }
    var ordinaryObserved = false
    XCTAssertThrowsError(try H3TransformerBlock.evaluate(checkpointURL: url, index: 0,
      input: activation, modulation: modulation, modulationIndices: MLXArray([Int32(0)]),
      positions: MLXArray.zeros([1, 3], dtype: .float32), preparedWeights: block,
      observe: { _, _ in ordinaryObserved = true; throw UnexpectedExecution.ordinaryAttention })) { error in
      XCTAssertEqual(error as? H3CheckpointError,
        .invalid("FastH3 Omni requires its dedicated attention execution contract."))
    }
    XCTAssertFalse(ordinaryObserved)
    let qkv = try block.project("attn.qkv_proj", activation: activation, rows: 21504, columns: 5376, qkv: true)
    let gate = try block.project("attn.to_gate_compress", activation: activation, rows: 7168, columns: 5376)
    eval(qkv, gate)
    try source.withParameters { scope in
      let expected = try scope.project("blocks.0.attn.qkv_proj", input: activation)
      XCTAssertEqual(qkv.view(dtype: .uint16).asArray(UInt16.self), expected.view(dtype: .uint16).asArray(UInt16.self))
      XCTAssertEqual(try scope.read("final_layer.video_out.weight").shape, [3072, 5376])
      XCTAssertEqual(try scope.read("final_layer.audio_out.weight").shape, [1024, 5376])
      XCTAssertEqual(try scope.read("final_layer.video_out.weight").dtype, .bfloat16)
      XCTAssertEqual(try scope.read("final_layer.audio_out.bias").dtype, .bfloat16)
      XCTAssertEqual(try scope.projection("final_layer.adaln_proj.linear").rows, 10752)
    }
    XCTAssertGreaterThan(block.storageBytes, 0)
    block.close()
    XCTAssertEqual(block.storageBytes, 0)
    XCTAssertThrowsError(try block.project("attn.to_gate_compress", activation: activation, rows: 7168, columns: 5376))
  }

  func testDigestBindsOpenedBytesAndRejectsEditedAndReplacedSources() throws {
    let url = try tinyFile(stem: "gate")
    defer { try? FileManager.default.removeItem(at: url) }
    let file = try SafeTensorFile(url: url)
    let expected = SHA256.hash(data: try Data(contentsOf: url)).map { String(format: "%02x", $0) }.joined()
    XCTAssertEqual(try H3OmniCheckpoint.checkedDigest(file: file, url: url), expected)
    let replacement = try tinyFile(stem: "other")
    defer { try? FileManager.default.removeItem(at: replacement) }
    try FileManager.default.removeItem(at: url)
    try FileManager.default.moveItem(at: replacement, to: url)
    XCTAssertThrowsError(try H3OmniCheckpoint.checkedDigest(file: file, url: url))
    let edited = try SafeTensorFile(url: url)
    let handle = try FileHandle(forWritingTo: url)
    try handle.seekToEnd(); try handle.write(contentsOf: Data([0])); try handle.close()
    XCTAssertThrowsError(try H3OmniCheckpoint.checkedDigest(file: edited, url: url))
  }

  func testDigestHostStorageRemainsBoundedWithinCallerAutoreleasePool() throws {
    // A sparse synthetic payload isolates hashing from MLX/model residency.
    let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".safetensors")
    defer { try? FileManager.default.removeItem(at: url) }
    let payloadBytes = 256 * 1024 * 1024
    let header = try JSONSerialization.data(withJSONObject: ["payload": [
      "dtype": "U8", "shape": [payloadBytes], "data_offsets": [0, payloadBytes]]], options: [.sortedKeys])
    var length = UInt64(header.count).littleEndian
    var prefix = withUnsafeBytes(of: &length) { Data($0) }; prefix.append(header)
    try prefix.write(to: url)
    let writer = try FileHandle(forWritingTo: url)
    try writer.truncate(atOffset: UInt64(prefix.count + payloadBytes)); try writer.close()
    var oracle = SHA256(); oracle.update(data: prefix)
    let zeros = Data(repeating: 0, count: 4 * 1024 * 1024)
    for _ in 0..<64 { oracle.update(data: zeros) }
    let expected = oracle.finalize().map { String(format: "%02x", $0) }.joined()
    let file = try SafeTensorFile(url: url)
    try autoreleasepool {
      let before = try H3RenderResourceUsage.capture(peakMLXBytes: 0).currentPhysicalBytes
      XCTAssertEqual(try H3OmniCheckpoint.checkedDigest(file: file, url: url), expected)
      let after = try H3RenderResourceUsage.capture(peakMLXBytes: 0).currentPhysicalBytes
      print("digest-only physical before=\(before) after=\(after) payload=\(payloadBytes)")
      XCTAssertLessThan(after, before + 64 * 1024 * 1024,
        "Checksum chunks must be released inside the digest loop, not at caller pool drain.")
    }
  }

  func testCancelledDigestRejectsBeforeReadingPayload() async throws {
    let url = try tinyFile(stem: "gate")
    defer { try? FileManager.default.removeItem(at: url) }
    let task = Task { () throws -> Bool in
      let file = try SafeTensorFile(url: url)
      withUnsafeCurrentTask { $0?.cancel() }
      do {
        _ = try H3OmniCheckpoint.checkedDigest(file: file, url: url)
        return false
      } catch is CancellationError { return true }
    }
    let cancelled = try await task.value
    XCTAssertTrue(cancelled)
  }

  func testExecutionAdmissionRejectsUnpinnedFileAndSymlink() throws {
    let url = try tinyFile(stem: "gate")
    defer { try? FileManager.default.removeItem(at: url) }
    XCTAssertThrowsError(try H3OmniCheckpoint.admit(url: url))
    let link = url.appendingPathExtension("link")
    defer { try? FileManager.default.removeItem(at: link) }
    try FileManager.default.createSymbolicLink(at: link, withDestinationURL: url)
    XCTAssertThrowsError(try H3OmniCheckpoint.inspect(url: link))
    XCTAssertThrowsError(try H3OmniCheckpoint.admit(url: link))
  }

  func testInstalledCancellationClosesActiveParameterScope() async throws {
    guard let path = ProcessInfo.processInfo.environment["WEETODD_H3_OMNI_CHECKPOINT"] else {
      throw XCTSkip("Set WEETODD_H3_OMNI_CHECKPOINT for installed cancellation release.")
    }
    let task = Task { () throws -> Bool in
      let source = try H3OmniCheckpoint.admit(url: URL(fileURLWithPath: path))
      var retired: H3OmniParameterScope?
      do {
        try source.withParameters { scope in
          retired = scope
          _ = try scope.read("final_layer.audio_out.bias", rows: 0..<128)
          withUnsafeCurrentTask { $0?.cancel() }
        }
        return false
      } catch is CancellationError {
        return retired?.isClosed == true && retired?.storageBytes == 0
      }
    }
    let released = try await task.value
    XCTAssertTrue(released)
  }

  private func tinyFile(stem: String) throws -> URL {
    var header: [String: Any] = [:], payload = Data()
    func append(_ name: String, dtype: String, shape: [Int], bytes: Data) {
      header[name] = ["dtype": dtype, "shape": shape,
        "data_offsets": [payload.count, payload.count + bytes.count]]
      payload.append(bytes)
    }
    let packed = (1...6).flatMap { [UInt32](repeating: UInt32($0) * 0x01010101, count: 16) }
    let scales = [UInt16](repeating: 0x3f80, count: 6)
    let biases: [UInt16] = [0, 0x3e00, 0x3e80, 0x3ec0, 0x3f00, 0x3f20]
    append(stem + ".weight", dtype: "U32", shape: [6, 16], bytes: packed.withUnsafeBytes { Data($0) })
    append(stem + ".scales", dtype: "BF16", shape: [6, 1], bytes: scales.withUnsafeBytes { Data($0) })
    append(stem + ".biases", dtype: "BF16", shape: [6, 1], bytes: biases.withUnsafeBytes { Data($0) })
    let json = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
    var length = UInt64(json.count).littleEndian
    var data = withUnsafeBytes(of: &length) { Data($0) }; data.append(json); data.append(payload)
    let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".safetensors")
    try data.write(to: url)
    return url
  }
}
