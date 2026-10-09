import Foundation
import CryptoKit
import Darwin
import MLX
import XCTest
@testable import H3MLX

final class H3PagedCheckpointTests: XCTestCase {
  private let cores: [(String, [Int])] = [
    ("attn.qkv_proj", [21504, 5376]), ("attn.out_proj", [5376, 7168]),
    ("mlp.fc1", [28672, 5376]), ("mlp.fc2", [5376, 14336]),
  ]
  // Logical checkpoint dimensions with sparse zero payloads: no model arrays,
  // payload mapping, copied weights or GPU execution are needed for admission.
  private func fixture() throws -> URL {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root.appendingPathComponent("pages"), withIntermediateDirectories: true)
    var total: UInt64 = 0
    func page(_ name: String, _ tensors: [(String, String, [Int])]) throws -> [String: Any] {
      var header: [String: Any] = [:], bytes: UInt64 = 0
      for (key, dtype, shape) in tensors {
        let size = UInt64(shape.reduce(1, *)) * UInt64(dtype == "BF16" ? 2 : 4)
        header[key] = ["dtype": dtype, "shape": shape, "data_offsets": [bytes, bytes + size]]
        bytes += size
      }
      let json = try JSONSerialization.data(withJSONObject: header, options: .sortedKeys)
      var length = UInt64(json.count).littleEndian
      var data = withUnsafeBytes(of: &length) { Data($0) }; data.append(json)
      let url = root.appendingPathComponent(name)
      try data.write(to: url)
      let file = try FileHandle(forWritingTo: url)
      try file.truncate(atOffset: UInt64(data.count) + bytes); try file.close()
      total += bytes
      return ["file": name, "sha256": String(repeating: "a", count: 64),
        "tensor_count": tensors.count, "tensor_bytes": bytes]
    }
    var fixed: [(String, String, [Int])] = [
      ("video_patch_proj.weight", "F32", [5376, 96]), ("video_patch_proj.bias", "F32", [5376]),
      ("audio_patch_proj.weight", "F32", [5376, 32]), ("audio_patch_proj.bias", "F32", [5376]),
      ("condition_proj.weight", "BF16", [5376, 5120]), ("condition_proj.bias", "BF16", [5376]),
      ("final_layer.video_out.weight", "F32", [96, 5376]), ("final_layer.video_out.bias", "F32", [96]),
      ("final_layer.audio_out.weight", "F32", [32, 5376]), ("final_layer.audio_out.bias", "F32", [32]),
      ("final_layer.norm.weight", "BF16", [5376]),
      ("final_layer.adaln_proj.linear.weight", "F32", [10752, 64]),
      ("final_layer.adaln_proj.linear.bias", "F32", [10752]),
      ("adaln_t_table", "F32", [1001, 64]), ("token_refiner.final_norm.weight", "BF16", [5376]),
    ]
    for index in 0..<2 {
      let stem = "token_refiner.blocks.\(index)."
      for (name, shape) in cores { fixed.append((stem + name + ".weight", "BF16", shape)) }
      for suffix in ["attn.q_norm", "attn.k_norm", "norm1", "norm2"] {
        fixed.append((stem + suffix + ".weight", "BF16", [suffix.hasPrefix("attn.") ? 128 : 5376]))
      }
    }
    let fixedRecord = try page("pages/fixed.safetensors", fixed)
    var blocks: [[String: Any]] = [], overrides: [String: Int] = [:]
    for index in 0..<50 {
      let stem = "blocks.\(index)."
      var tensors: [(String, String, [Int])] = [
        (stem + "adaln_proj.linear.weight", "F32", [96768, 64]),
        (stem + "adaln_proj.linear.bias", "F32", [96768]),
      ]
      for suffix in ["attn.q_norm", "attn.k_norm", "norm1", "norm2"] {
        tensors.append((stem + suffix + ".weight", "BF16", [suffix.hasPrefix("attn.") ? 128 : 5376]))
      }
      for (name, shape) in cores {
        if index >= 38 || (index >= 21 && name.hasPrefix("mlp.")) {
          overrides[stem + name] = 8
          tensors.append((stem + name + ".weight", "U32", [shape[0], shape[1] / 4]))
          for suffix in ["scales", "biases"] {
            tensors.append((stem + name + "." + suffix, "BF16", [shape[0], shape[1] / 64]))
          }
        } else { tensors.append((stem + name + ".weight", "BF16", shape)) }
      }
      blocks.append(try page(String(format: "pages/block-%03d.safetensors", index), tensors))
    }
    func json(_ name: String, _ value: [String: Any]) throws {
      try JSONSerialization.data(withJSONObject: value, options: .sortedKeys).write(to: root.appendingPathComponent(name))
    }
    try json("paged_manifest.json", ["format": "weetodd-h3-paged-v1", "source": "q8_extended",
      "num_blocks": 50, "source_tensor_bytes": total, "fixed": fixedRecord, "blocks": blocks])
    try json("config.json", ["hidden_size": 5376, "num_layers": 50, "num_attention_heads": 56,
      "attention_head_dim": 128, "ffn_hidden_size": 14336, "time_embed_dim": 64,
      "adaln_curve_grid": 1001, "rope_inv_freq_len": 16, "rope_theta": 10000])
    try json("quant_config.json", ["format": "minimax-h3-mlx-mixed-quant", "format_version": 1,
      "bits": 8, "group_size": 64, "quantize_core": false, "quantize_adaln": false,
      "profile": "q8_extended", "overrides": overrides])
    return root
  }
  private func alter(_ root: URL, _ name: String, _ body: (inout [String: Any]) -> Void) throws {
    let url = root.appendingPathComponent(name)
    var json = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any])
    body(&json)
    try JSONSerialization.data(withJSONObject: json).write(to: url)
  }

  func testHeaderOnlyMixedAffineRoutingAndMutationAreStrict() throws {
    let root = try fixture(); defer { try? FileManager.default.removeItem(at: root) }
    let layout = try H3CheckpointLayout(url: root)
    XCTAssertEqual(layout.curveRank, 64); XCTAssertEqual(layout.blockCount, 50)
    XCTAssertEqual(try H3CheckpointSource.fileURL(root).lastPathComponent, "fixed.safetensors")
    XCTAssertEqual(try H3CheckpointSource.fileURL(root, block: 38).lastPathComponent, "block-038.safetensors")
    XCTAssertThrowsError(try H3CheckpointSource.fileURL(root, block: 50))
    let page = root.appendingPathComponent("pages/block-049.safetensors")
    let file = try FileHandle(forWritingTo: page); try file.truncate(atOffset: 8); try file.close()
    XCTAssertThrowsError(try H3CheckpointSource.fileURL(root, block: 0), "Any page mutation invalidates the stage-bound source")
  }

  func testArchitectureNumbersAndBooleansCannotCoerceEachOther() throws {
    for value: Any in [true, 1.5] {
      let root = try fixture(); defer { try? FileManager.default.removeItem(at: root) }
      try alter(root, "quant_config.json") { $0["format_version"] = value }
      XCTAssertThrowsError(try H3CheckpointLayout(url: root))
    }
    let root = try fixture(); defer { try? FileManager.default.removeItem(at: root) }
    try alter(root, "quant_config.json") { $0["quantize_core"] = 0 }
    XCTAssertThrowsError(try H3CheckpointLayout(url: root))
  }

  func testFiftySlotsProfilePayloadAndSymlinkAdmission() throws {
    for failure in 0..<4 {
      let root = try fixture(); defer { try? FileManager.default.removeItem(at: root) }
      switch failure {
      case 0: try alter(root, "paged_manifest.json") { $0["num_blocks"] = 49 }
      case 1: try alter(root, "quant_config.json") { $0["overrides"] = [:] }
      case 2: try alter(root, "paged_manifest.json") { $0["source_tensor_bytes"] = 1 }
      default:
        let page = root.appendingPathComponent("pages/block-000.safetensors")
        let moved = root.appendingPathComponent("page-outside.safetensors")
        try FileManager.default.moveItem(at: page, to: moved)
        try FileManager.default.createSymbolicLink(at: page, withDestinationURL: moved)
      }
      XCTAssertThrowsError(try H3CheckpointLayout(url: root))
    }
  }

  func testCancellationBeforeHeaders() async throws {
    let root = try fixture(); defer { try? FileManager.default.removeItem(at: root) }
    let job = Task { () async throws -> Bool in
      while !Task.isCancelled { await Task.yield() }
      _ = try H3CheckpointLayout(url: root); return true
    }
    job.cancel()
    do { _ = try await job.value; XCTFail("Cancelled admission succeeded") }
    catch is CancellationError {} catch { XCTFail("Unexpected cancellation: \(error)") }
  }

  func testComputedRotaryMatchesOwnedPythonPagedFloat32FrequencyBits() throws {
    Device.withDefaultDevice(.cpu) {
      // Owned Python RotaryPosEmbed3D evaluates 1/(10000 ** (arange/32))
      // on CPU. The page writer intentionally omits serialized rope.inv_freq;
      // those original full-checkpoint words are a different oracle.
      let bits: [UInt32] = [0x3f800000, 0x3f0ff599, 0x3ea1e89b, 0x3e361887,
        0x3dcccccc, 0x3d6655c2, 0x3d0186e3, 0x3c91ad39, 0x3c23d70a,
        0x3bb8449c, 0x3b4f3e38, 0x3ae91528, 0x3a83126e, 0x3a136a16, 0x39a5cb60, 0x393a7752]
      XCTAssertEqual(H3TransformerBlock.computedInverseFrequency().asArray(Float.self).map(\.bitPattern), bits)
    }
  }

  func testAffineU32ProjectionMatchesIndependentSignedGroupDotProduct() throws {
    try Device.withDefaultDevice(.cpu) {
      var words: [UInt32] = [], expected = [Float](repeating: 0, count: 2)
      let x = (0..<128).map { Float($0 % 7 - 3) }
      let scales: [Float] = [0.5, 0.25, 2, 1]
      let biases: [Float] = [-4, 3, -1, 2]
      for row in 0..<2 {
        for start in stride(from: 0, to: 128, by: 4) {
          var word: UInt32 = 0
          for byte in 0..<4 {
            let column = start + byte, q = (column * 3 + row * 11) % 19
            word |= UInt32(q) << (8 * byte)
            expected[row] += x[column] * (Float(q) * scales[row * 2 + column / 64] + biases[row * 2 + column / 64])
          }
          words.append(word)
        }
      }
      let q8 = try H3QwenQ8Projection(packed: MLXArray(words, [2, 32]),
        scales: MLXArray(scales, [2, 2]), biases: MLXArray(biases, [2, 2]), columns: 128)
      XCTAssertEqual(try q8.project(MLXArray(x, [1, 128])).asArray(Float.self), expected)
      XCTAssertThrowsError(try H3QwenQ8Projection(packed: MLXArray(words, [2, 32]),
        scales: MLXArray(scales, [2, 2]), biases: MLXArray(biases, [2, 2]), columns: 127))
    }
  }

  func testInstalledPagedHeaderOnlyWhenExplicitlyRequested() throws {
    guard let path = ProcessInfo.processInfo.environment["WEETODD_H3_AFFINE_PAGED_CHECKPOINT"] else {
      throw XCTSkip("Set WEETODD_H3_AFFINE_PAGED_CHECKPOINT for installed paged header admission.")
    }
    let layout = try H3CheckpointLayout(url: URL(fileURLWithPath: path))
    XCTAssertEqual(layout.curveRank, 64); XCTAssertEqual(layout.blockCount, 50)
  }
  func testAdmittedPagedRootCannotSwitchToARegularFile() throws {
    let root = try fixture(); defer { try? FileManager.default.removeItem(at: root) }
    _ = try H3CheckpointLayout(url: root)
    try FileManager.default.removeItem(at: root)
    try Data("replacement".utf8).write(to: root)
    XCTAssertThrowsError(try H3CheckpointSource.fileURL(root))
    XCTAssertThrowsError(try H3CheckpointSource.checkUnchanged(root))
    XCTAssertThrowsError(try H3CheckpointLayout(url: root))
  }

  func testUnchangedWrapperAdmitsFreshSourceAndRejectsCachedMutations() throws {
    for name in ["config.json", "quant_config.json", "paged_manifest.json",
      "pages/fixed.safetensors", "pages/block-000.safetensors", "pages/block-049.safetensors"] {
      let root = try fixture(); defer { try? FileManager.default.removeItem(at: root) }
      // First call exercises uncached admission; subsequent calls exercise inspection.
      try H3CheckpointSource.checkUnchanged(root)
      try H3CheckpointSource.checkUnchanged(root)
      let handle = try FileHandle(forWritingTo: root.appendingPathComponent(name))
      try handle.seekToEnd(); try handle.write(contentsOf: Data([0])); try handle.close()
      XCTAssertThrowsError(try H3CheckpointSource.checkUnchanged(root), name)
    }
    for directory in ["", "pages"] {
      let root = try fixture(); defer { try? FileManager.default.removeItem(at: root) }
      try H3CheckpointSource.checkUnchanged(root)
      let folder = directory.isEmpty ? root : root.appendingPathComponent(directory)
      try Data().write(to: folder.appendingPathComponent("unexpected-entry"))
      XCTAssertThrowsError(try H3CheckpointSource.checkUnchanged(root), directory)
    }
  }

  func testUnchangedWrapperRejectsFreshMalformedSourcesAndReplacedPages() throws {
    for cached in [false, true] {
      for failure in 0..<5 {
        let root = try fixture(); defer { try? FileManager.default.removeItem(at: root) }
        if cached { try H3CheckpointSource.checkUnchanged(root) }
        switch failure {
        case 0: try alter(root, "paged_manifest.json") { $0["num_blocks"] = 49 }
        case 1:
          let handle = try FileHandle(forWritingTo: root.appendingPathComponent("pages/block-049.safetensors"))
          try handle.truncate(atOffset: 8); try handle.close()
        case 2:
          try Data([0]).write(to: root.appendingPathComponent("pages/fixed.safetensors"), options: .atomic)
        case 3:
          let page = root.appendingPathComponent("pages/block-049.safetensors")
          try FileManager.default.removeItem(at: page)
          try FileManager.default.createSymbolicLink(at: page,
            withDestinationURL: root.appendingPathComponent("pages/fixed.safetensors"))
        default:
          let pages = root.appendingPathComponent("pages")
          try FileManager.default.removeItem(at: pages)
          try Data([0]).write(to: pages)
        }
        XCTAssertThrowsError(try H3CheckpointSource.checkUnchanged(root), "cached=\(cached), failure=\(failure)")
      }
    }
  }

  func testUnchangedWrapperCancellationForFreshAndCachedSources() async throws {
    for cached in [false, true] {
      let root = try fixture(); defer { try? FileManager.default.removeItem(at: root) }
      if cached { try H3CheckpointSource.checkUnchanged(root) }
      let job = Task {
        while !Task.isCancelled { await Task.yield() }
        try H3CheckpointSource.checkUnchanged(root)
      }
      job.cancel()
      do { try await job.value; XCTFail("Cancelled identity check succeeded") }
      catch is CancellationError {} catch { XCTFail("Unexpected cancellation: \(error)") }
      // Cancellation must not poison the source cache or alter its identity.
      try H3CheckpointSource.checkUnchanged(root)
    }
  }

  func testInstalledIdentityWalkDeduplicationABBAABBA() throws {
    let environment = ProcessInfo.processInfo.environment
    guard let checkpointPath = environment["WEETODD_H3_IDENTITY_TIMING_CHECKPOINT"],
      let outputPath = environment["WEETODD_H3_IDENTITY_TIMING_RECEIPT"] else {
      throw XCTSkip("Opt-in actual FastH3 metadata-only identity timing; no MLX arrays or model payload reads.")
    }
    guard !FileManager.default.fileExists(atPath: outputPath) else {
      throw H3CheckpointError.invalid("Identity timing receipt must be fresh.")
    }
    let root = URL(fileURLWithPath: checkpointPath)
    let source = try H3CheckpointSource.inspect(root) // Header admission is outside timing.
    guard source.layout.fastVariant == .vsaV1 else {
      throw H3CheckpointError.invalid("Identity timing requires the actual admitted FastH3 VSA checkpoint.")
    }
    func sha(_ data: Data) -> String {
      SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }
    func pin(_ url: URL, directory: Bool) throws -> [String: Any] {
      var status = stat()
      guard lstat(url.path, &status) == 0,
        status.st_mode & S_IFMT == (directory ? S_IFDIR : S_IFREG) else {
        throw H3CheckpointError.invalid("Identity timing input changed type.")
      }
      return ["path": url.path, "device": Int(status.st_dev), "inode": UInt64(status.st_ino),
        "bytes": Int64(status.st_size), "mtimeSeconds": Int(status.st_mtimespec.tv_sec),
        "mtimeNanoseconds": Int(status.st_mtimespec.tv_nsec),
        "ctimeSeconds": Int(status.st_ctimespec.tv_sec), "ctimeNanoseconds": Int(status.st_ctimespec.tv_nsec)]
    }
    let directories = [root, root.appendingPathComponent("pages")]
    let metadata = ["config.json", "quant_config.json", "paged_manifest.json"].map { root.appendingPathComponent($0) }
    let files = metadata + [source.fixed] + source.blocks
    func identities() throws -> Data {
      let values = try directories.map { try pin($0, directory: true) }
        + files.map { try pin($0, directory: false) }
      return try JSONSerialization.data(withJSONObject: values, options: .sortedKeys)
    }
    let before = try identities()
    let metadataHashes = try metadata.map { ["path": $0.path, "sha256": sha(try Data(contentsOf: $0))] }
    // Declared before measurement: at least 0.5 s median gain across the
    // successful canonical path's 1,568 checks is required for a speed claim.
    let calls = 1568, minimumMedianGainSeconds = 0.5
    var oldSeconds: [Double] = [], newSeconds: [Double] = [], runs: [[String: Any]] = []
    func oldCheck() throws {
      if H3CheckpointSource.isPaged(root) { try H3CheckpointSource.inspect(root).checkUnchanged() }
    }
    // Warm both routes before the matched ABBAABBA sequence.
    try oldCheck()
    try H3CheckpointSource.checkUnchanged(root)
    for old in [true, false, false, true, true, false, false, true] {
      try H3CheckpointSource.checkUnchanged(root)
      let began = DispatchTime.now().uptimeNanoseconds
      for _ in 0..<calls {
        if old { try oldCheck() }
        else { try H3CheckpointSource.checkUnchanged(root) }
      }
      let seconds = Double(DispatchTime.now().uptimeNanoseconds - began) / 1_000_000_000
      if old { oldSeconds.append(seconds) } else { newSeconds.append(seconds) }
      runs.append(["route": old ? "old_duplicate_walk" : "production_deduplicated", "seconds": seconds])
      XCTAssertEqual(try identities(), before, "Every pinned source identity must remain unchanged.")
    }
    func median(_ values: [Double]) -> Double {
      let ordered = values.sorted(); return (ordered[1] + ordered[2]) / 2
    }
    let gain = median(oldSeconds) - median(newSeconds)
    let package = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
      .deletingLastPathComponent().deletingLastPathComponent()
    let sourceURL = package.appendingPathComponent("Sources/H3MLX/H3CheckpointSource.swift")
    let receipt: [String: Any] = ["scope": "Actual installed FastH3 VSA header-only identity timing; no inference, MLX arrays or model payload reads; not a whole-render speed proof.",
      "checkpoint": root.path, "identities": try JSONSerialization.jsonObject(with: before),
      "metadata": metadataHashes, "sourceSHA256": sha(try Data(contentsOf: sourceURL)),
      "testSHA256": sha(try Data(contentsOf: URL(fileURLWithPath: #filePath))),
      "testExecutable": CommandLine.arguments.first ?? "", "callsPerBatch": calls,
      "identityStatsPerWalk": 56, "runs": runs, "oldMedianSeconds": median(oldSeconds),
      "newMedianSeconds": median(newSeconds), "medianGainSeconds": gain,
      "predeclaredMinimumMedianGainSeconds": minimumMedianGainSeconds,
      "meaningfulSpeedGatePassed": gain >= minimumMedianGainSeconds]
    try JSONSerialization.data(withJSONObject: receipt, options: [.prettyPrinted, .sortedKeys])
      .write(to: URL(fileURLWithPath: outputPath), options: .withoutOverwriting)
    XCTAssertGreaterThanOrEqual(gain, minimumMedianGainSeconds,
      "Do not promote this candidate as a meaningful speed fix below the predeclared metadata-only gate.")
  }

}
