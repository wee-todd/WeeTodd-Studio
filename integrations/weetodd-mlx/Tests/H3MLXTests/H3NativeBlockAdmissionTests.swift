import Foundation
import TensorIO
import XCTest
@testable import H3MLX

final class H3NativeBlockAdmissionTests: XCTestCase {
  private func adapter(strength: Float = 1, profile: H3LoRAProfile = .turbo,
    layout: H3LoRAQKVLayout = .contiguousQKV, start: Int = 0) throws -> H3LoRAAdapter {
    try H3LoRAAdapter(url: URL(fileURLWithPath: "/private/nnc-admission-unused.safetensors"),
      strength: strength, profile: profile, qkvLayout: layout, startAfterEvaluations: start)
  }

  private func settings(task: String = "ref2va", steps: Int = 5,
    method: H3SamplingMethod = .euler, adapters: [H3LoRAAdapter]? = nil,
    context: Int = 0, refinement: Bool = false, rows: Int = 39967) throws {
    try H3NativeBlockAdmission.validateSettings(task: task, steps: steps,
      samplingMethod: method, adapters: adapters ?? [try adapter()],
      contextFrames: context, isRefinement: refinement, packedRows: rows)
  }

  func testBackendRawValuesAndEligibleBoundarySettings() throws {
    XCTAssertEqual(H3TransformerBackend.mlx.rawValue, "mlx")
    XCTAssertEqual(H3TransformerBackend.nncExperimental.rawValue, "nnc_experimental")
    XCTAssertNil(H3TransformerBackend(rawValue: "nnc"))
    try settings(rows: 1)
    try settings(rows: 40000)
    try settings(adapters: [adapter(profile: .auto, layout: .auto)])
  }

  func testUnsupportedTaskScheduleAndContextFailBeforeFiles() throws {
    for task in ["t2va", "i2va", "fl2va", "vsa", "Ref2VA"] {
      XCTAssertThrowsError(try settings(task: task))
    }
    for steps in [4, 6] { XCTAssertThrowsError(try settings(steps: steps)) }
    XCTAssertThrowsError(try settings(method: .resMultistep))
    XCTAssertThrowsError(try settings(context: 1))
    XCTAssertThrowsError(try settings(context: -1))
    XCTAssertThrowsError(try settings(refinement: true))
    for rows in [0, -1, 40001] { XCTAssertThrowsError(try settings(rows: rows)) }
  }

  func testUnsupportedAdapterStackStrengthProfileAndLayout() throws {
    XCTAssertThrowsError(try settings(adapters: []))
    XCTAssertThrowsError(try settings(adapters: [adapter(), adapter()]))
    let strengths: [Float] = [0, 0.5, -1, 1.01]
    for strength in strengths {
      XCTAssertThrowsError(try settings(adapters: [adapter(strength: strength)]))
    }
    XCTAssertThrowsError(try settings(adapters: [adapter(profile: .standard)]))
    XCTAssertThrowsError(try settings(adapters: [adapter(layout: .nativeInterleaved)]))
    XCTAssertThrowsError(try settings(adapters: [adapter(start: 1)]))
  }

  private typealias Header = [String: (String, [UInt64])]
  private func header(alphaShape: [UInt64] = []) -> Header {
    var result: Header = [:]
    for index in 0..<50 {
      for (target, output, input, rank) in [
        ("attn.qkv_proj", UInt64(21504), UInt64(5376), UInt64(384)),
        ("attn.out_proj", 5376, 7168, 128),
        ("mlp.fc1", 28672, 5376, 128), ("mlp.fc2", 5376, 14336, 128),
      ] {
        let base = "diffusion_model.blocks.\(index)." + target
        result[base + ".lora_A.weight"] = ("BF16", [rank, input])
        result[base + ".lora_B.weight"] = ("BF16", [output, rank])
        result[base + ".alpha"] = ("F32", alphaShape)
      }
    }
    return result
  }

  /// Logical factor bytes are sparse holes. This fixture writes only its small
  /// JSON header and never reads/allocates an adapter tensor payload.
  private func descriptors(_ tensors: Header) throws -> [String: TensorDescriptor] {
    let url = FileManager.default.temporaryDirectory
      .appendingPathComponent("h3-nnc-header-\(UUID().uuidString).safetensors")
    defer { try? FileManager.default.removeItem(at: url) }
    var records: [String: Any] = [:]
    var cursor: UInt64 = 0
    for name in tensors.keys.sorted() {
      let (dtype, shape) = tensors[name]!
      let width: UInt64 = dtype == "F32" ? 4 : 2
      let count = shape.reduce(UInt64(1), *)
      let end = cursor + count * width
      records[name] = ["dtype": dtype, "shape": shape, "data_offsets": [cursor, end]]
      cursor = end
    }
    let json = try JSONSerialization.data(withJSONObject: records, options: [.sortedKeys])
    var length = UInt64(json.count).littleEndian
    var bytes = withUnsafeBytes(of: &length) { Data($0) }
    bytes.append(json)
    try bytes.write(to: url, options: [.withoutOverwriting])
    let handle = try FileHandle(forWritingTo: url)
    defer { try? handle.close() }
    try handle.truncate(atOffset: UInt64(bytes.count) + cursor)
    return try SafeTensorFile(url: url).tensors
  }

  func testAllFiftyCompleteUniformComfyHeadersAndScalarShapes() throws {
    let alphaShapes: [[UInt64]] = [[], [1]]
    for alpha in alphaShapes {
      try H3NativeBlockAdmission.validateAdapterHeader(tensors: descriptors(header(alphaShape: alpha)))
    }
    var h = header()
    h["diffusion_model.blocks.3.adaln_proj.linear.alpha"] = ("F32", [])
    try H3NativeBlockAdmission.validateAdapterHeader(tensors: descriptors(h))
  }

  func testMissingPairNonUniformRankDtypeAndDimensionsFail() throws {
    let base = "diffusion_model.blocks.49.mlp.fc2"
    var h = header()
    h.removeValue(forKey: base + ".lora_B.weight")
    XCTAssertThrowsError(try H3NativeBlockAdmission.validateAdapterHeader(tensors: descriptors(h)))
    h = header(); h[base + ".lora_A.weight"] = ("BF16", [64, 14336])
    h[base + ".lora_B.weight"] = ("BF16", [5376, 64])
    XCTAssertThrowsError(try H3NativeBlockAdmission.validateAdapterHeader(tensors: descriptors(h)))
    h = header(); h[base + ".lora_A.weight"] = ("F16", [128, 14336])
    XCTAssertThrowsError(try H3NativeBlockAdmission.validateAdapterHeader(tensors: descriptors(h)))
    h = header(); h[base + ".lora_B.weight"] = ("BF16", [5375, 128])
    XCTAssertThrowsError(try H3NativeBlockAdmission.validateAdapterHeader(tensors: descriptors(h)))
    h = header(); h[base + ".alpha"] = ("BF16", [])
    XCTAssertThrowsError(try H3NativeBlockAdmission.validateAdapterHeader(tensors: descriptors(h)))
    h = header(); h[base + ".alpha"] = ("F32", [2])
    XCTAssertThrowsError(try H3NativeBlockAdmission.validateAdapterHeader(tensors: descriptors(h)))
  }

  func testUnexpectedMainBlockAndWrongNamespaceTargetsFail() throws {
    for name in ["diffusion_model.blocks.0.attn.q_norm.lora_A.weight",
      "model.diffusion_model.blocks.0.mlp.fc2.alpha",
      "diffusion_model.transformer_blocks.0.mlp.fc2.alpha"] {
      var h = header(); h[name] = ("F32", [])
      XCTAssertThrowsError(try H3NativeBlockAdmission.validateAdapterHeader(tensors: descriptors(h)))
    }
  }

  private func marker(_ object: [String: Any], width: UInt64 = 5376) throws {
    try H3NativeBlockAdmission.validateQuantizationMarker(
      JSONSerialization.data(withJSONObject: object), inputWidth: width)
  }

  func testQuantizationMarkerTypesAndGroupSemanticsMatchWorker() throws {
    try marker(["format": "int8_tensorwise"])
    try marker(["format": "int8_tensorwise", "convrot": true])
    for group in [4, 16, 64, 256] {
      try marker(["format": "int8_tensorwise", "convrot": true, "convrot_groupsize": group])
    }
    try marker(["format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 1024], width: 7168)
    // A disabled rotation ignores the optional group, as the actual worker does.
    try marker(["format": "int8_tensorwise", "convrot": false, "convrot_groupsize": "unused"])
    let invalid: [[String: Any]] = [
      ["format": "affine"], ["format": "int8_tensorwise", "extra": 1],
      ["format": "int8_tensorwise", "convrot": 1],
      ["format": "int8_tensorwise", "convrot": true, "convrot_groupsize": true],
      ["format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 16.5],
      ["format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 32],
      ["format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 1024],
    ]
    for object in invalid { XCTAssertThrowsError(try marker(object)) }
  }

  func testCoreAlphaMustBeFiniteStrictlyPositive() throws {
    try H3NativeBlockAdmission.validateCoreAlpha(128)
    try H3NativeBlockAdmission.validateCoreAlpha(Float.leastNonzeroMagnitude)
    let rejected: [Float] = [0, -0.0, -1, .nan, .infinity, -.infinity]
    for alpha in rejected { XCTAssertThrowsError(try H3NativeBlockAdmission.validateCoreAlpha(alpha)) }
  }

  func testCancelledAdmissionStopsBeforeOpeningFiles() async throws {
    let adapter = try adapter()
    let work = Task<Void, Error> {
      withUnsafeCurrentTask { $0?.cancel() }
      try H3NativeBlockAdmission.validateSettings(task: "ref2va", steps: 5,
        samplingMethod: .euler, adapters: [adapter], contextFrames: 0,
        isRefinement: false, packedRows: 39967)
    }
    do { try await work.value; XCTFail("Cancelled settings were admitted.") }
    catch is CancellationError { }
    let inspect = Task<Void, Error> {
      withUnsafeCurrentTask { $0?.cancel() }
      try H3NativeBlockAdmission.inspect(checkpoint: adapter.url, adapter: adapter.url)
    }
    do { try await inspect.value; XCTFail("Cancelled file inspection was admitted.") }
    catch is CancellationError { }
  }

  func testInspectRejectsDirectoryAndInvalidCheckpointBeforeAdapterLoading() throws {
    let directory = FileManager.default.temporaryDirectory
      .appendingPathComponent("h3-nnc-admission-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
    defer { try? FileManager.default.removeItem(at: directory) }
    let absent = directory.appendingPathComponent("absent.safetensors")
    XCTAssertThrowsError(try H3NativeBlockAdmission.inspect(checkpoint: directory, adapter: absent))
    let invalid = directory.appendingPathComponent("invalid.safetensors")
    try Data("not safetensors".utf8).write(to: invalid)
    XCTAssertThrowsError(try H3NativeBlockAdmission.inspect(checkpoint: invalid, adapter: absent))
  }

  func testInstalledHeaderOnlyNNCAdmissionWhenProvided() throws {
    let env = ProcessInfo.processInfo.environment
    guard let checkpoint = env["WEETODD_H3_NNC_ADMISSION_CHECKPOINT"],
      let adapter = env["WEETODD_H3_NNC_ADMISSION_ADAPTER"] else {
      throw XCTSkip("Set both NNC admission paths for CPU header/marker/alpha validation only.")
    }
    try H3NativeBlockAdmission.inspect(checkpoint: URL(fileURLWithPath: checkpoint),
      adapter: URL(fileURLWithPath: adapter))
  }
}
