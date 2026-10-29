import XCTest

@testable import Drama3Contracts

final class ModelAdmissionTests: XCTestCase {
  func testInspectionRejectsDiscoveryOnlyFolder() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    try Data("{\"model_type\":\"fish_drama3\"}".utf8).write(
      to: root.appendingPathComponent("config.json"))
    XCTAssertThrowsError(try Drama3ModelLoader.inspect(at: root))
  }
  func testControlHeadExtractionKeepsRequestedRowsInOrder() throws {
    let values: [Float] = [1, 2, 3, 4, 5, 6, 7, 8]
    let got = try Drama3ControlRows.extract(values: values, width: 2, rows: [3, 0, 2])
    XCTAssertEqual(got, [7, 8, 1, 2, 5, 6])
    XCTAssertThrowsError(try Drama3ControlRows.extract(values: values, width: 2, rows: [4]))
  }
}
extension ModelAdmissionTests {
  func testCodecAdmissionRequiresCompleteSchemaShapesAndFloatingWeights() throws {
    var tensors = Drama3CodecTensorSchema.shapes.mapValues {
      Drama3TensorInfo(dtype: "F32", shape: $0, data_offsets: [0, 0])
    }
    XCTAssertEqual(tensors.count, 538)
    try Drama3CodecAdmission.validate(tensors)
    let name = "quantizer.semantic_quantizer.quantizers.0.codebook.weight"
    XCTAssertEqual(tensors[name]?.shape, [4096, 8])
    let original = tensors.removeValue(forKey: name)!
    XCTAssertThrowsError(try Drama3CodecAdmission.validate(tensors))
    tensors[name] = .init(dtype: "F32", shape: [4096, 7], data_offsets: [0, 0])
    XCTAssertThrowsError(try Drama3CodecAdmission.validate(tensors))
    tensors[name] = .init(dtype: "I32", shape: original.shape, data_offsets: [0, 0])
    XCTAssertThrowsError(try Drama3CodecAdmission.validate(tensors))
  }
  func testTensorHeaderRejectsByteCountMismatchAndIntegerOverflow() throws {
    let file = FileManager.default.temporaryDirectory.appendingPathComponent(
      UUID().uuidString + ".safetensors")
    defer { try? FileManager.default.removeItem(at: file) }
    func write(_ shape: [Int]) throws {
      let header = try JSONSerialization.data(withJSONObject: [
        "bad": ["dtype": "F32", "shape": shape, "data_offsets": [0, 4]]
      ])
      var n = UInt64(header.count).littleEndian
      var bytes = Data(bytes: &n, count: 8)
      bytes.append(header)
      bytes.append(Data(repeating: 0, count: 4))
      try bytes.write(to: file)
    }
    try write([2, 2])
    XCTAssertThrowsError(try Drama3TensorFile(file))
    try write(Array(repeating: 2_000_000, count: 8))
    XCTAssertThrowsError(try Drama3TensorFile(file))
  }
}
