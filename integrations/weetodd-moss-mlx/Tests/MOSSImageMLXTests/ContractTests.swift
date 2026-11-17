import Foundation
import XCTest
@testable import MOSSImageMLX

final class ContractTests: XCTestCase {
  func testSelectedSimpleScheduleCanUseItsFullTrainingGrid() throws {
    var settings = MOSSGenerationSettings(); settings.schedule = "simple"; settings.steps = 1000
    try settings.validate()
    settings.schedule = "release"; XCTAssertThrowsError(try settings.validate())
  }
  func testMemoryAdmissionIncludesResidentAdapterDeltas() {
    let gib = 1_073_741_824
    let baseline = MOSSImagePreflight.estimate(textBytes: 16*gib, retainedBytes: 10*gib, adapterBytes: 0, tokenRows: 1000, pixels: 262144)
    let stack = MOSSImagePreflight.estimate(textBytes: 16*gib, retainedBytes: 10*gib, adapterBytes: 12*gib, tokenRows: 1000, pixels: 262144)
    XCTAssertEqual(baseline, 20*gib); XCTAssertEqual(stack, 27*gib)
  }
  func testSharedComponentDirectorySymlinkResolvesBeforeEnumeratingShards() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    let real = root.appendingPathComponent("real"), link = root.appendingPathComponent("vae")
    try FileManager.default.createDirectory(at: real, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    try FileManager.default.createSymbolicLink(at: link, withDestinationURL: real)
    let header = Data(#"{"test":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#.utf8)
    var size = UInt64(header.count).littleEndian
    var bytes = withUnsafeBytes(of: &size) { Data($0) }; bytes.append(header); bytes.append(Data(repeating: 0, count: 4))
    try bytes.write(to: real.appendingPathComponent("weights.safetensors"))
    let index = try MOSSCheckpointIndex(directory: link)
    try index.require("test", shape: [1]); try index.checkUnchanged()
  }
  func testScheduleKeepsTrainingGridAndRejectsCollapsedStepCount() throws {
    XCTAssertEqual(try MOSSSchedule.sigmas(steps: 1), [1, 0])
    let four = try MOSSSchedule.sigmas(steps: 4)
    XCTAssertEqual(four.count, 5)
    XCTAssertEqual(four[1], 0.91, accuracy: 0.00001)
    XCTAssertEqual(four[2], 0.717, accuracy: 0.00001)
    XCTAssertEqual(four[3], 0.024, accuracy: 0.00001)
    XCTAssertThrowsError(try MOSSSchedule.sigmas(steps: 1000))
    XCTAssertThrowsError(try MOSSSchedule.sigmas(steps: 25, shift: .nan))
    XCTAssertEqual(try MOSSSchedule.sigmas(steps: 50).count, 51)
  }
  func testConfigurationRejectsUnsupportedGeometryAndGuidance() throws {
    var settings = MOSSGenerationSettings()
    try settings.validate()
    settings.width = 1023
    XCTAssertThrowsError(try settings.validate())
    settings.width = 1024; settings.guidance = .infinity
    XCTAssertThrowsError(try settings.validate())
    settings.guidance = 3; settings.skipLayers = [42]
    XCTAssertThrowsError(try settings.validate())
  }
  func testModelAdmissionRejectsAdjacentTransformerAndWrongSmallDecoder() throws {
    let transformer = Data(#"{"dim":3072,"ffn_dim":8192,"num_heads":24,"num_layers":42,"text_dim":4096,"latent_channels":128,"freq_dim":256,"rope_base":10000}"#.utf8)
    try MOSSModelContract.validateTransformerConfig(transformer)
    XCTAssertThrowsError(try MOSSModelContract.validateTransformerConfig(Data(String(decoding: transformer, as: UTF8.self).replacingOccurrences(of: "3072", with: "5376").utf8)))
    let vae = Data(#"{"decoder_block_out_channels":[96,192,384,384],"latent_channels":32,"patch_size":[2,2],"norm_num_groups":32,"layers_per_block":2,"batch_norm_eps":0.0001}"#.utf8)
    try MOSSModelContract.validateVAEConfig(vae)
    XCTAssertThrowsError(try MOSSModelContract.validateVAEConfig(Data(String(decoding: vae, as: UTF8.self).replacingOccurrences(of: "96,192,384,384", with: "128,256,512,512").utf8)))
  }
  func testTensorIndexRejectsDuplicateKeysAndChangedFiles() throws {
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: folder) }
    let file = folder.appendingPathComponent("weights.safetensors")
    try writeTensor(file, name: "value", shape: [2], data: Data([0, 0, 128, 63, 0, 0, 0, 64]))
    let index = try MOSSCheckpointIndex(directory: folder)
    XCTAssertEqual(index.descriptors["value"]?.shape, [2])
    try index.checkUnchanged()
    try Data([0]).write(to: file)
    XCTAssertThrowsError(try index.checkUnchanged())
  }
  func testMissingTensorAndWrongShapeFailBeforeLoad() throws {
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: folder) }
    try writeTensor(folder.appendingPathComponent("one.safetensors"), name: "value", shape: [2], data: Data(repeating: 0, count: 8))
    let index = try MOSSCheckpointIndex(directory: folder)
    XCTAssertThrowsError(try index.require("missing", shape: [2]))
    XCTAssertThrowsError(try index.require("value", shape: [3]))
    try writeTensor(folder.appendingPathComponent("two.safetensors"), name: "value", shape: [2], data: Data(repeating: 0, count: 8))
    XCTAssertThrowsError(try MOSSCheckpointIndex(directory: folder))
  }
  private func writeTensor(_ url: URL, name: String, shape: [Int], data: Data) throws {
    let object: [String: Any] = [name: ["dtype": "F32", "shape": shape, "data_offsets": [0, data.count]]]
    let json = try JSONSerialization.data(withJSONObject: object, options: .sortedKeys)
    var length = UInt64(json.count).littleEndian
    var bytes = withUnsafeBytes(of: &length) { Data($0) }; bytes.append(json); bytes.append(data)
    try bytes.write(to: url)
  }
}
