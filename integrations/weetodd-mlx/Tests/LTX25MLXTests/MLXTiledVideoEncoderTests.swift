import Foundation
import MLX
import XCTest
import TensorIO
@testable import LTX25MLX

final class MLXTiledVideoEncoderTests: XCTestCase {
  func testAdmittedFullSpatialWindowAvoidsArtificialGuideSeam() throws {
    let plan = try MLXVideoEncodeTilePlan(frames: 17, width: 768, height: 448,
      maximumOwnedBufferBytes: 4 * 1024 * 1024 * 1024)
    XCTAssertEqual(plan.tiles.count, 1)
    XCTAssertEqual(plan.tiles.first?.width, 768)
    XCTAssertEqual(plan.tiles.first?.height, 448)
    XCTAssertTrue(plan.coverageWeights().allSatisfy { $0 == 1 })
  }

  func testExplicitSpatialTilingStillAvailableForIndependentSeamTests() throws {
    let plan = try MLXVideoEncodeTilePlan(frames: 17, width: 768, height: 448,
      spatialPolicy: .tiles, maximumOwnedBufferBytes: 4 * 1024 * 1024 * 1024)
    XCTAssertEqual(plan.tiles.count, 2)
    XCTAssertTrue(plan.coverageWeights().allSatisfy { $0 > 0 })
  }

  func testFullRateGuideIsPartitionedWithinFourGiBAndCoversEveryLatent() throws {
    let plan = try MLXVideoEncodeTilePlan(frames: 129, width: 1376, height: 768,
      tilePixels: 512, maximumOwnedBufferBytes: 4 * 1024 * 1024 * 1024)
    XCTAssertEqual(plan.latentShape, [17, 24, 43, 128])
    XCTAssertEqual(plan.tiles.count, 48)
    XCTAssertEqual(plan.tiles.first?.frames, 33)
    XCTAssertEqual(plan.tiles.last?.frames, 17)
    XCTAssertTrue(plan.tiles.allSatisfy { $0.ownedBufferBytes <= 4 * 1024 * 1024 * 1024 })
    XCTAssertTrue(plan.coverageWeights().allSatisfy { $0 > 0 })
    XCTAssertThrowsError(try MLXVideoEncodePlan(frames: 129, width: 1376, height: 768))
  }

  func testTilePlannerRejectsUnsafeOrIncompleteGeometry() throws {
    let larger = try MLXVideoEncodeTilePlan(frames: 129, width: 1376,
      height: 768, tilePixels: 768, maximumOwnedBufferBytes: 6 * 1024 * 1024 * 1024)
    XCTAssertEqual(larger.tiles.count, 16)
    XCTAssertTrue(larger.coverageWeights().allSatisfy { $0 > 0 })
    XCTAssertThrowsError(try MLXVideoEncodeTilePlan(frames: 129, width: 1376,
      height: 768, tilePixels: 768, maximumOwnedBufferBytes: 4 * 1024 * 1024 * 1024))
    XCTAssertThrowsError(try MLXVideoEncodeTilePlan(frames: 130, width: 1376, height: 768))
    XCTAssertThrowsError(try MLXVideoEncodeTilePlan(frames: 129, width: 1376, height: 768,
      tilePixels: 500))
  }

  func testRGB24TileReaderPreservesFrameAndPixelOrder() throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    let path = directory.appendingPathComponent("guide.rgb24")
    let bytes = (0..<(9 * 32 * 32 * 3)).map { UInt8($0 % 251) }
    try Data(bytes).write(to: path)
    let plan = try MLXVideoEncodeTilePlan(frames: 9, width: 32, height: 32)
    let tile = try MLXTiledVideoEncoder.loadTile(path: path, plan: plan, tile: plan.tiles[0])
      .asArray(Float.self)
    XCTAssertEqual(tile.count, bytes.count)
    XCTAssertEqual(tile[0], -1, accuracy: 0.00001)
    XCTAssertEqual(tile[1], 2.0 / 255.0 - 1, accuracy: 0.00001)
    XCTAssertEqual(tile[(8 * 32 * 32 + 31 * 32 + 31) * 3 + 2],
      Float(bytes.last!) * 2 / 255 - 1, accuracy: 0.00001)
    try Data(bytes.dropLast()).write(to: path)
    XCTAssertThrowsError(try MLXTiledVideoEncoder.loadTile(path: path, plan: plan,
      tile: plan.tiles[0]))
  }

  func testInstalledTiledGuideAgainstIndependentMLXOracleWhenRequested() throws {
    let env = ProcessInfo.processInfo.environment
    guard let checkpoint = env["WEETODD_MLX_VIDEO_ENCODER"],
      let directory = env["WEETODD_MLX_TILED_VIDEO_ORACLE"] else {
      throw XCTSkip("Installed tiled video encoder oracle is opt-in.")
    }
    try verifyOracle(checkpoint: checkpoint, directory: directory, frames: 9, tiles: 2)
  }

  func testInstalledTemporalAndSpatialTileSeamsAgainstIndependentMLXOracleWhenRequested() throws {
    let env = ProcessInfo.processInfo.environment
    guard let checkpoint = env["WEETODD_MLX_VIDEO_ENCODER"],
      let directory = env["WEETODD_MLX_TEMPORAL_TILE_ORACLE"] else {
      throw XCTSkip("Installed temporal tiled video encoder oracle is opt-in.")
    }
    try verifyOracle(checkpoint: checkpoint, directory: directory, frames: 41, tiles: 4)
  }

  private func verifyOracle(checkpoint: String, directory: String,
    frames: Int, tiles: Int) throws {
    let root = URL(fileURLWithPath: directory)
    let expected = try MLXWeight.read(SafeTensorFile(url: root.appendingPathComponent("latent.safetensors")),
      "latent")
    let plan = try MLXVideoEncodeTilePlan(frames: frames, width: 192, height: 128,
      tilePixels: 128, spatialPolicy: .tiles)
    XCTAssertEqual(plan.tiles.count, tiles)
    let output = try MLXTiledVideoEncoder.encode(guide: root.appendingPathComponent("guide.rgb24"),
      checkpoint: URL(fileURLWithPath: checkpoint), plan: plan)
    XCTAssertEqual(output.shape, expected.shape)
    let difference = output - expected
    let maxAbsolute = abs(difference).max().item(Float.self)
    let relative = sqrt((difference * difference).sum() / (expected * expected).sum()).item(Float.self)
    print("TILED_VIDEO_ORACLE frames=\(frames) maxabs=\(maxAbsolute) relative=\(relative) peak_mlx=\(Memory.peakMemory)")
    XCTAssertLessThan(maxAbsolute, 0.003)
    XCTAssertLessThan(relative, 0.0003)
  }
}
