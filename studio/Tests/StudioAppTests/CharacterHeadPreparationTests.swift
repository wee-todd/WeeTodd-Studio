import XCTest
import StudioCore
@testable import WeeToddStudio

final class CharacterHeadPreparationTests: XCTestCase {
  func testTransparentCutoutBecomesWhiteInTransport() throws {
    let rgba: [UInt8] = [20, 40, 60, 255, 90, 80, 70, 0]
    let result = try CharacterHeadPreparation.applyMask(rgba: rgba, mask: [255, 0], width: 2, height: 1)
    XCTAssertEqual(result.rgba, [20, 40, 60, 255, 90, 80, 70, 0])
    XCTAssertEqual(result.whiteRGB, [20, 40, 60, 255, 255, 255])
  }

  func testUnpremultiplicationPreservesWhiteHairAtHalfAlpha() {
    XCTAssertEqual(CharacterHeadPreparation.unpremultiply([128, 128, 128, 128]), [255, 255, 255, 128])
  }

  func testEmptyMaskIsRejectedInsteadOfClaimingRemoval() {
    XCTAssertThrowsError(try CharacterHeadPreparation.applyMask(
      rgba: [20, 40, 60, 255], mask: [0], width: 1, height: 1)) { error in
      XCTAssertEqual(error as? CharacterHeadPreparationError, .unusableMask)
    }
  }

  func testMultipleHeadsNeedExplicitSelection() {
    XCTAssertThrowsError(try CharacterHeadPreparation.resolveSelection(
      candidates: [PanelPixelRect(x: 0, y: 0, width: 10, height: 10), PanelPixelRect(x: 20, y: 0, width: 10, height: 10)], selection: nil)) { error in
      XCTAssertEqual(error as? CharacterHeadPreparationError, .selectionRequired)
    }
  }

  func testManualSourcePixelCropMapsToBoundedProxyAndBack() throws {
    let source = (width: 12_000, height: 8_000)
    let proxy = (width: 2_048, height: 1_365)
    let original = PanelPixelRect(x: 6_000, y: 2_000, width: 3_000, height: 3_000)
    let mapped = try CharacterHeadPreparation.mapRect(original, from: source, to: proxy)
    XCTAssertEqual(mapped, PanelPixelRect(x: 1_024, y: 341, width: 512, height: 513))
    let restored = try CharacterHeadPreparation.mapRect(mapped, from: proxy, to: source)
    XCTAssertLessThanOrEqual(abs(restored.x - original.x), 6)
    XCTAssertLessThanOrEqual(abs(restored.y - original.y), 6)
    XCTAssertLessThanOrEqual(abs(restored.width - original.width), 12)
    XCTAssertLessThanOrEqual(abs(restored.height - original.height), 12)
  }

  func testOrientationAwareDimensionsSwapForRotatedExifImages() throws {
    XCTAssertEqual(try CharacterHeadPreparation.orientedDimensions(
      rawWidth: 12_000, rawHeight: 8_000, orientation: 6).width, 8_000)
    XCTAssertEqual(try CharacterHeadPreparation.orientedDimensions(
      rawWidth: 12_000, rawHeight: 8_000, orientation: 6).height, 12_000)
    XCTAssertEqual(try CharacterHeadPreparation.orientedDimensions(
      rawWidth: 12_000, rawHeight: 8_000, orientation: 1).width, 12_000)
    XCTAssertThrowsError(try CharacterHeadPreparation.orientedDimensions(
      rawWidth: 32_769, rawHeight: 100, orientation: 1)) { error in
      XCTAssertEqual(error as? CharacterHeadPreparationError, .invalidDimensions)
    }
  }

  func testOversizedInputIsRejectedBeforeDecode() async throws {
    let source = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: source) }
    FileManager.default.createFile(atPath: source.path, contents: nil)
    let handle = try FileHandle(forWritingTo: source)
    try handle.truncate(atOffset: UInt64(64 * 1_024 * 1_024 + 1)); try handle.close()
    do {
      _ = try await CharacterHeadPreparation().prepare(source: source, selection: nil)
      XCTFail("Expected an input size rejection")
    } catch {
      XCTAssertEqual(error as? CharacterHeadPreparationError, .inputTooLarge)
    }
  }

}
