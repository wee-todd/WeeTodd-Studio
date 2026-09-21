import XCTest
@testable import StudioCore

final class CharacterHeadReferenceTests: XCTestCase {
  func testReferenceRoundTripsIndependentArtifactIDsAndWhiteMatte() throws {
    let ids = (UUID(), UUID(), UUID(), UUID(), UUID())
    let value = CharacterHeadReference(originalAssetID: ids.0, headCropAssetID: ids.1,
      maskAssetID: ids.2, rgbaCutoutAssetID: ids.3, whiteMatteAssetID: ids.4,
      sourceSHA256: "abc", preprocessingVersion: "head-v1")
    let decoded = try JSONDecoder().decode(CharacterHeadReference.self, from: JSONEncoder().encode(value))
    XCTAssertEqual(decoded, value)
    XCTAssertEqual(decoded.matteRGB, [255, 255, 255])
    XCTAssertEqual(decoded.artifactSHA256.count, 4)
  }
}
