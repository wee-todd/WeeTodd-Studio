import XCTest
@testable import StudioCore

final class ImageSeedModeTests: XCTestCase {
  func testNewImageDraftKeepsRandomSentinelAcrossRepeatedRequestsAndPersistence() throws {
    let draft = DrawThingsImageDraft(destination: ImageAssetDestination(scope: .project, projectID: UUID()))
    XCTAssertTrue(draft.randomSeedEachGeneration)
    for id in ["first", "second"] {
      let config = try XCTUnwrap(try draft.request(id: id)["configuration"] as? [String: Any])
      XCTAssertEqual(config["seed"] as? Int, -1)
    }
    let restored = try JSONDecoder().decode(DrawThingsImageDraft.self, from: JSONEncoder().encode(draft))
    XCTAssertTrue(restored.randomSeedEachGeneration)
  }

  func testExistingFixedSeedIsPreservedUntilUserChoosesRandomMode() throws {
    var draft = DrawThingsImageDraft(destination: ImageAssetDestination(scope: .project, projectID: UUID()))
    draft.seed = 2789787039
    var restored = try JSONDecoder().decode(DrawThingsImageDraft.self, from: JSONEncoder().encode(draft))
    restored.randomSeedEachGeneration = false
    XCTAssertEqual(restored.seed, 2789787039)
    restored.randomSeedEachGeneration = true
    XCTAssertEqual(restored.seed, -1)
    restored.randomSeedEachGeneration = false
    XCTAssertEqual(restored.seed, 0)
  }
}
