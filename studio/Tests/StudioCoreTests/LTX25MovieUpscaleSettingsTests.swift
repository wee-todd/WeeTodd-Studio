import XCTest
@testable import StudioCore
final class LTX25MovieUpscaleSettingsTests:XCTestCase {
  func testDefaultDoesNotEnableUnqualifiedMovieInference() throws {
    var value=LTX25MovieUpscaleSettings()
    XCTAssertFalse(value.experimentalEnabled);XCTAssertThrowsError(try value.validate())
    value.experimentalEnabled=true;XCTAssertNoThrow(try value.validate())
    XCTAssertEqual(value.mode,.refine);XCTAssertEqual(value.refinementStrength,0.35)
    let saved=try JSONEncoder().encode(value)
    XCTAssertEqual(try JSONDecoder().decode(LTX25MovieUpscaleSettings.self,from:saved),value)
  }
  func testUnusedAnchorsAndImplicitResumeRejectBeforeMediaPreparation() throws {
    var value=LTX25MovieUpscaleSettings();value.experimentalEnabled=true
    value.mode = .latentOnly;XCTAssertThrowsError(try value.validate())
    value.anchors = .none;XCTAssertNoThrow(try value.validate())
    value.resume=true;XCTAssertThrowsError(try value.validate())
    value.chunking=true;XCTAssertNoThrow(try value.validate())
    value.maximumAudioDriftSeconds = .nan;XCTAssertThrowsError(try value.validate())
  }
  func testExplicitAdapterCannotBecomeAnIgnoredInactiveSetting() throws {
    var value=LTX25MovieUpscaleSettings();value.experimentalEnabled=true
    value.pixelSpatialAdapterPath="/models/pixel.safetensors"
    XCTAssertThrowsError(try value.validate())
    value.mode = .pixelSpatial;XCTAssertNoThrow(try value.validate())
    value.pixelSpatialAdapterPath="relative.safetensors";XCTAssertThrowsError(try value.validate())
    value.pixelSpatialAdapterPath="/models/\u{0}pixel";XCTAssertThrowsError(try value.validate())
    value.pixelSpatialAdapterPath="/"+String(repeating:"a",count:4096);XCTAssertThrowsError(try value.validate())
  }
}
