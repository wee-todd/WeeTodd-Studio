import Foundation
import XCTest
@testable import StudioCore

final class H3RegenerateSettingsTests: XCTestCase {
  func testFixedPublisherDefaultsRoundTripAndRejectIgnoredControls() throws {
    let settings = H3RegenerateSettings()
    XCTAssertEqual(settings.targetWidth, 2560)
    XCTAssertEqual(settings.targetHeight, 1440)
    XCTAssertEqual(settings.schedule.rawValue, "publisherThreeStep")
    XCTAssertEqual(settings.encoderProfile.rawValue, "publisherPosteriorFP32")
    let data = try JSONEncoder().encode(settings)
    XCTAssertEqual(try JSONDecoder().decode(H3RegenerateSettings.self, from: data), settings)
    var object = try JSONSerialization.jsonObject(with: data) as! [String: Any]
    object["guidance"] = 7.5
    XCTAssertThrowsError(try JSONDecoder().decode(H3RegenerateSettings.self,
      from: JSONSerialization.data(withJSONObject: object)))
    object.removeValue(forKey: "guidance"); object["version"] = 99
    XCTAssertThrowsError(try JSONDecoder().decode(H3RegenerateSettings.self,
      from: JSONSerialization.data(withJSONObject: object)))
  }
  func testSourceClockIsPreservedOrRejectedWithoutResampling() throws {
    let settings = H3RegenerateSettings()
    XCTAssertNoThrow(try settings.validateSourceClock(numerator: 24, denominator: 1))
    XCTAssertNoThrow(try settings.validateSourceClock(numerator: 24000, denominator: 1000))
    XCTAssertThrowsError(try settings.validateSourceClock(numerator: 30000, denominator: 1001))
    XCTAssertThrowsError(try settings.validateSourceClock(numerator: 0, denominator: 1))
  }
}
