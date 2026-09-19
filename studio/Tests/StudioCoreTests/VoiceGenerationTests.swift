import Foundation
import XCTest
@testable import StudioCore
final class VoiceGenerationTests: XCTestCase {
  func testReferenceModesNeverSilentlyFallBack() throws {
    var d = VoiceDraft(); d.modelPath = "/model"; d.text = "Hello"
    XCTAssertThrowsError(try d.request())
    d.reference = VoiceReference(path: "/ref.wav", duration: 5, transcript: "A sample.")
    XCTAssertNoThrow(try d.request())
    d.engine = .fishS2Pro; d.referenceMode = .speakerIdentityOnly
    XCTAssertThrowsError(try d.request())
    d.referenceMode = .synthetic; d.reference = nil
    XCTAssertNoThrow(try d.request())
    d.engine = .qwen3TTS
    XCTAssertThrowsError(try d.request())
  }
  func testRequestHasExplicitEngineAndNativeReferenceMode() throws {
    var d = VoiceDraft(); d.modelPath = "/model"; d.text = "Hello"
    d.reference = VoiceReference(path: "/ref.wav", duration: 5, transcript: "A sample.")
    let object = try JSONSerialization.jsonObject(with: JSONEncoder().encode(d.request())) as! [String: Any]
    XCTAssertEqual(object["engine"] as? String, "qwen3TTS")
    XCTAssertEqual(object["reference_mode"] as? String, "audioAndTranscript")
  }
}
