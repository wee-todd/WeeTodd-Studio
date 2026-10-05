import XCTest
@testable import LTX25MLX

final class MLXMSRAudioRequestTests:XCTestCase {
  private func request() -> [String:Any] {
    let image:[String:Any]=["path":"/image.png","source_sha256":String(repeating:"a",count:64),
      "role":"subject","priority":"auto","reference_frames":"25","size_policy":"quality","strength":1,"attention_strength":1]
    return ["adapter_path":"/msr.safetensors","adapter_strength":1,"references":[image,image],
      "audio_references":[["path":"/voice.wav","source_sha256":String(repeating:"b",count:64),
        "image_slot":2,"source_start_seconds":0,"source_duration_seconds":8]]]
  }
  private func decode(_ value:[String:Any]) throws -> MLXMSRRequest {
    try JSONDecoder().decode(MLXMSRRequest.self,from:JSONSerialization.data(withJSONObject:value))
  }
  func testVoiceReferenceRoundTripsSparseIdentityAndExplicitInterval() throws {
    let value=try decode(request())
    XCTAssertEqual(value.audioReferences.first?.imageSlot,2)
    XCTAssertEqual(value.audioReferences.first?.sourceDurationSeconds,8)
    let copy=try JSONDecoder().decode(MLXMSRRequest.self,from:JSONEncoder().encode(value))
    XCTAssertEqual(copy.audioReferences.first?.imageSlot,2)
    XCTAssertEqual(copy.audioReferences.first?.effectiveDurationSeconds,5)
    var legacy=request();legacy.removeValue(forKey:"audio_references")
    XCTAssertTrue(try decode(legacy).audioReferences.isEmpty)
  }
  func testVoiceReferenceRejectsUnknownFieldsDuplicateOrUnpairedSlots() throws {
    var value=request();let voice=(value["audio_references"] as! [[String:Any]])[0]
    value["audio_references"]=[voice,voice];XCTAssertThrowsError(try decode(value))
    for slot:Any in [0,3,true] {
      var invalid=voice;invalid["image_slot"]=slot;value["audio_references"]=[invalid]
      XCTAssertThrowsError(try decode(value))
    }
    var invalid=voice;invalid["ignored"]=1;value["audio_references"]=[invalid]
    XCTAssertThrowsError(try decode(value))
    invalid=voice;invalid["source_start_seconds"] = -1;value["audio_references"]=[invalid]
    XCTAssertThrowsError(try decode(value))
    value=request();value["references"]=[(value["references"] as! [[String:Any]])[0]]
    XCTAssertThrowsError(try decode(value))
  }
}
