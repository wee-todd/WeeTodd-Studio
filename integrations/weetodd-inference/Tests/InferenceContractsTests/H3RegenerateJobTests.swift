import CryptoKit
import Foundation
import XCTest
@testable import InferenceContracts

final class H3RegenerateJobTests:XCTestCase {
  func fixture() throws -> [String:Any] {
    let sha=String(repeating:"a",count:64),prompt="A fixed source prompt."
    var components:[String:Any]=[:]
    for role in NativeCapabilityDescriptor.h3Regenerate.requiredRoles {
      let identity=NativeComponentIdentity(family:.h3,role:role,sourceSHA256:sha,layout:"fixture")
      components[role.rawValue]=["path":"/models/\(role.rawValue)","identity":try JSONSerialization.jsonObject(with:JSONEncoder().encode(identity))]
    }
    return ["version":1,"engine":"h3","task":"h3_regenerate",
      "source":["kind":"rgb24","payload":["path":"/source.rgb24","sha256":sha],"originalSHA256":sha,"audioSHA256":sha,
        "width":1344,"height":768,"visibleFrames":121,"paddedFrames":124,"fps":24,"encoderProfile":"nativePosteriorMeanF16"],
      "prompt":prompt,"promptSHA256":SHA256.hash(data:Data(prompt.utf8)).map{String(format:"%02x",$0)}.joined(),
      "qwen":["checkpointRoot":"/models/qwen","tokenizer":["path":"/models/tokenizer.json","sha256":sha],"files":[["path":"/models/qwen/pages.safetensors","sha256":sha]]],
      "components":components,"schedule":"publisherThreeStep","seed":UInt64.max,"target":["width":2560,"height":1440],
      "latentUpscaleMode":"directPublisher","decoder":"releasedH3","budget":["maximumStageBytes":UInt64(64)<<30,"availableTemporaryBytes":UInt64(64)<<30]]
  }
  func testImmutableVisibleRGBTransportAndExactSeedRoundtrip() throws {
    let data=try JSONSerialization.data(withJSONObject:fixture()),job=try H3RegenerateJob.decode(data)
    XCTAssertEqual(job.source.payloadBytes,UInt64(121*1344*768*3));XCTAssertEqual(job.source.latentFrames,37)
    XCTAssertEqual(job.seed,UInt64.max)
    XCTAssertEqual(try H3RegenerateJob.decode(JSONEncoder().encode(job)),job)
  }
  func testRejectsUnknownPayloadTimingAndPromptMutationBeforeFileAccess() throws {
    var object=try fixture();object["engineRecipe"]=["unsafe":"extra"]
    XCTAssertThrowsError(try H3RegenerateJob.decode(JSONSerialization.data(withJSONObject:object)))
    object=try fixture();object["prompt"]="changed"
    XCTAssertThrowsError(try H3RegenerateJob.decode(JSONSerialization.data(withJSONObject:object)))
    for fps in [23.976,25.0,30.0] {
      object=try fixture();var source=object["source"] as! [String:Any];source["fps"]=fps;object["source"]=source
      XCTAssertThrowsError(try H3RegenerateJob.decode(JSONSerialization.data(withJSONObject:object)))
    }
    object=try fixture();var source=object["source"] as! [String:Any];source["kind"]="normalizedVideoRows";object["source"]=source
    XCTAssertThrowsError(try H3RegenerateJob.decode(JSONSerialization.data(withJSONObject:object)))
    source["encoderProfile"]="nativeNormalizedLatents";object["source"]=source
    XCTAssertEqual(try H3RegenerateJob.decode(JSONSerialization.data(withJSONObject:object)).source.payloadBytes,UInt64(37*42*24*96*4))
  }
  func testPublisherPosteriorRequiresSeparateEncoderAndSeed() throws {
    var object=try fixture(),source=object["source"] as! [String:Any]
    source["encoderProfile"]="publisherPosteriorFP32";object["source"]=source
    XCTAssertThrowsError(try H3RegenerateJob.decode(JSONSerialization.data(withJSONObject:object)))
    var components=object["components"] as! [String:Any];components["publisherEncoder"]=components["videoVAE"];object["components"]=components
    XCTAssertThrowsError(try H3RegenerateJob.decode(JSONSerialization.data(withJSONObject:object)))
    object["posteriorSeed"]=17
    XCTAssertNoThrow(try H3RegenerateJob.decode(JSONSerialization.data(withJSONObject:object)))
  }
}
