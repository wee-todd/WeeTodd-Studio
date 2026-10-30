import Foundation
import InferenceContracts
import MLX
import XCTest
@testable import H3MLX

final class H3RegenerateRecipeTests:XCTestCase {
  static func fixture() throws -> [String:Any] {
    let hash=String(repeating:"a",count:64)
    let components=try NativeCapabilityDescriptor.h3Regenerate.requiredRoles.map { role -> [String:Any] in
      let identity=NativeComponentIdentity(family:.h3,role:role,
        sourceSHA256:role == .sharpInput ? H3SharpCurve.tableSHA256:hash,layout:"fixture")
      return ["path":"/missing/\(role.rawValue).safetensors","identity":try JSONSerialization.jsonObject(with:JSONEncoder().encode(identity))]
    }
    let decoder=NativeComponentIdentity(family:.h3,role:.decoder,sourceSHA256:hash,layout:"released-h3")
    return ["version":1,"task":"h3_regenerate","profile":"publisherThreeStep","schedule":"publisherThreeStep",
      "components":components,"source":["contentSHA256":hash,"audioSHA256":hash,"normalization":"h3_sampler_normalized_v1",
        "width":32,"height":32,"visibleFrames":5,"paddedFrames":5,"fps":24,"encoderProfile":"nativeNormalizedLatents"],
      "target":["width":32,"height":32],"promptSHA256":hash,"embeddingsSHA256":hash,"noiseSHA256":hash,"silenceSHA256":hash,
      "decoder":["kind":"releasedH3","identity":try JSONSerialization.jsonObject(with:JSONEncoder().encode(decoder))],
      "budget":["maximumStageBytes":UInt64(32*1024*1024*1024),"availableTemporaryBytes":UInt64(32*1024*1024*1024),"textRows":1]]
  }
  func testCanonicalRoundtripAndIdentityChangesBeforeModelLoading() throws {
    let fixture=try Self.fixture(), data=try JSONSerialization.data(withJSONObject:fixture)
    let prepared=try H3RegenerateRecipe.prepare(data:data)
    XCTAssertEqual(try H3RegenerateRecipe.prepare(data:prepared.encodedRecipe).fingerprint,prepared.fingerprint)
    for key in ["promptSHA256","embeddingsSHA256","noiseSHA256","silenceSHA256"] {
      var changed=fixture;changed[key]=String(repeating:"b",count:64)
      XCTAssertNotEqual(try H3RegenerateRecipe.prepare(data:JSONSerialization.data(withJSONObject:changed)).fingerprint,prepared.fingerprint)
    }
    var changed=fixture
    var components=changed["components"] as! [[String:Any]],identity=components[0]["identity"] as! [String:Any]
    identity["sourceSHA256"]=String(repeating:"c",count:64);components[0]["identity"]=identity;changed["components"]=components
    XCTAssertNotEqual(try H3RegenerateRecipe.prepare(data:JSONSerialization.data(withJSONObject:changed)).fingerprint,prepared.fingerprint)
  }
  func testUnknownFieldsExtractedStackAndCadenceRejectWithoutOpeningModels() throws {
    let fixture=try Self.fixture()
    for key in ["steps","sigma","guidance","unexpected"] {
      var changed=fixture;changed[key]=4
      XCTAssertThrowsError(try H3RegenerateRecipe.prepare(data:JSONSerialization.data(withJSONObject:changed)))
    }
    var changed=fixture;changed["profile"]="extractedRefiner384"
    XCTAssertThrowsError(try H3RegenerateRecipe.prepare(data:JSONSerialization.data(withJSONObject:changed)))
    changed=fixture
    var source=changed["source"] as! [String:Any];source["fps"]=30;changed["source"]=source
    XCTAssertThrowsError(try H3RegenerateRecipe.prepare(data:JSONSerialization.data(withJSONObject:changed)))
    source["fps"]=24;source["encoderProfile"]="publisherPosteriorFP32";changed["source"]=source
    XCTAssertThrowsError(try H3RegenerateRecipe.prepare(data:JSONSerialization.data(withJSONObject:changed)))
    source["posteriorNoiseSHA256"]=String(repeating:"d",count:64);changed["source"]=source
    XCTAssertNoThrow(try H3RegenerateRecipe.prepare(data:JSONSerialization.data(withJSONObject:changed)))
  }
  func testRunnerRejectsMismatchedPreparedRowsBeforeMissingCheckpointAccess() throws {
    var fixture=try Self.fixture()
    // Reach the intended shape guard after the earlier prepared-base admission.
    var components=fixture["components"] as! [[String:Any]]
    for index in components.indices {
      var identity=components[index]["identity"] as! [String:Any]
      if identity["role"] as? String == NativeComponentRole.transformer.rawValue {
        identity["sourceSHA256"]="37488d7799577d40db8e74318311344cb909c38632c0f693c52976a5509811ff"
        components[index]["identity"]=identity
      }
    }
    fixture["components"]=components
    let prepared=try H3RegenerateRecipe.prepare(data:JSONSerialization.data(withJSONObject:fixture))
    XCTAssertThrowsError(try H3RegenerateRunner.run(prepared:prepared,source:.zeros([1,1,96]),noise:.zeros([1,1,96]),
      encodedSilence:.zeros([1,prepared.admission.audioRows,32],dtype:.bfloat16),textEmbeddings:.zeros([1,1,5120]),textTags:[0])) { error in
      guard case H3CheckpointError.invalid(let message)=error else { return XCTFail("Unexpected failure: \(error)") }
      XCTAssertTrue(message.contains("prepared source"))
    }
  }
}
