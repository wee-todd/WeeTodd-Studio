import Foundation
import XCTest
@testable import StudioCore

final class H3RegenerateStrictTransportTests:XCTestCase {
  func testRuntimeAndRequestRejectIgnoredInterpreterAndSamplerControls() throws {
    let runtime=H3RegenerateRuntime(ffmpegPath:"/ffmpeg",ffprobePath:"/ffprobe",workerPath:"/worker")
    var runtimeJSON=try JSONSerialization.jsonObject(with:JSONEncoder().encode(runtime)) as! [String:Any]
    runtimeJSON["pythonPath"]="/python"
    XCTAssertThrowsError(try JSONDecoder().decode(H3RegenerateRuntime.self,from:JSONSerialization.data(withJSONObject:runtimeJSON)))
    var requestJSON=try JSONSerialization.jsonObject(with:JSONEncoder().encode(H3RegenerateRequest(sourcePath:"/source.mov",runtime:runtime))) as! [String:Any]
    requestJSON["sampler"]="ignored"
    XCTAssertThrowsError(try JSONDecoder().decode(H3RegenerateRequest.self,from:JSONSerialization.data(withJSONObject:requestJSON)))
  }
  func testNormalizedRowsRejectUnknownIdentityOrNormalizationControls() throws {
    let sha=String(repeating:"a",count:64)
    let value=H3RegenerateLatentSource(file:.init(path:"/rows",sha256:sha,bytes:2*96*4),sourceSHA256:sha,firstFrame:0,visibleFrames:5,paddedFrames:5,width:32,height:32)
    let data=try JSONEncoder().encode(value)
    XCTAssertEqual(try JSONDecoder().decode(H3RegenerateLatentSource.self,from:data),value)
    var object=try JSONSerialization.jsonObject(with:data) as! [String:Any];object["scale"]="ignored"
    XCTAssertThrowsError(try JSONDecoder().decode(H3RegenerateLatentSource.self,from:JSONSerialization.data(withJSONObject:object)))
    object.removeValue(forKey:"scale");var file=object["file"] as! [String:Any];file["tensorName"]="ignored";object["file"]=file
    XCTAssertThrowsError(try JSONDecoder().decode(H3RegenerateLatentSource.self,from:JSONSerialization.data(withJSONObject:object)))
  }
  func testOnlyTheTwoVerifiedOriginalBasePreparationsAreAdmitted() {
    XCTAssertEqual(H3RegeneratePreparation.supportedOriginalBaseSHA256,["600307eba6a67388b333aa5a94537a9ea83da65a6f37664ba22cf079e40a1f3b","37488d7799577d40db8e74318311344cb909c38632c0f693c52976a5509811ff"])
  }
}
