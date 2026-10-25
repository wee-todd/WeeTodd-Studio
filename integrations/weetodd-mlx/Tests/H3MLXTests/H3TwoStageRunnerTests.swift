import Foundation
import XCTest
@testable import H3MLX

final class H3TwoStageRunnerTests: XCTestCase {
  private func adapter(profile:String) throws -> URL {try adapter(metadata:["adapter_profile":profile])}
  private func adapter(metadata:[String:String]) throws -> URL {
    let url=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString+".safetensors")
    addTeardownBlock {try? FileManager.default.removeItem(at:url)}
    let object:[String:Any]=["__metadata__":metadata,"dummy":["dtype":"F32","shape":[1],"data_offsets":[0,4]]]
    var header=try JSONSerialization.data(withJSONObject:object);while header.count%8 != 0 {header.append(32)}
    var size=UInt64(header.count).littleEndian
    var data=withUnsafeBytes(of:&size){Data($0)};data.append(header);data.append(Data(count:4));try data.write(to:url)
    return url
  }
  func testAmbiguousGenericRoleAndCountOnlyNeedExplicitProfile() throws {
    for metadata in [["inference_steps":"20"],["adapter_role":"transformer_lora"],[:]] {
      let url=try adapter(metadata:metadata)
      XCTAssertThrowsError(try H3LoRAFile.isTurboForTwoStage(H3LoRAAdapter(url:url,strength:1),requestedSteps:5,samplingMethod:.euler))
      XCTAssertFalse(try H3LoRAFile.isTurboForTwoStage(H3LoRAAdapter(url:url,strength:1,profile:.standard),requestedSteps:5,samplingMethod:.euler))
    }
    for role in ["style","character","standard"] {
      let url=try adapter(metadata:["adapter_role":role])
      XCTAssertFalse(try H3LoRAFile.isTurboForTwoStage(H3LoRAAdapter(url:url,strength:1),requestedSteps:5,samplingMethod:.euler))
    }
  }
  func testTypedPassesKeepStandardLoRAsAndOmitOnlyIdentifiedTurbo() throws {
    let turbo=try adapter(profile:"turbo"),standard=try adapter(profile:"standard")
    let request=try H3T2VARequest(prompt:"Robot",width:1344,height:768,durationSeconds:3,seed:42,requestedSteps:5,
      transformer:URL(fileURLWithPath:"/base"),qwenPages:URL(fileURLWithPath:"/qwen"),tokenizer:URL(fileURLWithPath:"/tokenizer"),
      videoVAE:URL(fileURLWithPath:"/video"),audioVAE:URL(fileURLWithPath:"/audio"),
      loRAAdapters:[H3LoRAAdapter(url:turbo,strength:1),H3LoRAAdapter(url:standard,strength:-0.5)],canvasAdmission:.spatialRefinement)
    let plan=try H3TwoStagePlan(width:1344,height:768,durationSeconds:3)
    let runner=try H3TwoStageRunner(task:.text(request),plan:plan,resizeMethod:.bilinear)
    XCTAssertEqual(runner.baseTask.geometry.width,672)
    XCTAssertEqual(runner.finalTask.geometry.width,1344)
    XCTAssertEqual(runner.baseTask.adapters.count,2)
    XCTAssertEqual(runner.finalTask.adapters.count,1)
    XCTAssertEqual(runner.finalTask.adapters.first?.url,standard)
    XCTAssertEqual(runner.finalTask.adapters.first?.strength,-0.5)
    XCTAssertEqual(runner.omittedTurboAdapters,[turbo.path])
    XCTAssertThrowsError(try runner.preflight()) // Missing model headers fail before any weighted stage.
  }
}
