import XCTest
import LTX25Engine
import AdapterRuntime
import InferenceTestSupport
import LTX25MLX

/// CPU/header-only coverage, safe while a separate app is using the GPU.
final class MLXDenoiserContractTests: XCTestCase {
  func testSharedLayoutAndMathAreAvailableWithoutNNC() throws {
    let c=try AVBlockConfiguration(videoTokens:5,audioTokens:3,textTokens:4)
    let shapes=DenoiserLayout.weightShapes(c)
    XCTAssertEqual(shapes.count,58)
    XCTAssertEqual(shapes["adaln_single.linear.weight"],[36864,4096])
    XCTAssertEqual(shapes["audio_proj_out.weight"],[128,2048])
    XCTAssertEqual(DenoiserLayout.heads(c).count,8)
    XCTAssertEqual(try DenoiserMath.timestep(0),[Float](repeating:1,count:128)+[Float](repeating:0,count:128))
    XCTAssertThrowsError(try DenoiserMath.timestep(.nan))
    let rotary=try DenoiserMath.rotary(positions:[10,1024,1024],axes:3,tokens:1,heads:2,headWidth:8,maximumPositions:[20,2048,2048])
    XCTAssertEqual(rotary.cos,[Float](repeating:1,count:8))
  }
  func testFixedAdapterTargetsMustBeConsumedByTheSelectedModel() throws {
    let stem="adaln_single.linear"
    try withTensorFile(metadata:["model_version":"2.3.0"],tensors:[
      (stem+".lora_A.weight",[1,4096],"BF16"),(stem+".lora_B.weight",[36864,1],"BF16")]) { url in
      let stack=try MLXLoRAStack(adapters:[LoRAAdapter(path:url.path,strength:0.8)])
      XCTAssertNoThrow(try stack.validateTargets([stem:[36864,4096]]))
      XCTAssertThrowsError(try stack.validateTargets([:]))
      XCTAssertThrowsError(try stack.validateTargets([stem:[1,4096]]))
    }
  }
}
