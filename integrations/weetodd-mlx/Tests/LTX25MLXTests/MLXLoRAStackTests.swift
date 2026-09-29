import XCTest
import MLX
import LTX25MLX
import AdapterRuntime
import InferenceTestSupport

final class MLXLoRAStackTests: XCTestCase {
  func test23FixedAdapterChangesSwiftProjectionOutputAtQualifiedScale() throws {
    let stem = "diffusion_model.proj_out"
    var down = [Float](repeating: 0, count: 4096), up = [Float](repeating: 0, count: 128)
    down[0] = 1; up[0] = 2
    try withTensorFile(metadata: ["model_version": "2.3.0", "lora_alpha": "2"], tensors: [
      (stem + ".lora_down.weight", [1, 4096], "F32"),
      (stem + ".lora_up.weight", [128, 1], "F32")], payloads: [
        stem + ".lora_down.weight": down.withUnsafeBytes { Data($0) },
        stem + ".lora_up.weight": up.withUnsafeBytes { Data($0) }]) { url in
      let stack = try MLXLoRAStack(adapters: [LoRAAdapter(path: url.path, strength: 0.5)])
      let factors = try stack.loadFixed("proj_out.weight")
      XCTAssertEqual(factors.count, 1)
      XCTAssertEqual(factors[0].scale, 1)
      var input = [Float](repeating: 0, count: 4096); input[0] = 1
      let base = try MLXWeight(dense: .zeros([128, 4096]))
      let result = try base.projected(MLXArray(input, [1, 4096]), adapters: factors).asArray(Float.self)
      XCTAssertEqual(result[0], 2, accuracy: 1e-6)
      XCTAssertTrue(result.dropFirst().allSatisfy { $0 == 0 })
    }
  }
  func testCompatible23FactorsStayOrderedAndBounded() throws {
    let stem="transformer_blocks.0.attn1.to_q"
    try withTensorFile(metadata:["model_version":"2.3.0","lora_alpha":"2"],tensors:[
      (stem+".lora_A.weight",[1,4096],"BF16"), (stem+".lora_B.weight",[4096,1],"BF16")]) { url in
      let stack=try MLXLoRAStack(adapters:[LoRAAdapter(path:url.path,strength:0.5),LoRAAdapter(path:url.path,strength:-0.25)])
      let factors=try stack.load(block:0)
      XCTAssertEqual(factors["attn1.to_q.weight"]?.map(\.scale),[1,-0.5])
      XCTAssertEqual(factors["attn1.to_q.weight"]?.first?.down.dtype,.bfloat16)
      XCTAssertTrue(try stack.load(block:1).isEmpty)
      XCTAssertThrowsError(try MLXLoRAStack(adapters:[LoRAAdapter(path:url.path,strength:1)],maximumFactorBytes:4).load(block:0))
      let disabled=try MLXLoRAStack(adapters:[LoRAAdapter(path:"/missing",strength:1,enabled:false)])
      XCTAssertTrue(try disabled.load(block:0).isEmpty)
      let zero=try MLXLoRAStack(adapters:[LoRAAdapter(path:url.path,strength:0)])
      XCTAssertTrue(try zero.load(block:0).isEmpty)
    }
  }
}
