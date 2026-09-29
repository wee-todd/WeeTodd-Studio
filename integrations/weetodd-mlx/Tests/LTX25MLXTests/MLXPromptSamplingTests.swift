import XCTest
import MLX
import LTX25Engine
import LTX25MLX

final class MLXPromptSamplingTests:XCTestCase {
  func testTextFeedsSamplerAfterPreflightAndSnapshotsCallerInputs() throws {
    let support=MLXSamplingTests(), f=try support.fixture()
    let inputs=f.inputs.filter { !$0.key.hasSuffix("_text") }.mapValues { MLXArray($0) }
    var encoded=false
    let runner=try MLXPromptSamplingRunner(configuration:f.configuration,blockCount:1,encode:{ prompt in
      XCTAssertEqual(prompt,"A red fox."); encoded=true
      for input in inputs.values { input[.ellipsis] = MLXArray.zeros(input.shape) }
      return f.inputs.filter { $0.key.hasSuffix("_text") }.mapValues { MLXArray($0) }
    })
    let result=try runner.evaluate(prompt:"A red fox.",inputs:inputs,
      schedule:SamplingSchedule(sigmas:f.schedule.sigmas,eta:f.schedule.eta),
      fixedWeights:{ name,shape in XCTAssertTrue(encoded); return try support.weight(name,shape) },
      blockWeights:{ try support.weight("transformer_blocks.\($0)."+$1,$2) },
      noise:{ index,name,shape in MLXArray(f.noise["\(index).\(name)"]!,shape) })
    for name in ["video","audio"] {
      XCTAssertLessThan(zip(result[name]!.asArray(Float.self),f.expected[name]!).map { abs($0-$1) }.max()!,0.0001)
    }
  }
  func testInvalidRequestFailsBeforeTextAndTextFailureCanRetry() throws {
    enum Stop:Error { case stop }
    let support=MLXSamplingTests(), f=try support.fixture()
    let inputs=f.inputs.filter { !$0.key.hasSuffix("_text") }.mapValues { MLXArray($0) }
    var encodes=0, loads=0
    let runner=try MLXPromptSamplingRunner(configuration:f.configuration,blockCount:1,encode:{ _ in
      encodes += 1; throw Stop.stop
    })
    let fixed:MLXDenoiser.FixedProvider={ _,_ in loads += 1; throw Stop.stop }
    let blocks:MLXDenoiser.BlockProvider={ _,_,_ in loads += 1; throw Stop.stop }
    let ancestral=try SamplingSchedule(sigmas:[1,0.5,0])
    XCTAssertThrowsError(try runner.evaluate(prompt:"x",inputs:inputs,schedule:ancestral,fixedWeights:fixed,blockWeights:blocks))
    let euler=try SamplingSchedule(sigmas:[1,0],eta:0)
    XCTAssertThrowsError(try runner.evaluate(prompt:"x",inputs:[:],schedule:euler,fixedWeights:fixed,blockWeights:blocks))
    XCTAssertEqual(encodes,0); XCTAssertEqual(loads,0)
    for _ in 0..<2 {
      XCTAssertThrowsError(try runner.evaluate(prompt:"x",inputs:inputs,schedule:euler,fixedWeights:fixed,blockWeights:blocks))
    }
    XCTAssertEqual(encodes,2); XCTAssertEqual(loads,0)
  }
}
