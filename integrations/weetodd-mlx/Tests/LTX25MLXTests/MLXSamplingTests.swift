import XCTest
import MLX
import LTX25Engine
import LTX25MLX

final class MLXSamplingTests:XCTestCase {
  func testFrozenAudioDoesNotStepOrDrawNoiseButKeepsJointVideoEvaluation() throws {
    let f=try fixture(),runner=try MLXSamplingRunner(configuration:f.configuration,blockCount:1)
    let inputs=f.inputs.mapValues { MLXArray($0) }
    let source=inputs["audio_latent"]!.reshaped([f.configuration.audioTokens,128])
    var noiseNames:[String]=[]
    let result=try runner.evaluate(inputs,schedule:SamplingSchedule(sigmas:[1,0.5,0]),frozenAudio:true,
      fixedWeights:weight,blockWeights:{ try self.weight("transformer_blocks.\($0)."+$1,$2) },
      noise:{ _,name,shape in noiseNames.append(name);return .ones(shape) },
      preview:{ state,_ in XCTAssertEqual(state["audio"]!.asArray(Float.self),source.asArray(Float.self)) })
    XCTAssertEqual(noiseNames,["video"])
    XCTAssertEqual(result["audio"]!.asArray(Float.self),source.asArray(Float.self))
    XCTAssertTrue(MLX.isFinite(result["video"]!).all().item(Bool.self))
  }
  func testAudioReferenceMaskProtectsSourceTokensThroughAncestralNoise() throws {
    let f=try fixture(),runner=try MLXSamplingRunner(configuration:f.configuration,blockCount:1)
    let inputs=f.inputs.mapValues { MLXArray($0) }
    let clean=inputs["audio_latent"]!.reshaped([f.configuration.audioTokens,128])
    let condition=try MLXAudioDenoiseCondition(clean:clean,mask:[0,0.5,1])
    var previews=0
    let result=try runner.evaluate(inputs,schedule:SamplingSchedule(sigmas:[1,0.75,0.5,0]),
      audioConditioning:condition,fixedWeights:weight,
      blockWeights:{ try self.weight("transformer_blocks.\($0)."+$1,$2) },
      noise:{ _,_,shape in MLXArray.ones(shape)*4 },preview:{ output,_ in
        previews += 1
        XCTAssertEqual(output["audio"]![0].asArray(Float.self),clean[0].asArray(Float.self))
      })
    XCTAssertEqual(previews,3)
    XCTAssertEqual(result["audio"]![0].asArray(Float.self),clean[0].asArray(Float.self))
    var reads=0
    XCTAssertThrowsError(try runner.evaluate(inputs,schedule:SamplingSchedule(sigmas:[1,0]),
      audioConditioning:condition,frozenAudio:true,
      fixedWeights:{ name,shape in reads += 1;return try self.weight(name,shape) },
      blockWeights:{ try self.weight("transformer_blocks.\($0)."+$1,$2) }))
    XCTAssertEqual(reads,0)
  }
  struct Fixture:Decodable {
    struct Schedule:Decodable { let sigmas:[Double]; let eta:Double }
    let configuration:AVBlockConfiguration
    let inputs:[String:[Float]]
    let schedule:Schedule
    let noise:[String:[Float]]
    let expected:[String:[Float]]
    let bf16_state:Bool?
  }
  func fixture() throws -> Fixture {
    let url=Bundle.module.url(forResource:"trajectory-reference",withExtension:"json",subdirectory:"Fixtures")!
    return try JSONDecoder().decode(Fixture.self,from:Data(contentsOf:url))
  }
  func weight(_ name:String,_ shape:[Int]) throws -> MLXWeight {
    let seed=name.utf8.reduce(0) { $0+Int($1) }
    let norm:Float=name.hasSuffix("q_norm.weight") || name.hasSuffix("k_norm.weight") ? 1 : 0
    return try MLXWeight(dense:MLXArray((0..<shape.reduce(1,*)).map { Float(($0*17+seed)%31-15)/128+norm },shape))
  }
  func testAllOneMaskUsesGlobalTimestepLikeUnconditionedSampling() throws {
    let f=try fixture(),runner=try MLXSamplingRunner(configuration:f.configuration,blockCount:1)
    let inputs=f.inputs.mapValues { MLXArray($0) },schedule=try SamplingSchedule(sigmas:[0.731,0.730,0],eta:0)
    let blocks:MLXDenoiser.BlockProvider={ try self.weight("transformer_blocks.\($0)."+$1,$2) }
    let base=try runner.evaluate(inputs,schedule:schedule,bfloat16State:["video","audio"],fixedWeights:weight,blockWeights:blocks)
    let condition=try MLXVideoDenoiseCondition(clean:inputs["video_latent"]!.reshaped([5,128]),mask:[1,1,1,1,1])
    let actual=try runner.evaluate(inputs,schedule:schedule,videoConditioning:condition,bfloat16State:["video","audio"],fixedWeights:weight,blockWeights:blocks)
    for name in ["video","audio"] { XCTAssertEqual(actual[name]!.asArray(Float.self),base[name]!.asArray(Float.self)) }
  }
  func testReleasedBF16StateBoundariesApplyAtEveryStep() throws {
    let f=try fixture(),runner=try MLXSamplingRunner(configuration:f.configuration,blockCount:1)
    var previews=0
    _=try runner.evaluate(f.inputs.mapValues { MLXArray($0) },
      schedule:SamplingSchedule(sigmas:f.schedule.sigmas,eta:f.schedule.eta),bfloat16State:["video","audio"],
      fixedWeights:weight,blockWeights:{ try self.weight("transformer_blocks.\($0)."+$1,$2) },
      noise:{ index,name,shape in MLXArray(f.noise["\(index).\(name)"]!,shape) },preview:{ output,_ in
        previews += 1
        for value in output.values { XCTAssertEqual(value.asArray(Float.self),value.asType(.bfloat16).asType(.float32).asArray(Float.self)) }
      })
    XCTAssertEqual(previews,3)
    var weighted=false
    XCTAssertThrowsError(try runner.evaluate(f.inputs.mapValues { MLXArray($0) },schedule:SamplingSchedule(sigmas:[1,0],eta:0),bfloat16State:["other"],
      fixedWeights:{ name,shape in weighted=true;return try self.weight(name,shape) },blockWeights:{ try self.weight("transformer_blocks.\($0)."+$1,$2) }))
    XCTAssertFalse(weighted)
    var missing=f.inputs.mapValues { MLXArray($0) };missing.removeValue(forKey:"video_latent")
    XCTAssertThrowsError(try runner.evaluate(missing,schedule:SamplingSchedule(sigmas:[1,0],eta:0),bfloat16State:["video"],
      fixedWeights:{ name,shape in weighted=true;return try self.weight(name,shape) },blockWeights:{ try self.weight("transformer_blocks.\($0)."+$1,$2) }))
    XCTAssertFalse(weighted)
  }
  func testAncestralTrajectoryMatchesIndependentMLXWithOneHeadLoad() throws {
    let f=try fixture(), runner=try MLXSamplingRunner(configuration:f.configuration,blockCount:1)
    var counts:[String:Int]=[:], completed:[Int]=[], blocks:[Int]=[]
    let result=try runner.evaluate(f.inputs.mapValues { MLXArray($0) },
      schedule:SamplingSchedule(sigmas:f.schedule.sigmas,eta:f.schedule.eta),
      fixedWeights:{ name,shape in counts[name,default:0] += 1; return try self.weight(name,shape) },
      blockWeights:{ try self.weight("transformer_blocks.\($0)."+$1,$2) },
      noise:{ index,name,shape in MLXArray(f.noise["\(index).\(name)"]!,shape) },
      stageProgress:{ index,event in if event.stage == "transformer" { blocks.append(index) } },
      progress:{ completed.append($0.completedSteps) },
      preview:{ _,_ in XCTAssertEqual(runner.residentWeightBytes,0) })
    XCTAssertEqual(completed,[1,2,3]); XCTAssertEqual(blocks,[1,2,3])
    XCTAssertEqual(counts["adaln_single.linear.weight"],1)
    XCTAssertEqual(counts["patchify_proj.weight"],3)
    for name in ["video","audio"] {
      let actual=result[name]!.asArray(Float.self), expected=f.expected[name]!
      XCTAssertLessThan(zip(actual,expected).map { abs($0-$1) }.max()!,0.0001)
    }
    XCTAssertEqual(runner.residentWeightBytes,0)
  }
  func testReferenceMaskedAncestralSamplingMatchesIndependentOracle() throws {
    try checkMaskedTrajectory("trajectory-reference-masked")
  }
  func testBF16MaskedSamplingMatchesProductionX0Wrapper() throws {
    try checkMaskedTrajectory("trajectory-bf16-masked")
  }
  func testCloseFloat32SigmasDoNotAliasInBF16HeadCache() throws {
    try checkMaskedTrajectory("trajectory-close-sigmas")
  }
  func testAncestralNoiseNeverChangesFullyProtectedEndpointTokens() throws {
    let f=try fixture(),runner=try MLXSamplingRunner(configuration:f.configuration,blockCount:1)
    let inputs=f.inputs.mapValues { MLXArray($0) }
    let clean=inputs["video_latent"]!.reshaped([5,128])
    let condition=try MLXVideoDenoiseCondition(clean:clean,mask:[0,1,0.5,1,0])
    var previews=0
    _ = try runner.evaluate(inputs,schedule:SamplingSchedule(sigmas:[1,0.75,0.5,0]),videoConditioning:condition,bfloat16State:f.bf16_state == true ? ["video","audio"] : [],
      fixedWeights:weight,blockWeights:{ try self.weight("transformer_blocks.\($0)."+$1,$2) },
      noise:{ _,_,shape in MLXArray.ones(shape)*4 },preview:{ output,_ in
        previews += 1
        for row in [0,4] { XCTAssertEqual(output["video"]![row].asArray(Float.self),clean[row].asArray(Float.self)) }
      })
    XCTAssertEqual(previews,3)
  }
  private func checkMaskedTrajectory(_ resource:String) throws {
    let url=Bundle.module.url(forResource:resource,withExtension:"json",subdirectory:"Fixtures")!
    let f=try JSONDecoder().decode(Fixture.self,from:Data(contentsOf:url))
    let runner=try MLXSamplingRunner(configuration:f.configuration,blockCount:1)
    let inputs=f.inputs.mapValues { MLXArray($0) }
    let condition=try MLXVideoDenoiseCondition(clean:inputs["video_latent"]!.reshaped([5,128]),mask:[0,1,0.5,1,0])
    let result=try runner.evaluate(inputs,schedule:SamplingSchedule(sigmas:f.schedule.sigmas,eta:f.schedule.eta),
      videoConditioning:condition,bfloat16State:f.bf16_state == true ? ["video","audio"] : [],fixedWeights:weight,
      blockWeights:{ try self.weight("transformer_blocks.\($0)."+$1,$2) },
      noise:{ index,name,shape in MLXArray(f.noise["\(index).\(name)"]!,shape) })
    for name in ["video","audio"] {
      let error=zip(result[name]!.asArray(Float.self),f.expected[name]!).map { abs($0-$1) }.max()!
      print("MASKED_TRAJECTORY \(resource) \(name) maxabs=\(error)")
      XCTAssertLessThan(error,resource == "trajectory-close-sigmas" ? 1e-6 : 0.0001)
    }
    XCTAssertEqual(result["video"]![0].asArray(Float.self),condition.clean[0].asArray(Float.self))
  }
  func testMissingNoiseFailsBeforeWeightsAndPreviewFailureAllowsRetry() throws {
    enum Stop:Error { case stop }
    let f=try fixture(), runner=try MLXSamplingRunner(configuration:f.configuration,blockCount:1)
    let inputs=f.inputs.mapValues { MLXArray($0) }
    var loads=0
    XCTAssertThrowsError(try runner.evaluate(inputs,schedule:SamplingSchedule(sigmas:f.schedule.sigmas),
      fixedWeights:{ _,_ in loads += 1; throw Stop.stop },blockWeights:{ _,_,_ in loads += 1; throw Stop.stop }))
    XCTAssertEqual(loads,0)
    let schedule=try SamplingSchedule(sigmas:[1,0.5,0],eta:0)
    let blocks:(Int,String,[Int]) throws -> MLXWeight = { try self.weight("transformer_blocks.\($0)."+$1,$2) }
    XCTAssertThrowsError(try runner.evaluate(inputs,schedule:schedule,fixedWeights:weight,blockWeights:blocks,
      preview:{ _,_ in throw Stop.stop }))
    XCTAssertEqual(runner.residentWeightBytes,0)
    let result=try runner.evaluate(inputs,schedule:schedule,fixedWeights:weight,blockWeights:blocks)
    XCTAssertEqual(result["video"]?.size,640)
  }

  func testPreviewMutationCannotAlterCurrentOrReturnedLatents() throws {
    let f=try fixture(), runner=try MLXSamplingRunner(configuration:f.configuration,blockCount:1)
    let inputs=f.inputs.mapValues { MLXArray($0) }
    let schedule=try SamplingSchedule(sigmas:[1,0.5,0],eta:0)
    let blocks:(Int,String,[Int]) throws -> MLXWeight = { try self.weight("transformer_blocks.\($0)."+$1,$2) }
    let base=try runner.evaluate(inputs,schedule:schedule,fixedWeights:weight,blockWeights:blocks)
    let result=try runner.evaluate(inputs,schedule:schedule,fixedWeights:weight,blockWeights:blocks,
      preview:{ values,_ in
        for value in values.values { value[.ellipsis] = MLXArray.zeros(value.shape) }
      })
    for name in ["video","audio"] { XCTAssertEqual(base[name]!.asArray(Float.self),result[name]!.asArray(Float.self)) }
  }

  func testCapturedInputMutationCannotChangePreparedSession() throws {
    let f=try fixture(), runner=try MLXSamplingRunner(configuration:f.configuration,blockCount:1)
    let inputs=f.inputs.mapValues { MLXArray($0) }
    let schedule=try SamplingSchedule(sigmas:[1,0.5,0],eta:0)
    let blocks:(Int,String,[Int]) throws -> MLXWeight = { try self.weight("transformer_blocks.\($0)."+$1,$2) }
    let base=try runner.evaluate(inputs,schedule:schedule,fixedWeights:weight,blockWeights:blocks)
    let result=try runner.evaluate(inputs,schedule:schedule,fixedWeights:weight,blockWeights:blocks,
      stageProgress:{ _,_ in
        for value in inputs.values { value[.ellipsis] = MLXArray.zeros(value.shape) }
      })
    for name in ["video","audio"] { XCTAssertEqual(base[name]!.asArray(Float.self),result[name]!.asArray(Float.self)) }
  }
  @MainActor func testNoiseCancellationDoesNotRequestNextModality() async throws {
    let canceled=try await Task {
      let f=try self.fixture(), runner=try MLXSamplingRunner(configuration:f.configuration,blockCount:1)
      var requested:[String]=[], blocks=0
      do {
        _=try runner.evaluate(f.inputs.mapValues { MLXArray($0) },schedule:SamplingSchedule(sigmas:[1,0.5,0]),
          fixedWeights:self.weight,blockWeights:{ index,name,shape in
            blocks += 1; return try self.weight("transformer_blocks.\(index)."+name,shape)
          },noise:{ _,name,shape in
            requested.append(name)
            withUnsafeCurrentTask { $0?.cancel() }
            return .zeros(shape)
          })
        return false
      } catch is CancellationError {
        XCTAssertEqual(requested,["video"]); XCTAssertEqual(blocks,0)
        return runner.residentWeightBytes == 0
      }
    }.value
    XCTAssertTrue(canceled)
    try testAncestralTrajectoryMatchesIndependentMLXWithOneHeadLoad()
  }

}
