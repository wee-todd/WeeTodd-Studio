import XCTest
import MLX
import LTX25Engine
@testable import LTX25MLX

final class MLXDenoiserTests: XCTestCase {
  func testGeneratedKeyframeMarkerChangesOnlyMarkedProjectionRows() throws {
    let f = try fixture()
    let inputs = f.inputs.mapValues { MLXArray($0) }
    let blocks: MLXDenoiser.BlockProvider = { try self.weight("transformer_blocks.\($0)." + $1, $2) }
    let baseline = try MLXDenoiser(configuration: f.configuration, blockCount: 1)
      .evaluate(inputs, sigma: f.sigma, fixedWeights: weight, blockWeights: blocks)
    var markerReads = 0
    let marked = try MLXDenoiser(configuration: f.configuration, blockCount: 1, keyframeMarkerRows: 1)
      .evaluate(inputs, sigma: f.sigma, fixedWeights: { name, shape in
        if name == "keyframes_abs_pos_embedding" {
          markerReads += 1
          XCTAssertEqual(shape, [1, f.configuration.videoDimension])
          return try MLXWeight(dense: MLXArray.ones(shape) * 3)
        }
        return try self.weight(name, shape)
      }, blockWeights: blocks)
    XCTAssertEqual(markerReads, 1)
    XCTAssertGreaterThan((marked["video"]! - baseline["video"]!).abs().max().item(Float.self), 0.0001)
    XCTAssertThrowsError(try MLXDenoiser(configuration: f.configuration, blockCount: 1,
      keyframeMarkerRows: f.configuration.videoTokens + 1))
  }
  struct Fixture: Decodable {
    let configuration: AVBlockConfiguration
    let inputs: [String:[Float]]
    let sigma: Float
    let expected: [String:[Float]]
  }
  func fixture() throws -> Fixture {
    let url=Bundle.module.url(forResource:"denoiser-reference",withExtension:"json",subdirectory:"Fixtures")!
    return try JSONDecoder().decode(Fixture.self,from:Data(contentsOf:url))
  }
  func weight(_ name:String,_ shape:[Int]) throws -> MLXWeight {
    let seed=name.utf8.reduce(0) { $0+Int($1) }
    let norm:Float=name.hasSuffix("q_norm.weight") || name.hasSuffix("k_norm.weight") ? 1 : 0
    return try MLXWeight(dense:MLXArray((0..<shape.reduce(1,*)).map { Float(($0*17+seed)%31-15)/128+norm },shape))
  }
  func testFullDenoiserMatchesIndependentMLXFixture() throws {
    let f=try fixture(), runner=try MLXDenoiser(configuration:f.configuration,blockCount:1)
    var stages:[String]=[]
    let result=try runner.evaluate(f.inputs.mapValues { MLXArray($0) },sigma:f.sigma,
      fixedWeights:weight,blockWeights:{ try self.weight("transformer_blocks.\($0)."+$1,$2) },progress:{
      stages.append($0.stage)
    })
    for name in ["video","audio"] {
      let actual=result[name]!.asArray(Float.self), expected=f.expected[name]!
      XCTAssertEqual(actual.count,expected.count)
      XCTAssertLessThan(zip(actual,expected).map { abs($0-$1) }.max()!,0.0001,name)
    }
    XCTAssertEqual(stages.count,14)
    XCTAssertEqual(Array(stages.suffix(3)),["transformer_released","proj_out","audio_proj_out"])
    XCTAssertEqual(runner.residentWeightBytes,0)
  }
  func testNativeFixedProviderMatchesIndependentDenoiserAndRejectsReplacement() throws {
    let f=try fixture(),url=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString+".safetensors")
    defer { try? FileManager.default.removeItem(at:url) }
    var arrays:[String:MLXArray]=[:]
    for (name,shape) in DenoiserLayout.weightShapes(f.configuration) {
      arrays["model.diffusion_model."+name]=try weight(name,shape).tensor()
    }
    try save(arrays:arrays,url:url)
    let source=try MLXFixedSource(url:url,configuration:f.configuration,nativeLoading:true)
    for _ in 0..<2 {
      let runner=try MLXDenoiser(configuration:f.configuration,blockCount:1)
      let result=try runner.evaluate(f.inputs.mapValues { MLXArray($0) },sigma:f.sigma,
        fixedWeights:{ try source.read($0,shape:$1) },
        blockWeights:{ try self.weight("transformer_blocks.\($0)."+$1,$2) })
      for name in ["video","audio"] {
        XCTAssertLessThan(zip(result[name]!.asArray(Float.self),f.expected[name]!).map { abs($0-$1) }.max()!,0.0001)
      }
      XCTAssertEqual(runner.residentWeightBytes,0)
    }
    try Data(contentsOf:url).write(to:url,options:.atomic)
    XCTAssertThrowsError(try source.read("proj_out.bias",shape:[128]))
  }
  func testPerTokenVideoTimestepMatchesIndependentOracle() throws {
    let url=Bundle.module.url(forResource:"denoiser-per-token",withExtension:"json",subdirectory:"Fixtures")!
    let f=try JSONDecoder().decode(Fixture.self,from:Data(contentsOf:url))
    let runner=try MLXDenoiser(configuration:f.configuration,blockCount:1)
    let inputs=f.inputs.mapValues { MLXArray($0) }
    let output=try runner.evaluate(inputs,sigma:f.sigma,videoDenoiseMask:[0,1,0.5,1,0],fixedWeights:weight,
      blockWeights:{ try self.weight("transformer_blocks.\($0)."+$1,$2) })
    for name in ["video","audio"] {
      XCTAssertLessThan(zip(output[name]!.asArray(Float.self),f.expected[name]!).map { abs($0-$1) }.max()!,0.0001)
    }
    var reads=0
    for mask:[Float] in [[0],[0,1,Float.nan,1,0],[0,1,-0.1,1,0]] {
      XCTAssertThrowsError(try runner.evaluate(inputs,sigma:f.sigma,videoDenoiseMask:mask,
        fixedWeights:{ name,shape in reads += 1;return try self.weight(name,shape) },
        blockWeights:{ try self.weight("transformer_blocks.\($0)."+$1,$2) }))
    }
    XCTAssertEqual(reads,0)
  }
  func testFrozenAudioAndImageReferenceShareCompactVideoRows() throws {
    let f=try fixture(), runner=try MLXDenoiser(configuration:f.configuration,blockCount:1)
    let inputs=f.inputs.mapValues { MLXArray($0) }
    let mask:[Float]=[0,1,0.5,1,0]
    let prepared=try runner.prepare(inputs,sigmas:[f.sigma],videoDenoiseMask:mask,
      frozenAudio:true,weights:weight,adapters:{ _ in [] },progress:{ _ in })
    let result=try runner.evaluatePrepared(inputs,sigma:f.sigma,preparation:prepared,
      fixedWeights:weight,
      blockWeights:{ try self.weight("transformer_blocks.\($0)."+$1,$2) },
      fixedAdapters:{ _ in [] },blockAdapters:{ _ in [:] },progress:{ _ in })
    XCTAssertEqual(result["video"]?.size,f.configuration.videoTokens*128)
    XCTAssertEqual(result["audio"]?.size,f.configuration.audioTokens*128)
  }
  func testAudioPerTokenTimestepSharesHeadCacheAndRejectsInvalidMasksBeforeWeights() throws {
    let f=try fixture(),runner=try MLXDenoiser(configuration:f.configuration,blockCount:1)
    let inputs=f.inputs.mapValues { MLXArray($0) }
    let blocks:MLXDenoiser.BlockProvider={ try self.weight("transformer_blocks.\($0)."+$1,$2) }
    let baseline=try runner.evaluate(inputs,sigma:f.sigma,fixedWeights:weight,blockWeights:blocks)
    let allOne=try runner.evaluate(inputs,sigma:f.sigma,audioDenoiseMask:Array(repeating:1,count:f.configuration.audioTokens),
      fixedWeights:weight,blockWeights:blocks)
    for name in ["video","audio"] {
      XCTAssertLessThan((allOne[name]!-baseline[name]!).abs().max().item(Float.self),0.0001)
    }
    let mixed=try runner.evaluate(inputs,sigma:f.sigma,videoDenoiseMask:[0,1,0.5,1,0],
      audioDenoiseMask:[0,0.5,1],fixedWeights:weight,blockWeights:blocks)
    XCTAssertTrue(mixed.values.allSatisfy { MLX.isFinite($0).all().item(Bool.self) })
    XCTAssertGreaterThan((mixed["audio"]!-baseline["audio"]!).abs().max().item(Float.self),0.00001)
    var reads=0
    for mask:[Float] in [[0],[0,1,Float.nan],[0,1,-0.1]] {
      XCTAssertThrowsError(try runner.evaluate(inputs,sigma:f.sigma,audioDenoiseMask:mask,
        fixedWeights:{ name,shape in reads += 1;return try self.weight(name,shape) },blockWeights:blocks))
    }
    XCTAssertEqual(reads,0)
  }
  func testInvalidInputAndObserverFailureDoNotLeakAndAllowRetry() throws {
    enum Stop:Error { case stop }
    let f=try fixture(), runner=try MLXDenoiser(configuration:f.configuration,blockCount:1)
    let inputs=f.inputs.mapValues { MLXArray($0) }
    var calls=0
    for sigma:Float in [-1,1.0001,2,.nan,.infinity] {
      XCTAssertThrowsError(try runner.evaluate(inputs,sigma:sigma,fixedWeights:{ name,shape in
        calls += 1; return try self.weight(name,shape)
      },blockWeights:{ _,name,shape in calls += 1; return try self.weight(name,shape) }))
    }
    XCTAssertEqual(calls,0)
    XCTAssertThrowsError(try runner.evaluate(inputs,sigma:f.sigma,fixedWeights:weight,
      blockWeights:{ _,_,_ in XCTFail("Reached blocks after observer failed"); throw Stop.stop },progress:{ _ in throw Stop.stop }))
    XCTAssertEqual(runner.residentWeightBytes,0)
    let result=try runner.evaluate(inputs,sigma:f.sigma,fixedWeights:weight,
      blockWeights:{ try self.weight("transformer_blocks.\($0)."+$1,$2) })
    XCTAssertEqual(result["video"]?.size,640)
  }
  func testMaskedDenoiserSnapshotsBeforePreparationCallbacks() throws {
    let f=try fixture(), runner=try MLXDenoiser(configuration:f.configuration,blockCount:1)
    let inputs=f.inputs.mapValues { MLXArray($0) }
    let blocks:(Int,String,[Int]) throws -> MLXWeight = { try self.weight("transformer_blocks.\($0)."+$1,$2) }
    let base=try runner.evaluate(inputs,sigma:f.sigma,videoDenoiseMask:[0,1,0.5,1,0],fixedWeights:weight,blockWeights:blocks)
    let result=try runner.evaluate(inputs,sigma:f.sigma,videoDenoiseMask:[0,1,0.5,1,0],fixedWeights:weight,blockWeights:blocks,
      progress:{ _ in for value in inputs.values { value[.ellipsis] = MLXArray.zeros(value.shape) } })
    for name in ["video","audio"] { XCTAssertEqual(base[name]!.asArray(Float.self),result[name]!.asArray(Float.self)) }
  }
  func testFixedAdaptersAffectOutputAndWrongTargetsFail() throws {
    let f=try fixture(), runner=try MLXDenoiser(configuration:f.configuration,blockCount:1)
    let inputs=f.inputs.mapValues { MLXArray($0) }
    let blocks:(Int,String,[Int]) throws -> MLXWeight = { try self.weight("transformer_blocks.\($0)."+$1,$2) }
    let base=try runner.evaluate(inputs,sigma:f.sigma,fixedWeights:weight,blockWeights:blocks)
    let target="proj_out.weight"
    let down=MLXArray.ones([2,f.configuration.videoDimension])*0.01
    let up=MLXArray.ones([128,2])*0.02
    let factor=try MLXLoRA(down:down,up:up,strength:0.5)
    let adapted=try runner.evaluate(inputs,sigma:f.sigma,fixedWeights:weight,blockWeights:blocks,
      fixedAdapters:{ name in name == target ? [factor] : [] })
    XCTAssertNotEqual(base["video"]!.asArray(Float.self),adapted["video"]!.asArray(Float.self))
    XCTAssertEqual(base["audio"]!.asArray(Float.self),adapted["audio"]!.asArray(Float.self))
    XCTAssertThrowsError(try runner.evaluate(inputs,sigma:f.sigma,fixedWeights:weight,blockWeights:blocks,
      fixedAdapters:{ _ in [factor] }))
    XCTAssertEqual(runner.residentWeightBytes,0)
  }

  @MainActor func testCancellationInsideFixedProviderStopsBeforeNextRead() async throws {
    let canceled=try await Task {
      let f=try fixture(), runner=try MLXDenoiser(configuration:f.configuration,blockCount:1)
      var calls=0
      do {
        _ = try runner.evaluate(f.inputs.mapValues { MLXArray($0) },sigma:f.sigma,
          fixedWeights:{ name,shape in
            calls += 1
            let value=try self.weight(name,shape)
            withUnsafeCurrentTask { $0?.cancel() }
            return value
          },blockWeights:{ try self.weight("transformer_blocks.\($0)."+$1,$2) })
        return false
      } catch is CancellationError { return calls == 1 && runner.residentWeightBytes == 0 }
    }.value
    XCTAssertTrue(canceled)
  }

  func testAllFixedAdaptersMatchDenseFusionIncludingTimestepHeads() throws {
    let f=try fixture(), runner=try MLXDenoiser(configuration:f.configuration,blockCount:1)
    let inputs=f.inputs.mapValues { MLXArray($0) }
    let shapes=DenoiserLayout.weightShapes(f.configuration)
    let blocks:(Int,String,[Int]) throws -> MLXWeight = { try self.weight("transformer_blocks.\($0)."+$1,$2) }
    func factors(_ name:String) throws -> [MLXLoRA] {
      let shape=shapes[name]!
      let down=MLXArray((0..<shape[1]).map { Float($0%7-3)/128 },[1,shape[1]])
      let up=MLXArray((0..<shape[0]).map { Float($0%5-2)/64 },[shape[0],1])
      return try [MLXLoRA(down:down,up:up,strength:0.5),MLXLoRA(down:down,up:up,strength:-0.2)]
    }
    let factorized=try runner.evaluate(inputs,sigma:f.sigma,fixedWeights:weight,blockWeights:blocks,fixedAdapters:factors)
    let fused=try runner.evaluate(inputs,sigma:f.sigma,fixedWeights:{ name,shape in
      let base=try self.weight(name,shape)
      guard name.hasSuffix(".weight") else { return base }
      // Independent CPU outer product for the rank-one synthetic fixture.
      var values=try base.tensor().asArray(Float.self)
      for row in 0..<shape[0] {
        for column in 0..<shape[1] {
          let down=Float(column%7-3)/128, up=Float(row%5-2)/64
          values[row*shape[1]+column] += down*up*0.3
        }
      }
      return try MLXWeight(dense:MLXArray(values,shape))
    },blockWeights:blocks)
    for name in ["video","audio"] {
      let a=factorized[name]!.asArray(Float.self), b=fused[name]!.asArray(Float.self)
      XCTAssertLessThan(zip(a,b).map { abs($0-$1) }.max()!,0.0001)
    }
  }
}
