import XCTest
import MLX
@testable import LTX25MLX
import LTX25Engine

final class MLXAVBlockTests: XCTestCase {
  struct Fixture: Decodable {
    let configuration: AVBlockConfiguration
    let inputs: [String:[Float]]
    let expected: [String:[Float]]
  }
  func testCompactAudioModulationsMatchExpandedRowsAlongsideVideo() throws {
    let url=Bundle.module.url(forResource:"block-reference",withExtension:"json",subdirectory:"Fixtures")!
    let f=try JSONDecoder().decode(Fixture.self,from:Data(contentsOf:url))
    let compact=try MLXAVBlock(configuration:f.configuration),expanded=try MLXAVBlock(configuration:f.configuration)
    for block in [compact,expanded] {
      try block.load { name,shape in
        let seed=name.utf8.reduce(0) { $0+Int($1) }
        let norm:Float=name.hasSuffix("q_norm.weight") || name.hasSuffix("k_norm.weight") ? 1 : 0
        return try MLXWeight(dense:MLXArray((0..<shape.reduce(1,*)).map { Float(($0*17+seed)%31-15)/128+norm },shape))
      }
    }
    var a=f.inputs.mapValues { MLXArray($0) },b=a
    for (prefix,tokens,names) in [("video",f.configuration.videoTokens,["video_modulation","video_av_modulation"]),
      ("audio",f.configuration.audioTokens,["audio_modulation","audio_av_modulation"])] {
      let index=MLXArray((0..<tokens).map { Int32($0%2) })
      a[prefix+"_modulation_indices"]=index
      for name in names {
        let width=compact.inputShapes[name]![1]
        let table=broadcast(a[name]!.reshaped([1,width]),to:[2,width])+MLXArray([Float(0),Float(0.08)],[2,1])
        a[name]=table;b[name]=take(table,index,axis:0)
      }
    }
    let actual=try compact.evaluate(a),expected=try expanded.evaluate(b)
    for key in ["video","audio"] {
      XCTAssertLessThan((actual[key]!-expected[key]!).abs().max().item(Float.self),0.00003)
    }
    for badIndex in [Int32(-1),Int32(2)] {
      var bad=a
      bad["audio_modulation_indices"]=broadcast(MLXArray([badIndex]),to:[f.configuration.audioTokens])
      XCTAssertThrowsError(try compact.validateInputs(bad))
    }
  }
  func testCompressedModulationsMatchExpandedWithChangingIndicesAndTables() throws {
    let url=Bundle.module.url(forResource:"block-reference",withExtension:"json",subdirectory:"Fixtures")!
    let f=try JSONDecoder().decode(Fixture.self,from:Data(contentsOf:url))
    let compact=try MLXAVBlock(configuration:f.configuration),expanded=try MLXAVBlock(configuration:f.configuration)
    for block in [compact,expanded] {
      try block.load { name,shape in
        let seed=name.utf8.reduce(0) { $0+Int($1) }
        let norm:Float=name.hasSuffix("q_norm.weight") || name.hasSuffix("k_norm.weight") ? 1 : 0
        return try MLXWeight(dense:MLXArray((0..<shape.reduce(1,*)).map { Float(($0*17+seed)%31-15)/128+norm },shape))
      }
    }
    for iteration in 0..<3 {
      var a=f.inputs.mapValues { MLXArray($0) },b=a
      let index=MLXArray((0..<f.configuration.videoTokens).map { Int32(($0+iteration)%2) })
      a["video_modulation_indices"]=index
      for name in ["video_modulation","video_av_modulation"] {
        let width=compact.inputShapes[name]![1]
        let table=broadcast(a[name]!.reshaped([1,width]),to:[2,width])+MLXArray([Float(iteration)*0.01,0.08],[2,1])
        a[name]=table;b[name]=take(table,index,axis:0)
      }
      let actual=try compact.evaluate(a),expected=try expanded.evaluate(b)
      for key in ["video","audio"] { XCTAssertLessThan((actual[key]!-expected[key]!).abs().max().item(Float.self),0.00003) }
      for badIndex in [MLXArray([Int32(-1)]),MLXArray([Int32(2)])] {
        var bad=a;bad["video_modulation_indices"]=broadcast(badIndex,to:[f.configuration.videoTokens])
        XCTAssertThrowsError(try compact.validateInputs(bad))
      }
    }
    XCTAssertEqual(compact.compiledGraphBuildCount,1)
    // Switching an instance back to expanded inputs must rebuild the argument layout.
    _ = try compact.evaluate(f.inputs.mapValues { MLXArray($0) })
    XCTAssertEqual(compact.compiledGraphBuildCount,2)
  }
  func testCompiledGraphUsesCurrentWeightsPromptsTokenModulationAndOrderedAdapters() throws {
    let url=Bundle.module.url(forResource:"block-reference",withExtension:"json",subdirectory:"Fixtures")!
    let f=try JSONDecoder().decode(Fixture.self,from:Data(contentsOf:url))
    weak var lifetime:MLXAVBlock?
    try autoreleasepool {
      let compiled=try MLXAVBlock(configuration:f.configuration,compileGraph:true)
      let eager=try MLXAVBlock(configuration:f.configuration,compileGraph:false)
      lifetime=compiled
      var prior:[Float]?
      for iteration in 0..<4 {
        for block in [compiled,eager] {
          try block.load { name,shape in
            let seed=name.utf8.reduce(0) { $0+Int($1) }+iteration
            let norm:Float=name.hasSuffix("q_norm.weight") || name.hasSuffix("k_norm.weight") ? 1 : 0
            return try MLXWeight(dense:MLXArray((0..<shape.reduce(1,*)).map { Float(($0*17+seed)%31-15)/128+norm },shape))
          }
          let shape=block.weightShapes["ff.proj_out.weight"]!
          let first=try MLXLoRA(down:.ones([2,shape[1]])*Float(0.03+Double(iteration)*0.01),up:.ones([shape[0],2])*Float(0.01),strength:Float(iteration+1)*0.2)
          let second=try MLXLoRA(down:.ones([1,shape[1]])*Float(-0.04),up:.ones([shape[0],1])*Float(0.02),strength:-0.3)
          try block.setAdapters(["ff.proj_out.weight":iteration%2 == 0 ? [first,second] : [second,first]])
        }
        var inputs=f.inputs.mapValues { MLXArray($0) }
        inputs["video_text"]=inputs["video_text"]!+Float(iteration)*0.1
        for name in ["video_modulation","video_av_modulation"] {
          let width=compiled.inputShapes[name]![1]
          inputs[name]=broadcast(inputs[name]!.reshaped([1,width]),to:[f.configuration.videoTokens,width])+MLXArray((0..<f.configuration.videoTokens).map { Float($0+iteration)*0.01 },[f.configuration.videoTokens,1])
        }
        let actual=try compiled.evaluate(inputs),expected=try eager.evaluate(inputs)
        for key in ["video","audio"] {
          XCTAssertLessThan((actual[key]!-expected[key]!).abs().max().item(Float.self),0.00003)
        }
        let values=actual["video"]!.asArray(Float.self)
        if let prior { XCTAssertGreaterThan(zip(values,prior).map { abs($0-$1) }.max()!,0.00001) }
        prior=values
        compiled.release();eager.release()
        XCTAssertEqual(compiled.loadedBytes,0)
      }
      // The argument layout remains reusable across different blocks/adapters;
      // MLX internally specializes tensor shapes without capturing their data.
      XCTAssertEqual(compiled.compiledGraphBuildCount,1)
    }
    XCTAssertNil(lifetime,"A compiled graph must not retain its owning block.")
  }
  func testJointBlockMatchesIndependentPythonMLXFixtureAndRelease() throws {
    let url = Bundle.module.url(forResource:"block-reference",withExtension:"json",subdirectory:"Fixtures")!
    let f = try JSONDecoder().decode(Fixture.self,from:Data(contentsOf:url))
    let block = try MLXAVBlock(configuration:f.configuration)
    try block.load { name,shape in
      let seed = name.utf8.reduce(0) { $0+Int($1) }
      let norm: Float = name.hasSuffix("q_norm.weight") || name.hasSuffix("k_norm.weight") ? 1 : 0
      let values = (0..<shape.reduce(1,*)).map { Float(($0*17+seed)%31-15)/128+norm }
      return try MLXWeight(dense:MLXArray(values,shape))
    }
    let inputs = f.inputs.mapValues { MLXArray($0) }
    let output = try block.evaluate(inputs)
    for name in ["video","audio"] {
      let actual = try XCTUnwrap(output[name]).asArray(Float.self)
      let expected = try XCTUnwrap(f.expected[name])
      XCTAssertEqual(actual.count,expected.count)
      XCTAssertLessThan(zip(actual,expected).map { abs($0-$1) }.max()!,0.00003,name)
    }
    block.release()
    XCTAssertEqual(block.loadedBytes,0)
    XCTAssertThrowsError(try block.evaluate(inputs))
    XCTAssertThrowsError(try block.load { _,_ in throw LTXError.invalid("load failed") })
    XCTAssertThrowsError(try block.evaluate(inputs))
  }

  func testMalformedInputsFailBeforeExecution() throws {
    let c = try AVBlockConfiguration(videoTokens:5,audioTokens:3,textTokens:4)
    let block = try MLXAVBlock(configuration:c)
    XCTAssertThrowsError(try block.evaluate([:]))
  }
  func testNonFusedAttentionCannotBypassMemoryAdmissionWithSmallWidth() throws {
    let c=try AVBlockConfiguration(videoDimension:32,audioDimension:16,heads:2,
      videoHeadDimension:16,audioHeadDimension:8,videoTokens:65536,audioTokens:3,textTokens:4)
    XCTAssertThrowsError(try MLXAVBlock(configuration:c))
  }
  func testNonzeroFactorizedAdapterMatchesIndependentDenseBlockFusion() throws {
    let url=Bundle.module.url(forResource:"lora-block-reference",withExtension:"json",subdirectory:"Fixtures")!
    let f=try JSONDecoder().decode(Fixture.self,from:Data(contentsOf:url))
    let block=try MLXAVBlock(configuration:f.configuration)
    try block.load { name,shape in
      let seed=name.utf8.reduce(0) { $0+Int($1) }
      let norm:Float=name.hasSuffix("q_norm.weight") || name.hasSuffix("k_norm.weight") ? 1 : 0
      return try MLXWeight(dense:MLXArray((0..<shape.reduce(1,*)).map { Float(($0*17+seed)%31-15)/128+norm },shape))
    }
    let down=MLXArray((0..<256).map { Float($0%7-3)/32 },[2,128])
    let up=MLXArray((0..<64).map { Float($0%5-2)/64 },[32,2])
    try block.setAdapters(["ff.proj_out.weight":[MLXLoRA(down:down,up:up,strength:0.25)]])
    let result=try block.evaluate(f.inputs.mapValues { MLXArray($0) })
    for name in ["video","audio"] {
      let actual=result[name]!.asArray(Float.self), expected=f.expected[name]!
      XCTAssertLessThan(zip(actual,expected).map { abs($0-$1) }.max()!,0.00003)
    }
    block.release()
  }
  @MainActor func testCancellationDuringWeightLoadClearsPartialBlock() async throws {
    let canceled=try await Task {
      let c=try AVBlockConfiguration(videoDimension:32,audioDimension:16,heads:2,
        videoHeadDimension:16,audioHeadDimension:8,videoTokens:5,audioTokens:3,textTokens:4)
      let block=try MLXAVBlock(configuration:c)
      var calls=0
      do {
        try block.load { _,shape in
          calls += 1
          if calls == 10 { withUnsafeCurrentTask { $0?.cancel() } }
          return try MLXWeight(dense:.zeros(shape))
        }
        return false
      } catch is CancellationError { return block.loadedBytes == 0 && calls == 10 }
    }.value
    XCTAssertTrue(canceled)
  }
}
