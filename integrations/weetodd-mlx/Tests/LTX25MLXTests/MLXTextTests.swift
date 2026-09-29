import XCTest
import Foundation
import Darwin
import MLX
import LTX25MLX
import LTX25Text
import LTX25Engine
import TensorIO

final class MLXTextTests:XCTestCase {
  func testInstalledTextStageTiming() throws {
    let env=ProcessInfo.processInfo.environment
    guard let request=env["WEETODD_MLX_TEXT_TIMING_REQUEST"] else { throw XCTSkip("Text timing is opt-in.") }
    let json=try JSONSerialization.jsonObject(with:Data(contentsOf:URL(fileURLWithPath:request))) as! [String:Any]
    let encoder=try MLXTextEncoder(gemmaRoot:URL(fileURLWithPath:json["gemma_root"] as! String),
      connectorURL:URL(fileURLWithPath:json["connector_checkpoint"] as! String),
      nativeWeightLoading:env["WEETODD_MLX_TEXT_NATIVE_WEIGHTS"] == "1")
    for run in 0..<2 {
      Memory.peakMemory=0
      let start=Date(); var last=start, stages:[String:Double]=[:]
      let output=try encoder.encode(prompt:json["prompt"] as! String,progress:{ event in
        let now=Date(); stages[event.stage,default:0] += now.timeIntervalSince(last);last=now
      })
      print("TEXT_TIMING run=\(run) seconds=\(Date().timeIntervalSince(start)) stages=\(stages) mlx_peak=\(Memory.peakMemory)")
      if let destination=env["WEETODD_MLX_TEXT_TIMING_OUTPUT"] {
        try save(arrays:["video":output.video,"audio":output.audio],url:URL(fileURLWithPath:destination+"-\(run).safetensors"))
      }
    }
  }
  func fixture(_ name:String) throws -> SafeTensorFile {
    let url=try XCTUnwrap(Bundle.module.url(forResource:name,withExtension:"safetensors",subdirectory:"Fixtures"))
    return try SafeTensorFile(url:url)
  }
  func testGemmaSlidingAndFullAgainstIndependentMLX() throws {
    for full in [false,true] {
      let file=try fixture(full ? "gemma-full" : "gemma-sliding")
      var c=GemmaLayerConfiguration()
      c.width=8; c.hidden=12; c.heads=4; c.kvHeads=full ? 1 : 2; c.headWidth=4
      c.keyEqualsValue=full; c.rotaryFraction=full ? 0.5 : 1; c.theta=full ? 1e6 : 1e4; c.window=full ? nil : 2
      let output=try MLXGemmaLayer.evaluate(MLXWeight.read(file,"input"),configuration:c) {
        try MLXWeight(file:file,name:$0,shape:$1)
      }
      let actual=output.asArray(Float.self), expected=try file.readFloat32(named:"expected")
      XCTAssertLessThan(zip(actual,expected).map { abs($0-$1) }.max()!,0.0001)
    }
  }
  func testTrainedConnectorAgainstIndependentMLX() throws {
    let file=try fixture("text-connector")
    let output=try MLXTextConnector.evaluate(MLXWeight.read(file,"input"),width:8,heads:2) {
      try MLXWeight(file:file,name:$0,shape:$1)
    }
    let actual=output.asArray(Float.self), expected=try file.readFloat32(named:"expected")
    XCTAssertLessThan(zip(actual,expected).map { abs($0-$1) }.max()!,0.0001)
  }
  func testInterleavedNormalizedStatesHaveFeatureThenLayerOrder() throws {
    let states=[MLXArray([Float(3),4],[1,2]),MLXArray([Float(0),2],[1,2])]
    let result=try MLXTextMath.interleaved(states).asArray(Float.self)
    let expected:[Float]=[3/sqrt(12.5+1e-6),0,4/sqrt(12.5+1e-6),2/sqrt(2+1e-6)]
    for (a,b) in zip(result,expected) { XCTAssertEqual(a,b,accuracy:1e-6) }
    XCTAssertThrowsError(try MLXTextMath.interleaved([]))
  }
  func testMalformedLayerInputFailsBeforeWeights() throws {
    var c=GemmaLayerConfiguration(); c.width=0
    XCTAssertThrowsError(try MLXGemmaLayer.evaluate(.ones([1,8]),configuration:c) { _,_ in
      XCTFail("Invalid configuration reached weights"); throw TextEncodingError.invalid("test")
    })
    XCTAssertThrowsError(try MLXTextConnector.evaluate(.ones([3,8]),width:7,heads:2) { _,_ in
      XCTFail("Invalid connector reached weights"); throw TextEncodingError.invalid("test")
    })
  }
  func testTextMemoryAdmissionIncludesStatesAndLargeProjectionSlabs() throws {
    let small=try MLXTextEncodingPlan(promptTokens:8)
    let large=try MLXTextEncodingPlan(promptTokens:1024)
    XCTAssertGreaterThan(large.ownedBufferBytes,small.ownedBufferBytes)
    XCTAssertEqual(large.interleavedBytes,1024*3840*49*4)
    XCTAssertThrowsError(try MLXTextEncodingPlan(promptTokens:1025))
    XCTAssertThrowsError(try MLXTextEncodingPlan(promptTokens:8,maximumOwnedBufferBytes:1))
  }
  func testInstalledPromptAgainstIndependentTextOracle() throws {
    let env=ProcessInfo.processInfo.environment
    guard let root=env["WEETODD_MLX_GEMMA_ROOT"], let connector=env["WEETODD_MLX_CONNECTOR"],
      let reference=env["WEETODD_MLX_TEXT_REFERENCE"] else { throw XCTSkip("Installed text qualification is opt-in.") }
    let encoder=try MLXTextEncoder(gemmaRoot:URL(fileURLWithPath:root),connectorURL:URL(fileURLWithPath:connector))
    let prompt=try env["WEETODD_MLX_TEXT_PROMPT_FILE"].map { try String(contentsOfFile:$0,encoding:.utf8) } ?? "A red fox."
    var events=0, checkedReentry=false
    XCTAssertThrowsError(try encoder.encode(prompt:prompt,maximumOwnedBufferBytes:1))
    Memory.peakMemory=0
    let started=Date()
    let output=try encoder.encode(prompt:prompt,progress:{ event in
      events += 1
      if !checkedReentry {
        checkedReentry=true
        XCTAssertThrowsError(try encoder.encode(prompt:"Nested call must fail before loading."))
      }
      print("MLX_TEXT_PROGRESS \(event.stage) \(event.completed)/\(event.total)")
    })
    let seconds=Date().timeIntervalSince(started)
    var usage=rusage(); getrusage(RUSAGE_SELF,&usage)
    var info=task_vm_info_data_t()
    var count=mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size/MemoryLayout<integer_t>.size)
    let status=withUnsafeMutablePointer(to:&info) { ptr in
      ptr.withMemoryRebound(to:integer_t.self,capacity:Int(count)) {
        task_info(mach_task_self_,task_flavor_t(TASK_VM_INFO),$0,&count)
      }
    }
    XCTAssertEqual(status,KERN_SUCCESS)
    print("MLX_TEXT_PROCESS peak_footprint_bytes=\(info.ledger_phys_footprint_peak) current_footprint_bytes=\(info.phys_footprint)")
    print("MLX_TEXT_METRICS seconds=\(seconds) peak_mlx_bytes=\(Memory.peakMemory) peak_rss_bytes=\(usage.ru_maxrss) tokens=\(output.tokenIDs.count)")
    XCTAssertEqual(events,66)
    XCTAssertEqual(output.video.shape,[1024,4096]); XCTAssertEqual(output.audio.shape,[1024,2048])
    let file=try SafeTensorFile(url:URL(fileURLWithPath:reference))
    let ids=try file.withTensorBytes(named:"token_ids") { bytes in
      stride(from:0,to:bytes.count,by:4).map { Int(bytes.loadUnaligned(fromByteOffset:$0,as:Int32.self).littleEndian) }
    }
    XCTAssertEqual(ids,output.tokenIDs)
    for (name,array) in [("video",output.video),("audio",output.audio)] {
      let actual=array.asArray(Float.self), expected=try file.readFloat32(named:name)
      let error=zip(actual,expected).map { abs($0-$1) }.max()!
      let squared=zip(actual,expected).reduce(0.0) { $0+pow(Double($1.0)-Double($1.1),2) }
      let norm=expected.reduce(0.0) { $0+Double($1)*Double($1) }
      print("MLX_TEXT_\(name) max_abs=\(error) relative=\(sqrt(squared/norm))")
      XCTAssertLessThan(error,0.001)
      XCTAssertLessThan(sqrt(squared/norm),0.0001)
    }
  }
  @MainActor func testCancellationDuringTextWeightLoadAndRetry() async throws {
    let canceled=try await Task {
      let file=try self.fixture("text-connector")
      var loads=0
      do {
        _=try MLXTextConnector.evaluate(MLXWeight.read(file,"input"),width:8,heads:2) { name,shape in
          loads += 1
          if loads == 2 { withUnsafeCurrentTask { $0?.cancel() } }
          return try MLXWeight(file:file,name:name,shape:shape)
        }
        return false
      } catch is CancellationError { return loads == 2 }
    }.value
    XCTAssertTrue(canceled)
    try testTrainedConnectorAgainstIndependentMLX()
  }

  func testInstalledPromptToSamplerAgainstSavedTextContexts() throws {
    let env=ProcessInfo.processInfo.environment
    guard let root=env["WEETODD_MLX_GEMMA_ROOT"], let transformer=env["WEETODD_MLX_TRANSFORMER_ROOT"],
      let connector=env["WEETODD_MLX_CONNECTOR"], let reference=env["WEETODD_MLX_TEXT_REFERENCE"],
      let fixture=env["WEETODD_MLX_LATENT_FIXTURE"] else { throw XCTSkip("Installed prompt sampling is opt-in.") }
    let c=try AVBlockConfiguration(videoTokens:5,audioTokens:3,textTokens:1024)
    let encoder=try MLXTextEncoder(gemmaRoot:URL(fileURLWithPath:root),connectorURL:URL(fileURLWithPath:connector))
    let weights=try MLXDenoiserWeights(root:URL(fileURLWithPath:transformer),configuration:c)
    let file=try SafeTensorFile(url:URL(fileURLWithPath:fixture))
    var inputs:[String:MLXArray]=[:]
    for name in DenoiserLayout.inputShapes(c).keys where !name.hasSuffix("_text") {
      inputs[name]=try MLXWeight.read(file,name)
    }
    let prompt=try env["WEETODD_MLX_TEXT_PROMPT_FILE"].map { try String(contentsOfFile:$0,encoding:.utf8) } ?? "A red fox."
    var textEvents=0, blocks=0
    let runner=try MLXPromptSamplingRunner(configuration:c,encoder:encoder,textProgress:{ _ in textEvents += 1 })
    let schedule=try SamplingSchedule(sigmas:[0.731,0],eta:0)
    let started=Date()
    let actual=try runner.evaluate(prompt:prompt,inputs:inputs,schedule:schedule,
      fixedWeights:{ name,shape in
        XCTAssertEqual(textEvents,66)
        return try weights.readFixed(name,shape)
      },blockWeights:weights.readBlock,stageProgress:{ _,event in
        if event.stage == "transformer" { blocks += 1 }
      })
    print("MLX_PROMPT_SAMPLING seconds=\(Date().timeIntervalSince(started)) blocks=\(blocks)")
    XCTAssertEqual(blocks,48)
    let oracle=try SafeTensorFile(url:URL(fileURLWithPath:reference))
    inputs["video_text"]=try MLXWeight.read(oracle,"video").reshaped([1024,4096])
    inputs["audio_text"]=try MLXWeight.read(oracle,"audio").reshaped([1024,2048])
    let expected=try MLXSamplingRunner(configuration:c).evaluate(inputs,schedule:schedule,
      fixedWeights:weights.readFixed,blockWeights:weights.readBlock)
    for name in ["video","audio"] {
      let error=zip(actual[name]!.asArray(Float.self),expected[name]!.asArray(Float.self)).map { abs($0-$1) }.max()!
      print("MLX_PROMPT_SAMPLING_\(name) max_abs=\(error)")
      XCTAssertLessThan(error,0.002)
    }
  }

}
