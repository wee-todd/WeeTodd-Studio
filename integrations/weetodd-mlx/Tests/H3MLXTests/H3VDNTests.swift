import Foundation
import MLX
import XCTest
@testable import H3MLX

final class H3VDNTests: XCTestCase {
  func testWindowsKeepGlobalRowsAndBothAnchorsWithoutDuplicatingLocalKeys() throws {
    let layout = try H3VDNLayout(sequence: 47, videoStart: 5, frames: 21,
      height: 1, width: 2, textStart: 0, textLength: 3)
    let groups = layout.attentionGroups
    XCTAssertEqual(groups.first?.query, 0..<5)
    XCTAssertEqual(groups[1].query, 5..<7)
    XCTAssertEqual(groups[1].keys, [0..<47])
    // Frames 10...14 see frames 5...19, plus anchors 0 and 20.
    let middle = try XCTUnwrap(groups.first { $0.query == 25..<35 })
    XCTAssertEqual(middle.keys, [0..<5,15..<45,5..<7,45..<47])
    let visited = groups.flatMap { Array($0.query) }
    XCTAssertEqual(visited, Array(0..<47))
    for group in groups {
      let keys = group.keys.flatMap { Array($0) }
      XCTAssertEqual(Set(keys).count, keys.count)
    }
  }

  func testLayoutRejectsOverflowOverlappingTextAndEmptyGrids() {
    for values in [(10,2,Int.max,2,2,0,1),(10,2,2,1,2,3,1),
      (10,2,0,1,2,0,1),(10,2,2,0,2,0,1)] {
      XCTAssertThrowsError(try H3VDNLayout(sequence:values.0,videoStart:values.1,
        frames:values.2,height:values.3,width:values.4,textStart:values.5,textLength:values.6))
    }
    XCTAssertThrowsError(try H3VDNLayout(sequence:110,videoStart:2,frames:108,
      height:1,width:1,textStart:0,textLength:1))
  }

  func testInversePreservesCorrelatedFloat32Systems() throws {
    let matrix = MLXArray([Float(2),1,1,3],[1,2,2])
    let inverse = try H3VDNMath.inverse(matrix)
    let actual = inverse.asArray(Float.self)
    for (a,b) in zip(actual,[Float(0.6),-0.2,-0.2,0.4]) {
      XCTAssertEqual(a,b,accuracy:2e-6)
    }
    let vector = MLXArray((0..<128).map { Float($0 % 7 - 3)/5 })
    let wide = MLX.eye(128) + vector.expandedDimensions(axis:1)*vector.expandedDimensions(axis:0)
    let result = try H3VDNMath.inverse(wide)
    XCTAssertLessThan(max(abs(matmul(wide,result)-MLX.eye(128))).item(Float.self),2e-4)
    XCTAssertTrue(MLX.isFinite(result).all().item(Bool.self))
    XCTAssertThrowsError(try H3VDNMath.inverse(MLXArray.ones([2,3],dtype:.float32)))
    XCTAssertThrowsError(try H3VDNMath.inverse(MLX.eye(2).asType(.bfloat16)))
    XCTAssertThrowsError(try H3VDNMath.inverse(MLXArray([Float(-1)],[1,1])))
  }

  func testTemporalFilterUsesZeroPaddingAndTapOrder() throws {
    let input = MLXArray([Float(1),2,3],[3,1,1,1]).asType(.bfloat16)
    let taps = MLXArray([Float(1),2,3,4,5],[1,1,5]).asType(.bfloat16)
    let output = try H3VDNMath.temporal(input,weights:taps)
    XCTAssertEqual(output.asType(.float32).asArray(Float.self),[26,20,14])
    XCTAssertThrowsError(try H3VDNMath.temporal(input,weights:MLXArray.ones([1,1,3])))
  }

  func testRetentionNeverRoundsFrameMeanOrProjectionOperandsToBF16() throws {
    // Halfway between two BF16 values. Promoting after the projection would
    // lose this mean and perturb every prefix/suffix retention coefficient.
    let result=try H3VDNMath.retention(frameMeans:MLXArray([Float(1.00390625)],[1,1]),
      down:MLXArray.ones([1,1],dtype:.bfloat16),up:MLXArray.ones([1,1],dtype:.bfloat16),
      bias:MLXArray.zeros([1],dtype:.bfloat16),logScale:MLXArray.zeros([1],dtype:.bfloat16),heads:1,dim:1)
    let expected=Float(1)/(1+exp(Float(1.00390625)))
    XCTAssertEqual(result.item(Float.self),expected,accuracy:1e-7)
    XCTAssertEqual(result.dtype,.float32)
  }

  func testBidirectionalScanAndExcludedLocalWindowUseIndependentBoundaryValues() throws {
    let transitions = MLXArray([Float(0.5),0.25,0.1],[3,1,1,1])
    let injections = MLXArray([Float(1),2,3],[3,1,1,1])
    let start = MLXArray([Float(2)],[1,1,1])
    let scan = try H3VDNMath.scan(transitions:transitions,injections:injections,start:start)
    XCTAssertEqual(scan.prefix.asArray(Float.self),[2,2.5,3.25])
    let expected:[Float] = [2.4,2.8,3.2]
    for (a,b) in zip(scan.suffix.asArray(Float.self),expected) { XCTAssertEqual(a,b,accuracy:1e-6) }
    // A window covering all three frames has only the two virtual text states.
    let state = try H3VDNMath.gather(prefix:scan.prefix,suffix:scan.suffix,
      alpha:transitions.reshaped([3,1,1]),text:start,bounds:[(-1,4),(-1,4),(-1,4)])
    for (a,b) in zip(state.asArray(Float.self),[Float(1.025),0.30,0.225]) {
      XCTAssertEqual(a,b,accuracy:2e-6)
    }
  }

  func testInstalledBranchAdmissionChecksAllFiftyBlocks() throws {
    guard let value=ProcessInfo.processInfo.environment["WEETODD_H3_VDN_STAGE"] else {
      throw XCTSkip("Opt-in installed VDN branch header qualification")
    }
    let root=URL(fileURLWithPath:value)
    let checkpoint=try H3VDNCheckpoint(stage:root)
    XCTAssertEqual(checkpoint.tensorCount,800)
    let weights=try checkpoint.readBlock(49)
    XCTAssertEqual(weights.count,16)
    XCTAssertEqual(weights["to_out_linear.weight"]?.shape,[5376,7168])
  }

  func testAdmissionRejectsMissingOrMisshapedFinalBlockAndUnsupportedSpec() throws {
    let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at:root.appendingPathComponent("linear_branch"),withIntermediateDirectories:true)
    defer { try? FileManager.default.removeItem(at:root) }
    let linear:[String:Any] = ["delta_rule":"vdn_solve","linear_head_dim":128,
      "enable_text_state":true,"bridge":"alpha","a_fp32":true,"short_conv":["targets":["k","v"]]]
    var config:[String:Any] = ["anchor_frames":"both","enable_softmax_gate":true,
      "linear_attention":linear,"softmax_attention":["chunk":5,"radius":1]]
    func spec() throws {
      let payload:[String:Any] = ["format_version":2,"base":["class_name":"MiniMaxH3Transformer3DModel"],
        "transforms":[["type":"hybrid_attention","version":2,"config":config]]]
      try JSONSerialization.data(withJSONObject:payload).write(to:root.appendingPathComponent("model_spec.json"))
    }
    func branch(_ kind:Int) throws {
      var header:[String:Any]=[:];var offset:UInt64=0
      for block in 0..<50 {
        for (suffix,original) in H3VDNCheckpoint.blockShapes {
          if kind == 1 && block == 49 && suffix == "to_out_linear.weight" { continue }
          var shape=original
          if kind == 2 && block == 49 && suffix == "to_out_linear.weight" { shape[0] -= 1 }
          let size=UInt64(shape.reduce(1,*))*2
          header["transformer_blocks.\(block).attn."+suffix]=["dtype":"BF16","shape":shape,"data_offsets":[offset,offset+size]]
          offset += size
        }
      }
      let body=try JSONSerialization.data(withJSONObject:header)
      var length=UInt64(body.count).littleEndian
      var data=withUnsafeBytes(of:&length) { Data($0) };data.append(body)
      let url=root.appendingPathComponent("linear_branch/model.safetensors")
      try data.write(to:url)
      let file=try FileHandle(forWritingTo:url);defer { try? file.close() }
      try file.truncate(atOffset:UInt64(data.count)+offset)
    }
    try spec();try branch(0)
    XCTAssertEqual(try H3VDNCheckpoint(stage:root).tensorCount,800)
    try branch(1);XCTAssertThrowsError(try H3VDNCheckpoint(stage:root))
    try branch(2);XCTAssertThrowsError(try H3VDNCheckpoint(stage:root))
    try branch(0);config["anchor_frames"]="none";try spec()
    XCTAssertThrowsError(try H3VDNCheckpoint(stage:root))
  }

  func testCancelledSolveStopsBeforeMetalDispatch() async {
    let task=Task {
      withUnsafeCurrentTask { $0?.cancel() }
      _ = try H3VDNMath.inverse(MLX.eye(2))
    }
    do { _ = try await task.value;XCTFail("Cancelled solve unexpectedly executed") }
    catch { XCTAssertTrue(error is CancellationError) }
  }

  func testCompleteHybridCoreMatchesIndependentBF16AndFloat32Reference() throws {
    let url=try XCTUnwrap(Bundle.module.url(forResource:"vdn-core-reference",withExtension:"json",subdirectory:"Fixtures"))
    let fixture=try JSONSerialization.jsonObject(with:Data(contentsOf:url)) as! [String:Any]
    func array(_ value:Any) -> MLXArray {
      let item=value as! [String:Any]
      return MLXArray((item["values"] as! [NSNumber]).map(\.floatValue),(item["shape"] as! [Int]))
        .asType(.bfloat16)
    }
    let inputs=(fixture["inputs"] as! [String:Any]).mapValues(array)
    let weights=(fixture["weights"] as! [String:Any]).mapValues(array)
    let layout=try H3VDNLayout(sequence:19,videoStart:5,frames:7,
      height:1,width:2,textStart:0,textLength:3)
    let output=try H3VDNAttention.evaluate(input:inputs["x"]!,
      raw:[inputs["q_raw"]!,inputs["k_raw"]!,inputs["v_raw"]!],
      query:inputs["query"]!,key:inputs["key"]!,value:inputs["value"]!,
      layout:layout,weights:weights,projectSoftmax:{ matmul($0,inputs["projection"]!.T) })
    let expected=array(fixture["expected"]!)
    XCTAssertEqual(output.shape,[1,19,8])
    let difference=output.asType(.float32)-expected.asType(.float32)
    XCTAssertLessThan(max(abs(difference)).item(Float.self),0.008)
    XCTAssertLessThan(sqrt(mean(difference*difference)/mean(expected.asType(.float32)*expected.asType(.float32))).item(Float.self),0.01)
    XCTAssertTrue(MLX.isFinite(output).all().item(Bool.self))
  }

  func testInstalledFullWidthBranchMatchesSeparateReferenceWithoutRetainingAllWeights() throws {
    let env=ProcessInfo.processInfo.environment
    guard let stage=env["WEETODD_H3_VDN_STAGE"],let fixtureRoot=env["WEETODD_H3_VDN_ORACLE"] else {
      throw XCTSkip("Opt-in independently exported installed VDN core witness")
    }
    let checkpoint=try H3VDNCheckpoint(stage:URL(fileURLWithPath:stage))
    for block in [0,49] {
      let input=try loadArrays(url:URL(fileURLWithPath:fixtureRoot).appendingPathComponent("block-\(block).safetensors"))
      let layout=try H3VDNLayout(sequence:26,videoStart:5,frames:21,height:1,width:1,textStart:0,textLength:3)
      let started=Date();Memory.peakMemory=Memory.activeMemory
      let result=try H3VDNAttention.evaluate(input:input["x"]!,
        raw:[input["q_raw"]!,input["k_raw"]!,input["v_raw"]!],
        query:input["q_raw"]!.transposed(0,2,1,3),
        key:input["k_raw"]!.transposed(0,2,1,3),value:input["v_raw"]!.transposed(0,2,1,3),
        layout:layout,weights:try checkpoint.readBlock(block),
        projectSoftmax:{ $0[0..<1,0..<26,0..<5376] })
      let expected=input["expected"]!.asType(.float32)
      let difference=result.asType(.float32)-expected
      let relative=sqrt(mean(difference*difference)/mean(expected*expected)).item(Float.self)
      let maximum=max(abs(difference)).item(Float.self)
      XCTAssertTrue(MLX.isFinite(result).all().item(Bool.self))
      XCTAssertLessThan(relative,0.015)
      print("VDN_INSTALLED_CORE block=\(block) relative=\(relative) max=\(maximum) seconds=\(Date().timeIntervalSince(started)) peakMLXBytes=\(Memory.peakMemory)")
    }
  }
}
