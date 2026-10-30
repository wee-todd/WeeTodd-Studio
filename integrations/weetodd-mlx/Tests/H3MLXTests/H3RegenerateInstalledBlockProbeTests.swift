import Foundation
import CryptoKit
import Darwin
import InferenceContracts
import MLX
import MLXRandom
import XCTest
@testable import H3MLX

final class H3RegenerateInstalledBlockProbeTests:XCTestCase {
  func testInstalledFull2KBlockColdAndWarm() throws {
    guard let fixture=ProcessInfo.processInfo.environment["WEETODD_H3_REGENERATE_BLOCK_JOB"],
      let log=ProcessInfo.processInfo.environment["WEETODD_H3_REGENERATE_BLOCK_LOG"],
      let worker=ProcessInfo.processInfo.environment["WEETODD_H3_REGENERATE_BLOCK_WORKER"] else {
      throw XCTSkip("Explicit installed full2K block probe requires a frozen job and external JSONL log.")
    }
    let environment=ProcessInfo.processInfo.environment
    let iterations=Int(environment["WEETODD_H3_REGENERATE_BLOCK_ITERATIONS"] ?? "2") ?? 0
    let activationOutput=environment["WEETODD_H3_REGENERATE_BLOCK_OUTPUT"]
    let referenceOutput=environment["WEETODD_H3_REGENERATE_BLOCK_REFERENCE"]
    guard (1...2).contains(iterations),activationOutput == nil || iterations == 1,
      referenceOutput == nil || iterations == 1 else {
      throw H3CheckpointError.invalid("Block activation comparison requires exactly one bounded block.")
    }
    _ = worker // Probe context verifies and binds the explicitly frozen library.
    try H3BoundedProbeGPUContext.run { error in
    let job=try H3RegenerateJob.decode(Data(contentsOf:URL(fileURLWithPath:fixture)))
    let tokenizer=try H3QwenTokenizer(url:URL(fileURLWithPath:job.qwen.tokenizer.path))
    let request=try H3QwenRequest.text(job.prompt,tokenizer:tokenizer)
    let admission=try H3RegenerateAdmission(width:2560,height:1440,visibleFrames:121,paddedFrames:124,fps:24,
      textRows:request.tags.count,maximumStageBytes:UInt64(H3BoundedProbeGPUContext.activeLimit),availableTemporaryBytes:64*1024*1024*1024)
    try admission.validateExecution(qualificationProbe:true)
    for component in [job.components.transformer,job.components.refiner,job.components.sharpInput,job.components.sharpCurve] {
      let handle=try FileHandle(forReadingFrom:URL(fileURLWithPath:component.path));defer{try? handle.close()}
      var hash=SHA256()
      while try autoreleasepool(invoking:{ () throws -> Bool in
        guard let data=try handle.read(upToCount:4*1024*1024),!data.isEmpty else { return false }
        hash.update(data:data);return true
      }) {}
      guard hash.finalize().map({String(format:"%02x",$0)}).joined() == component.identity.sourceSHA256 else {
        throw H3CheckpointError.invalid("Probe component differs from the frozen whole-file identity.")
      }
    }
    let layout=try H3PackedLayout(geometry:admission.geometry,textTags:request.tags,anchors:[])
    let attentionPlan=try H3RegenerateAttentionPlan.runtime(rows:admission.packedRows,heads:56,headWidth:128,dtype:.bfloat16)
    let clock=try H3RowSchedule(regenerate:layout,schedule:.publisherThreeStep)
    let base=URL(fileURLWithPath:job.components.transformer.path)
    let adapter=try H3RegenerateAdapter(url:URL(fileURLWithPath:job.components.refiner.path),profile:.publisherThreeStep,companions:[.sharpCurve])
    let sharp=try H3SharpCurve(tableURL:URL(fileURLWithPath:job.components.sharpInput.path),branchesURL:URL(fileURLWithPath:job.components.sharpCurve.path))
    defer { adapter.close();sharp.close();Stream.gpu.synchronize();Memory.clearCache() }
    let time=try H3TimeEmbedding.evaluate(checkpointURL:base,timesteps:MLXArray(clock.table))
    let modulation=try H3AdaLNProjection.evaluate(checkpointURL:base,blockIndex:0,timeEmbeddings:time)
    let positions=MLXArray(layout.positions.flatMap{[$0.x,$0.y,$0.z]},[admission.packedRows,3])
    let indices=MLXArray(zip(clock.indicesByStep[0],layout.tags).map{$0*3+Swift.max(0,$1)})
    let angles=try H3TransformerBlock.prepareRotaryAngles(checkpointURL:base,positions:positions,maximumRows:H3RegenerateAdmission.maximumPackedRows)
    let input=(MLXRandom.normal([1,admission.packedRows,5376],key:MLXRandom.key(42))*0.05).asType(.bfloat16)
    eval(input,modulation,positions,indices)
    let outputURL=URL(fileURLWithPath:log)
    XCTAssertFalse(FileManager.default.fileExists(atPath:log))
    guard FileManager.default.createFile(atPath:log,contents:nil) else { throw H3CheckpointError.invalid("Cannot create probe log.") }
    let handle=try FileHandle(forWritingTo:outputURL);defer{try? handle.close()}
    func record(_ object:[String:Any]) throws {
      var data=try JSONSerialization.data(withJSONObject:object,options:.sortedKeys);data.append(10)
      try handle.write(contentsOf:data);try handle.synchronize()
    }
    try record(["event":"begin","packedRows":admission.packedRows,"latentFrames":37,"width":2560,"height":1440,
      "input":"seeded-synthetic-hidden-activation","sourceSHA256":job.components.transformer.identity.sourceSHA256,
      "fullModelExecution":false,"allKeysVisible":true,"attentionExecution":attentionPlan.metadata])
    for iteration in 0..<iterations {
      // A fresh stage also redoes all merges, matching a new production block.
      try H3BoundedProbeGPUContext.checkBudget(error)
      let start=Date();Memory.peakMemory = Memory.activeMemory
      try adapter.withStage(.regenerate) {
        let result=try H3TransformerBlock.evaluate(checkpointURL:base,index:0,input:input,modulation:modulation,
          modulationIndices:indices,positions:positions,lora:adapter,sharp:sharp,modelTimes:clock.table,
          regenerateAdmission:admission,regenerateAttentionPlan:attentionPlan,rotaryAngles:angles,maximumRows:H3RegenerateAdmission.maximumPackedRows,
          observe:{ name,value in
            eval(value);try H3BoundedProbeGPUContext.checkBudget(error)
            try record(["event":"phase","iteration":iteration,"phase":name,"seconds":Date().timeIntervalSince(start),"activeBytes":Memory.activeMemory,"peakBytes":Memory.peakMemory])
          })
        eval(result);try H3BoundedProbeGPUContext.checkBudget(error);XCTAssertTrue(all(isFinite(result)).item(Bool.self))
        var usage=rusage();getrusage(RUSAGE_SELF,&usage)
        try record(["event":"complete","iteration":iteration,"seconds":Date().timeIntervalSince(start),
          "peakBytes":Memory.peakMemory,"processMaximumResidentBytes":usage.ru_maxrss,"outputShape":result.shape])
        if let activationOutput {
          guard activationOutput.hasPrefix("/"),!FileManager.default.fileExists(atPath:activationOutput),
            result.dtype == .bfloat16 else { throw H3CheckpointError.invalid("Block activation output must be a new absolute BF16 artifact.") }
          XCTAssertTrue(FileManager.default.createFile(atPath:activationOutput,contents:nil))
          let destination=try FileHandle(forWritingTo:URL(fileURLWithPath:activationOutput))
          defer { try? destination.close() }
          let reference=try referenceOutput.map { try FileHandle(forReadingFrom:URL(fileURLWithPath:$0)) }
          defer { try? reference?.close() }
          if let reference {
            XCTAssertEqual(try reference.seekToEnd(),UInt64(result.size*2));try reference.seek(toOffset:0)
          }
          var hash=SHA256(),referenceHash=SHA256()
          var maximum:Float=0,squaredError:Double=0,squaredReference:Double=0
          var identical=true
          for first in stride(from:0,to:admission.packedRows,by:8192) {
            try autoreleasepool {
              let rows=first..<min(first+8192,admission.packedRows)
              let actual=result[0..<1,rows,0..<5376]
              let data=try H3RegenerateArtifact.copyData(actual)
              try destination.write(contentsOf:data);hash.update(data:data)
              if let reference {
                let expectedBytes=try XCTUnwrap(reference.read(upToCount:data.count))
                XCTAssertEqual(expectedBytes.count,data.count);referenceHash.update(data:expectedBytes)
                let expected=expectedBytes.withUnsafeBytes {
                  MLXArray($0,[1,rows.count,5376],type:UInt16.self).view(dtype:.bfloat16)
                }
                let expected32=expected.asType(.float32),delta=actual.asType(.float32)-expected32
                maximum=max(maximum,MLX.max(abs(delta)).item(Float.self))
                squaredError += Double(sum(delta*delta).item(Float.self))
                squaredReference += Double(sum(expected32*expected32).item(Float.self))
                identical = identical && data == expectedBytes
                XCTAssertTrue(all(isFinite(expected)).item(Bool.self))
              }
            }
          }
          try destination.synchronize()
          var artifact:[String:Any]=["event":"activation","dtype":"BF16LE","shape":result.shape,
            "output":activationOutput,"sha256":hash.finalize().map { String(format:"%02x",$0) }.joined()]
          if reference != nil {
            let relative=(squaredError/max(squaredReference,Double.leastNonzeroMagnitude)).squareRoot()
            artifact["referenceSHA256"]=referenceHash.finalize().map { String(format:"%02x",$0) }.joined()
            artifact["maximumAbsoluteDifference"]=maximum;artifact["relativeL2"]=relative
            artifact["bitIdentical"]=identical
            XCTAssertTrue(identical);XCTAssertEqual(maximum,0);XCTAssertEqual(relative,0)
          }
          try record(artifact)
        }
      }
      Stream.gpu.synchronize();Memory.clearCache()
      XCTAssertEqual(adapter.mergedWeights.count,0)
      try record(["event":"weights_released","iteration":iteration,"mergedProjectionCount":adapter.mergedWeights.count,
        "activeMLXBytes":Memory.activeMemory])
    }
    try error.check()
    }
  }
}
