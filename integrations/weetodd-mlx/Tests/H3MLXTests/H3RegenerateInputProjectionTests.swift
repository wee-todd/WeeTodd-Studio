import Foundation
import InferenceContracts
import MLX
import XCTest
@testable import H3MLX

final class H3RegenerateInputProjectionTests:XCTestCase {
  func testFullGridWindowsBoundIntermediatesAndOrdinaryCapRemains() throws {
    let windows=try H3InputProjection.regenerateWindows(rows:133200)
    XCTAssertEqual(windows.first,0..<8192);XCTAssertEqual(windows.last?.upperBound,133200)
    XCTAssertEqual(windows.reduce(0){$0+$1.count},133200)
    XCTAssertTrue(windows.allSatisfy{$0.count<=8192})
    XCTAssertThrowsError(try H3InputProjection.regenerateWindows(rows:390001))
    XCTAssertThrowsError(try H3InputProjection.regenerateWindows(rows:10,chunkRows:8193))
    XCTAssertThrowsError(try H3InputProjection.evaluate(checkpointURL:URL(fileURLWithPath:"/absent"),kind:.video,input:MLXArray.zeros([1,65537,96])))
  }
  func testChunkedFP32ProjectionThenBF16CastMatchesFullProjection() throws {
    let input=MLXArray((0..<7*96).map{Float($0%31)/31},[1,7,96])
    let weight=MLXArray((0..<5376*96).map{Float(($0%11)-5)/100},[5376,96]),bias=MLXArray.ones([5376])*0.125
    func project(_ rows:MLXArray)->MLXArray { addMM(bias,rows,weight.T) }
    let expected=project(input).asType(.bfloat16)
    let actual=try H3InputProjection.projectRegenerateRows(kind:.video,input:input,chunkRows:2,project:project)
    XCTAssertEqual(actual.shape,[1,7,5376]);XCTAssertEqual(actual.dtype,.bfloat16)
    XCTAssertLessThanOrEqual(max(abs(actual.asType(.float32)-expected.asType(.float32))).item(Float.self),0.00390625)
  }
  func testInstalledFullGridInputPackingAndFinalHeads() throws {
    guard let fixture=ProcessInfo.processInfo.environment["WEETODD_H3_REGENERATE_BOUNDARY_JOB"],
      let worker=ProcessInfo.processInfo.environment["WEETODD_H3_REGENERATE_BLOCK_WORKER"] else {throw XCTSkip("Explicit installed full-grid boundary probe")}
    GPU.metallib=try NativeMLXMetalLibrary.resolve(executable:URL(fileURLWithPath:worker))
    let job=try H3RegenerateJob.decode(Data(contentsOf:URL(fileURLWithPath:fixture))),base=URL(fileURLWithPath:job.components.transformer.path)
    let tokenizer=try H3QwenTokenizer(url:URL(fileURLWithPath:job.qwen.tokenizer.path)),request=try H3QwenRequest.text(job.prompt,tokenizer:tokenizer)
    let admission=try H3RegenerateAdmission(width:2560,height:1440,visibleFrames:120,paddedFrames:124,fps:24,textRows:request.tags.count,maximumStageBytes:128<<30,availableTemporaryBytes:128<<30)
    let layout=try H3PackedLayout(geometry:admission.geometry,textTags:request.tags,anchors:[]),clock=try H3RowSchedule(regenerate:layout,schedule:.publisherThreeStep)
    let started=Date();Memory.peakMemory=Memory.activeMemory
    defer{Stream.gpu.synchronize();Memory.clearCache()}
    let video=try H3InputProjection.evaluateRegenerate(checkpointURL:base,kind:.video,input:MLXArray.zeros([1,admission.videoRows,96]))
    let audio=try H3InputProjection.evaluateRegenerate(checkpointURL:base,kind:.audio,input:MLXArray.zeros([1,admission.audioRows,32]))
    let text=try H3InputProjection.evaluate(checkpointURL:base,kind:.condition,input:MLXArray.zeros([1,request.tags.count,5120],dtype:.bfloat16))
    let packed=try H3PackedSequence(layout:layout,text:text,video:video,audio:audio,timestepIndices:clock.indicesByStep[0])
    XCTAssertEqual(packed.embeddings.shape,[1,admission.packedRows,5376])
    let time=try H3TimeEmbedding.evaluate(checkpointURL:base,timesteps:MLXArray(clock.table))
    let result=try H3FinalLayer.evaluate(checkpointURL:base,input:packed.embeddings,timeEmbeddings:time,timestepIndices:packed.timestepIndices,
      videoIndices:packed.videoIndices,audioIndices:packed.audioIndices,maximumRows:H3RegenerateAdmission.maximumPackedRows,observe:{_,_ in})
    eval(result.video,result.audio)
    XCTAssertEqual(result.video.shape,[1,133200,96]);XCTAssertEqual(result.audio.shape,[1,admission.audioRows,32])
    XCTAssertTrue(all(isFinite(result.video)).item(Bool.self));XCTAssertTrue(all(isFinite(result.audio)).item(Bool.self))
    print("Full-grid input/pack/final heads GREEN rows=\(admission.packedRows) seconds=\(Date().timeIntervalSince(started)) peakMLXBytes=\(Memory.peakMemory)")
  }
}
