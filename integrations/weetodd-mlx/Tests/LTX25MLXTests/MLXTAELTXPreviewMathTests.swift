import Foundation
import XCTest
import MLX
import TensorIO
import InferenceTestSupport
@testable import LTX25MLX

final class MLXTAELTXPreviewMathTests:XCTestCase {
  func testTemporalExpansionUsesTimeThenChannelAndHistoryStartsEmpty() throws {
    let projected=MLXArray((0..<16).map(Float.init),[2,1,2,4])
    let expanded=MLXTAELTXPreviewDecoder.temporalExpand(projected)
    XCTAssertEqual(expanded.shape,[4,1,2,2])
    XCTAssertEqual(expanded.asArray(Float.self),[0,1,4,5,2,3,6,7,8,9,12,13,10,11,14,15])
    let memory=MLXTAELTXPreviewDecoder.causalHistory(expanded)
    XCTAssertEqual(memory.asArray(Float.self),[0,0,0,0,0,1,4,5,2,3,6,7,8,9,12,13])
  }
  func testPixelShuffleAndPublisherUInt8RoundingPreserveRGBOrdering() throws {
    let x=MLXArray((0..<48).map(Float.init),[1,1,1,48])
    let shuffled=MLXTAELTXPreviewDecoder.pixelShuffle(x)
    XCTAssertEqual(shuffled.shape,[1,4,4,3])
    XCTAssertEqual(shuffled[0,0,0].asArray(Float.self),[0,16,32])
    XCTAssertEqual(shuffled[0,0,1].asArray(Float.self),[1,17,33])
    XCTAssertEqual(shuffled[0,1,0].asArray(Float.self),[4,20,36])
    XCTAssertEqual(shuffled[0,3,3].asArray(Float.self),[15,31,47])
    XCTAssertEqual(Array(MLXTAELTXPreviewDecoder.rgb8(MLXArray([Float(-1),0,0.3,0.5,1,2]))),[0,0,76,128,255,255])
  }
  func testGroupedConvolutionMatchesIndependentScalarOperator() throws {
    let frames=2,height=3,width=2,channels=4,groups=2
    let values=(0..<frames*height*width*channels).map { sin(Float($0)*0.3) }
    let weights=(0..<4*2*3*3).map { cos(Float($0)*0.2)*0.05 }
    let bias:[Float]=[0.1,-0.2,0.3,-0.4]
    var expected:[Float]=[]
    for t in 0..<frames { for y in 0..<height { for x in 0..<width { for out in 0..<4 {
      var value=bias[out]
      for local in 0..<2 { for ky in 0..<3 { for kx in 0..<3 {
        let yy=y+ky-1,xx=x+kx-1,input=(out/2)*2+local
        if yy>=0 && yy<height && xx>=0 && xx<width {
          value += values[((t*height+yy)*width+xx)*channels+input]*weights[((out*2+local)*3+ky)*3+kx]
        }
      } } }
      expected.append(value)
    } } } }
    let actual=try MLXTAELTXPreviewDecoder.convolve(MLXArray(values,[frames,height,width,channels]),
      weight:MLXArray(weights,[4,2,3,3]),bias:MLXArray(bias),groups:groups).asArray(Float.self)
    for (a,b) in zip(actual,expected) { XCTAssertEqual(a,b,accuracy:0.00001) }
  }
  func testDecodeValidatesInputAndCheckpointReplacementBeforeLoadingWeights() throws {
    try withTensorFile(tensors:MLXTAELTXPreviewHeaderTests.tensors(wide:false)) { url in
      let decoder=try MLXTAELTXPreviewDecoder(checkpoint:url)
      for input in [MLXArray.zeros([1,128,1,1,1],dtype:.float16),MLXArray.zeros([1,127,1,1,1]),
        MLXArray.full([1,128,1,1,1],values:MLXArray(Float.nan))] {
        XCTAssertThrowsError(try decoder.decode(latent:input))
        XCTAssertEqual(decoder.residentWeightBytes,0)
      }
      let replacement=url.appendingPathExtension("replacement")
      defer { try? FileManager.default.removeItem(at:replacement) }
      try FileManager.default.copyItem(at:url,to:replacement)
      try FileManager.default.removeItem(at:url)
      try FileManager.default.moveItem(at:replacement,to:url)
      XCTAssertThrowsError(try decoder.decode(latent:.zeros([1,128,1,1,1])))
      XCTAssertEqual(decoder.residentWeightBytes,0)
    }
  }
  func testInstalledStandardAndWideAgainstPublisherOracles() throws {
    let env=ProcessInfo.processInfo.environment
    guard let checkpoints=env["WEETODD_TAE_CHECKPOINT_DIRECTORY"],let oracles=env["WEETODD_TAE_ORACLE_DIRECTORY"] else {
      throw XCTSkip("Installed TAELTX publisher oracle is opt-in.")
    }
    for variant in ["taeltx2_3","taeltx2_3_wide"] {
      let decoder=try MLXTAELTXPreviewDecoder(checkpoint:URL(fileURLWithPath:checkpoints).appendingPathComponent(variant+".safetensors"))
      XCTAssertEqual(decoder.variant.rawValue,variant)
      for frames in [2,3] {
        let prefix=URL(fileURLWithPath:oracles).appendingPathComponent("\(variant)-f\(frames)")
        let file=try SafeTensorFile(url:prefix.appendingPathExtension("safetensors"))
        let latent=try MLXWeight.read(file,"video")
        let expected=try Data(contentsOf:prefix.appendingPathExtension("rgb24"))
        let before=latent.asArray(Float.self)
        let frame=try decoder.decode(latent:latent)
        let output=try XCTUnwrap(file.tensors["output"])
        XCTAssertEqual(frame.width,Int(output.shape[4]))
        XCTAssertEqual(frame.height,Int(output.shape[3]))
        XCTAssertEqual(frame.frameIndex+1,Int(output.shape[1]))
        XCTAssertEqual(latent.asArray(Float.self),before)
        XCTAssertEqual(frame.rgb.count,expected.count)
        let errors=zip(frame.rgb,expected).map { abs(Int($0)-Int($1)) }
        print("TAE \(variant) frames=\(frames) seconds=\(decoder.lastDecodeSeconds) max_u8_error=\(errors.max() ?? 0) mean_u8_error=\(Double(errors.reduce(0,+))/Double(errors.count)) peak_mlx=\(Memory.peakMemory)")
        XCTAssertLessThanOrEqual(errors.max() ?? 0,2)
        XCTAssertLessThan(Double(errors.reduce(0,+))/Double(errors.count),0.15)
        XCTAssertEqual(decoder.residentWeightBytes,0)
        if frames==3 {
          let tail=MLXArray.full([1,128,1,2,3],values:MLXArray(Float(1000)))
          let extended=concatenated([latent,tail],axis:2)
          let bounded=try decoder.decode(latent:extended)
          XCTAssertEqual(bounded.frameIndex,16)
          XCTAssertEqual(bounded.rgb,frame.rgb,"Preview must use the first three latent frames and ignore the finite tail.")
          XCTAssertEqual(decoder.residentWeightBytes,0)
        }
      }
    }
  }
  func testNonfiniteWeightsReleaseResidentStateAndRestoreWorkspace() throws {
    var payload=Data(count:256*2),nan=Float16.nan.bitPattern.littleEndian
    payload.replaceSubrange(0..<2,with:withUnsafeBytes(of:&nan) { Data($0) })
    let key="MLX_CONV_WINOGRAD_WORKING_SET",prior=getenv("MLX_CONV_WINOGRAD_WORKING_SET").map { String(cString:$0) }
    defer { if let prior { setenv(key,prior,1) } else { unsetenv(key) } }
    setenv(key,"134217728",1)
    try withTensorFile(tensors:MLXTAELTXPreviewHeaderTests.tensors(wide:false),payloads:["decoder.1.bias":payload]) { url in
      let decoder=try MLXTAELTXPreviewDecoder(checkpoint:url)
      XCTAssertThrowsError(try decoder.decode(latent:.zeros([1,128,1,1,1])))
      XCTAssertEqual(decoder.residentWeightBytes,0)
      XCTAssertEqual(String(cString:getenv(key)),"134217728")
    }
  }
  func testDecoderCanSucceedOnSameInstanceAfterFiniteValidationFailure() throws {
    let half=[Float16](repeating:0.5,count:48)
    let payload=half.withUnsafeBufferPointer { Data(buffer:$0) }
    try withTensorFile(tensors:MLXTAELTXPreviewHeaderTests.tensors(wide:false),payloads:["decoder.22.bias":payload]) { url in
      let decoder=try MLXTAELTXPreviewDecoder(checkpoint:url)
      XCTAssertThrowsError(try decoder.decode(latent:MLXArray.full([1,128,1,1,1],values:MLXArray(Float.nan))))
      // All learned coefficients are zero except the final RGB patch bias.
      // This independently fixes the entire decoded image at 0.5/UInt8 128.
      let frame=try decoder.decode(latent:.zeros([1,128,1,1,1]))
      XCTAssertEqual(frame.width,32)
      XCTAssertEqual(frame.height,32)
      XCTAssertEqual(frame.frameIndex,0)
      XCTAssertEqual(frame.rgb,Data(repeating:128,count:32*32*3))
      XCTAssertEqual(decoder.residentWeightBytes,0)
    }
  }
  func testInstalled384PixelBoundsAndTransientMemory() throws {
    let env=ProcessInfo.processInfo.environment
    guard env["WEETODD_TAE_BENCHMARK"]=="1",let checkpoints=env["WEETODD_TAE_CHECKPOINT_DIRECTORY"] else {
      throw XCTSkip("Installed TAELTX 384-pixel qualification is opt-in.")
    }
    for (shape,wantedHeight,wantedWidth) in [([1,128,3,24,42],224,384),([1,128,3,16,16],384,384)] {
      let input=MLXArray((0..<shape.reduce(1,*)).map { sin(Float($0)*0.017)*0.7 },shape)
      let before=input.asArray(Float.self)
      for variant in ["taeltx2_3","taeltx2_3_wide"] {
        let decoder=try MLXTAELTXPreviewDecoder(checkpoint:URL(fileURLWithPath:checkpoints).appendingPathComponent(variant+".safetensors"))
        Stream.gpu.synchronize();Memory.clearCache()
        let baseline=Memory.activeMemory;Memory.peakMemory=baseline
        let frame=try decoder.decode(latent:input)
        let peak=Memory.peakMemory-baseline
        XCTAssertEqual(frame.width,wantedWidth)
        XCTAssertEqual(frame.height,wantedHeight)
        XCTAssertEqual(frame.rgb.count,wantedHeight*wantedWidth*3)
        XCTAssertEqual(frame.frameIndex,16)
        XCTAssertEqual(decoder.residentWeightBytes,0)
        XCTAssertEqual(input.asArray(Float.self),before)
        XCTAssertLessThan(peak,1024*1024*1024)
        XCTAssertLessThan(Memory.activeMemory-baseline,8*1024*1024)
        print("TAE_BOUND \(variant) size=\(frame.width)x\(frame.height) seconds=\(decoder.lastDecodeSeconds) peak_mlx_delta=\(peak) retained_mlx_delta=\(Memory.activeMemory-baseline)")
      }
    }
  }
  func testCancellationPropagatesWithoutRetainingWeights() async throws {
    let task=Task.detached { () throws -> Int in
      try withTensorFile(tensors:MLXTAELTXPreviewHeaderTests.tensors(wide:false)) { url in
        let decoder=try MLXTAELTXPreviewDecoder(checkpoint:url)
        withUnsafeCurrentTask { $0?.cancel() }
        do {
          _=try decoder.decode(latent:.zeros([1,128,1,1,1]))
          XCTFail("A cancelled preview must propagate cancellation.")
        } catch is CancellationError {} catch { throw error }
        return decoder.residentWeightBytes
      }
    }
    let resident=try await task.value
    XCTAssertEqual(resident,0)
  }
}
