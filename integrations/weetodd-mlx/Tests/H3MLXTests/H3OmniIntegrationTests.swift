import CryptoKit
import Foundation
import InferenceContracts
import MLX
import XCTest
@testable import H3MLX

final class H3OmniIntegrationTests: XCTestCase {
  private func installed() throws -> H3OmniCheckpoint {
    guard let file = ProcessInfo.processInfo.environment["WEETODD_H3_OMNI_CHECKPOINT"] else {
      throw XCTSkip("Set the pinned Omni source for bounded installed integration tests.")
    }
    return try H3OmniCheckpoint.admit(url:URL(fileURLWithPath:file))
  }
  func testInstalledDedicatedBlockUsesActualShapesAndReleasesOwner() throws {
    let source = try installed()
    let plan = try H3OmniVSAPlan(prefixSegments:[.dense(role:.text,rows:1),.dense(role:.image,rows:1)],targetVideoGrid:[1,1,1])
    let owner = try H3PreparedBlock(checkpointURL:source.url,index:0,projectionMode:.weightDecoded,omni:source)
    defer { owner.close() }
    let time = try H3TimeEmbedding.evaluate(checkpointURL:source.url,timesteps:MLXArray([Float(0.001),0.999]),omni:source)
    let modulation = try H3AdaLNProjection.evaluate(checkpointURL:source.url,blockIndex:0,timeEmbeddings:time,omni:source)
    let positions=MLXArray([Float](repeating:0,count:9),[3,3])
    let angles=try H3TransformerBlock.prepareRotaryAngles(checkpointURL:source.url,positions:positions,omni:source)
    let output=try H3TransformerBlock.evaluateOmni(owner:owner,input:MLXArray.ones([1,3,5376],dtype:.bfloat16)*0.125,
      modulation:modulation,modulationIndices:MLXArray([Int32(0),3,0]),angles:angles,plan:plan)
    XCTAssertEqual(output.shape,[1,3,5376]);XCTAssertEqual(output.dtype,.bfloat16)
    XCTAssertTrue(output.asType(.float32).asArray(Float.self).allSatisfy(\.isFinite))
    owner.close();XCTAssertEqual(owner.storageBytes,0);XCTAssertTrue(owner.isClosed)
    XCTAssertThrowsError(try H3TransformerBlock.evaluateOmni(owner:owner,input:output,
      modulation:modulation,modulationIndices:MLXArray([Int32(0),3,0]),angles:angles,plan:plan))
  }

  func testInstalledFusedFinalHeadsMatchIndependentSourceParameterOracleAtAllEightCalls() throws {
    let source=try installed(), plan=try H3PDDPlan(contract:H3OmniContract(data:omniContractData()))
    try source.withParameters { scope in
      let shift=try scope.read("final_layer.adaln_proj.linear.bias").asType(.float32).asArray(Float.self)
      for modality:H3PDDPlan.Modality in [.video,.audio] {
        let count=modality == .video ? 96:32, name=modality == .video ? "video_out":"audio_out"
        let weight=try scope.read("final_layer."+name+".weight").asType(.float32).asArray(Float.self)
        let bias=try scope.read("final_layer."+name+".bias").asType(.float32).asArray(Float.self)
        let schedulerShift:Double=modality == .video ? 12:3
        func sigma(_ index:Int)->Double { let u=0.999*Double(32-index)/32;return schedulerShift*u/(1+(schedulerShift-1)*u/0.999) }
        var maximumError:Double=0
        for step in 0..<8 {
          let output=try H3FinalLayer.evaluateOmni(source:source,input:MLXArray.zeros([1,2,5376],dtype:.bfloat16),
            timeEmbeddings:MLXArray.zeros([1,2688],dtype:.bfloat16),timestepIndices:MLXArray([Int32(0),0]),
            videoIndices:MLXArray([Int32(0)]),audioIndices:MLXArray([Int32(1)]),plan:plan,step:step)
          XCTAssertEqual(output.video.dtype,.float32);XCTAssertEqual(output.audio.dtype,.float32)
          let actual=(modality == .video ? output.video:output.audio).asArray(Float.self)
          let start=step*4, total=sigma(start+4)-sigma(start)
          let coeff=(start..<(start+4)).map { Float((sigma($0+1)-sigma($0))/total) }
          var stepError:Double=0
          // Runtime heads fuse in FP32, even though the installed bank is stored in BF16.
          // Independent scalar accumulation must not round fused parameters back to storage dtype.
          for channel in 0..<count {
            var b:Float=0
            for h in 0..<4 { b += coeff[h]*bias[(start+h)*count+channel] }
            var expected=Double(b)
            for feature in 0..<5376 {
              var w:Float=0
              for h in 0..<4 { w += coeff[h]*weight[((start+h)*count+channel)*5376+feature] }
              expected += Double(w)*Double(shift[feature])
            }
            stepError=max(stepError,abs(Double(actual[channel])-expected))
          }
          XCTAssertLessThanOrEqual(stepError,0.0001,"\(name) step\(step) FP32 fusion/projection oracle")
          maximumError=max(maximumError,stepError)
        }
        print("Omni FP32 \(name) all 8 calls/all \(count) channels maximum error: \(maximumError)")
      }
    }
  }

  func testInstalledSharedStageReleasesSuccessErrorAndCancellation() throws {
    let source=try installed(), plan=try H3PDDPlan(contract:H3OmniContract(data:omniContractData()))
    let geometry=try H3Geometry(width:64,height:64,durationSeconds:5)
    let layout=try H3ReferenceLayout(geometry:geometry,textTags:[0],references:[.image(latentHeight:2,latentWidth:2)])
    let attention=try H3OmniVSAPlan(prefixSegments:[.dense(role:.text,rows:1),.dense(role:.image,rows:1),.dense(role:.targetAudio,rows:layout.targetAudioIndices.count)],targetVideoGrid:[geometry.videoLatentFrames,2,2])
    let context=H3OmniRunner.Context(source:source,plan:plan,attention:attention)
    for disposition in 0..<3 {
      var retained:H3ReferenceDiTState?
      do {
        let value=try H3ReferenceDiTState.withStage({
          let state=try H3ReferenceDiTState(checkpointURL:source.url,layout:layout,
            textEmbeddings:MLXArray.zeros([1,1,5120],dtype:.bfloat16),timestepTable:[0.001,0.999,1],blockCount:50,omni:context)
          retained=state;return state
        }) { state in
          XCTAssertTrue(state.isResident);XCTAssertGreaterThan(state.residentActivationBytes,0)
          if disposition == 1 { throw H3CheckpointError.invalid("injected downstream failure") }
          if disposition == 2 { throw CancellationError() }
          return 7
        }
        XCTAssertEqual(disposition,0);XCTAssertEqual(value,7)
      } catch {
        XCTAssertNotEqual(disposition,0)
        if disposition == 2 { XCTAssertTrue(error is CancellationError) }
      }
      XCTAssertNotNil(retained);XCTAssertFalse(retained?.isResident ?? true)
      XCTAssertEqual(retained?.residentActivationBytes,0)
    }
  }

  func testAtomicPublicationRejectsLateFailureAndCancellation() throws {
    let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
    defer { try? FileManager.default.removeItem(at:root) }
    let staging=root.appendingPathComponent("staging"),output=root.appendingPathComponent("output")
    try FileManager.default.createDirectory(at:staging,withIntermediateDirectories:true)
    let bytes=Data("final AV and receipt".utf8)
    try bytes.write(to:staging.appendingPathComponent("receipt"))
    XCTAssertThrowsError(try H3WorkerReceipt.publish(staging:staging,output:output) { throw CancellationError() })
    XCTAssertFalse(FileManager.default.fileExists(atPath:output.path))
    let media = root.appendingPathComponent("source")
    try bytes.write(to:media)
    let identity = try NativeMediaSource(path:media.path,sha256:SHA256.hash(data:bytes).map { String(format:"%02x",$0) }.joined())
    try Data("changed source".utf8).write(to:media)
    XCTAssertThrowsError(try H3WorkerReceipt.publish(staging:staging,output:output) { try identity.verify() })
    XCTAssertFalse(FileManager.default.fileExists(atPath:output.path))
    try bytes.write(to:media)
    try H3WorkerReceipt.publish(staging:staging,output:output) { try identity.verify() }
    XCTAssertFalse(FileManager.default.fileExists(atPath:staging.path))
    XCTAssertEqual(try Data(contentsOf:output.appendingPathComponent("receipt")),bytes)
  }
}
