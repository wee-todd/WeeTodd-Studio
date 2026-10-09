import Foundation
import XCTest
import MLX
import LTX25Engine
@testable import LTX25MLX

final class MLXGuidedSamplingTests: XCTestCase {
  func testSingleStageDevCannotHideRefinementOrChangeHQAndLegacyDefaults() throws {
    var fields:[String:Any]=["mode":"guided","steps":30,"single_stage":true,
      "stg_audio":false,"distilled_adapter_path":"","stg_scale":1,"stg_blocks":[29],
      "video_cfg_scale":4,"audio_cfg_scale":4,"modality_scale":1]
    func decode() throws -> MLXGuidedSampling {
      try JSONDecoder().decode(MLXGuidedSampling.self,from:JSONSerialization.data(withJSONObject:fields))
    }
    let admitted=try decode()
    XCTAssertTrue(admitted.singleStage);XCTAssertFalse(admitted.stgAudio)
    XCTAssertEqual(admitted.transformerEvaluations,90)
    fields["mode"]="guided_hq";XCTAssertThrowsError(try decode())
    fields["mode"]="guided";fields["distilled_adapter_path"]="/unexpected.safetensors"
    XCTAssertThrowsError(try decode())
    fields["single_stage"]=false;fields.removeValue(forKey:"stg_audio")
    let legacy=try decode();XCTAssertFalse(legacy.singleStage);XCTAssertTrue(legacy.stgAudio)
    fields["stg_audio"]=1;XCTAssertThrowsError(try decode())
  }
  private func policy(_ mode: MLXGuidedSampling.Mode = .guided, sigmas: [Double]? = nil) throws -> MLXGuidedSampling {
    try MLXGuidedSampling(mode: mode, steps: sigmas.map { $0.count-1 } ?? 3,
      stg: 0, videoRescale: 0, audioRescale: 0, modality: 1, stgBlocks: [],
      sigmas: sigmas, distilledAdapterPath: "/models/refinement.safetensors")
  }
  func testSeparateGuidanceBranchesAndVarianceRescaling() throws {
    Device.withDefaultDevice(.cpu) {
      let conditional=MLXArray([Float(1),2,4,8]), negative=MLXArray([Float(0),0,1,2])
      let perturbed=MLXArray([Float(0.2),0.3,0.4,0.5]), isolated=MLXArray([Float(0.8),1,1.6,2])
      let output=MLXGuidanceMath.combine(conditional,negative:negative,perturbed:perturbed,
        isolated:isolated,cfg:3,stg:1,modality:3,rescale:0)
      for (actual,expected) in zip(output.asArray(Float.self),[Float(4.2),9.7,18.4,39.5]) {
        XCTAssertEqual(actual,expected,accuracy:0.00001)
      }
      let scaled=MLXGuidanceMath.combine(conditional,negative:negative,perturbed:perturbed,
        isolated:isolated,cfg:3,stg:1,modality:3,rescale:1)
      XCTAssertEqual(scaled.variance().item(Float.self),conditional.variance().item(Float.self),accuracy:0.00002)
      let neutral=MLXGuidanceMath.combine(conditional,negative:nil,perturbed:nil,isolated:nil,
        cfg:1,stg:0,modality:1,rescale:0)
      XCTAssertEqual(neutral.asArray(Float.self),conditional.asArray(Float.self))
    }
  }
  func testAdaptiveScheduleAndHQAdmissionRejectMalformedControls() throws {
    let values=try MLXGuidedSampling.adaptiveSigmas(steps:30,videoTokens:4096)
    XCTAssertEqual(values.count,31); XCTAssertEqual(values.first,1); XCTAssertEqual(values.last,0)
    XCTAssertEqual(values[29],0.1,accuracy:1e-12)
    XCTAssertTrue(zip(values,values.dropFirst()).allSatisfy { $0 > $1 })
    let large=try MLXGuidedSampling.adaptiveSigmas(steps:64,videoTokens:131072)
    XCTAssertTrue(zip(large,large.dropFirst()).allSatisfy { $0 > $1 })
    XCTAssertThrowsError(try policy(.guidedHQ,sigmas:[1,0.0001,0]))
    XCTAssertThrowsError(try policy(sigmas:[1,0.4,0.6,0]))
    XCTAssertThrowsError(try policy(sigmas:[1,Double.nan,0]))
    XCTAssertThrowsError(try JSONDecoder().decode(MLXGuidedSampling.self,from:Data(
      #"{"mode":"guided","steps":3,"distilled_adapter_path":"/model","unknown":true}"#.utf8)))
  }
  func testEulerAndHQTrajectoriesAgainstIndependentPythonSamplers() throws {
    let url=try XCTUnwrap(Bundle.module.url(forResource:"ltx-dev-guided-trajectory",withExtension:"json",subdirectory:"Fixtures"))
    let fixture=try XCTUnwrap(JSONSerialization.jsonObject(with:Data(contentsOf:url)) as? [String:Any])
    func floats(_ value:Any?) throws -> [Float] { try XCTUnwrap(value as? [NSNumber]).map { $0.floatValue } }
    let sigmas=try XCTUnwrap(fixture["sigmas"] as? [Double])
    let noise=try XCTUnwrap(fixture["noise"] as? [[NSNumber]])
    try Device.withDefaultDevice(.cpu) {
      let video=MLXArray(try floats(fixture["video"]),[3,4]), audio=MLXArray(try floats(fixture["audio"]),[4,4])
      let vb=MLXArray(try floats(fixture["videoBias"]),[3,4]), ab=MLXArray(try floats(fixture["audioBias"]),[4,4])
      for (name,mode) in [("euler",MLXGuidedSampling.Mode.guided),("hq",.guidedHQ)] {
        let sampling=try policy(mode,sigmas:sigmas)
        let expected=try XCTUnwrap(fixture[name] as? [String:Any])
        var calls=0,draws=0,events:[Int]=[]
        let output=try MLXGuidedTrajectory.evaluate(["video":video,"audio":audio],sampling:sampling,
          schedule:SamplingSchedule(sigmas:sigmas,eta:0),frozenAudio:false,
          noise:{ _,_,_,shape in
            defer { draws+=1 }
            return MLXArray(noise[draws].map(\.floatValue),shape)
          },predict:{ state,sigma in
            calls+=1
            let time=DenoiserMath.bfloat16(sigma)
            // The independent source sampler casts model inputs and X0
            // outputs to BF16 in both modes, retaining HQ trajectory in FP32.
            let v=(state["video"]!.asType(.bfloat16).asType(.float32)*Float(0.31)+time*Float(0.17)+vb)
              .asType(.bfloat16).asType(.float32)
            let a=(state["audio"]!.asType(.bfloat16).asType(.float32)*Float(0.23)-time*Float(0.11)+ab)
              .asType(.bfloat16).asType(.float32)
            return ["video":v,"audio":a]
          },progress:{ events.append($0.completedSteps) })
        for modality in ["video","audio"] {
          let reference=try floats(expected[modality]),actual=output[modality]!.asArray(Float.self)
          XCTAssertEqual(actual.count,reference.count)
          for (a,b) in zip(actual,reference) { XCTAssertEqual(a,b,accuracy:0.000001,"\(name) \(modality)") }
        }
        XCTAssertEqual(calls,mode == .guided ? 3 : 7)
        XCTAssertEqual(draws,mode == .guided ? 0 : 12)
        XCTAssertEqual(events,mode == .guided ? [1,2,3] : [1,2,3,4])
      }
    }
  }
  func testFrozenAudioRemainsExactThroughGuidedTrajectory() throws {
    try Device.withDefaultDevice(.cpu) {
      let video=MLXArray([Float(0.2),0.4],[1,2]),audio=MLXArray([Float(0.123456),0.987654],[1,2])
      let output=try MLXGuidedTrajectory.evaluate(["video":video,"audio":audio],sampling:policy(sigmas:[1,0.5,0]),
        schedule:SamplingSchedule(sigmas:[1,0.5,0],eta:0),frozenAudio:true,
        noise:{ _,_,_,_ in XCTFail("Euler does not draw noise"); return video },
        predict:{ state,_ in ["video":state["video"]!*Float(0.2),"audio":audio*100] },progress:{ _ in })
      XCTAssertEqual(output["audio"]!.asArray(Float.self),audio.asArray(Float.self))
    }
  }
  func testSeededHQNoiseMatchesIndependentRecordedPythonDrawOrder() throws {
    let url=try XCTUnwrap(Bundle.module.url(forResource:"ltx-dev-guided-trajectory",withExtension:"json",subdirectory:"Fixtures"))
    let fixture=try XCTUnwrap(JSONSerialization.jsonObject(with:Data(contentsOf:url)) as? [String:Any])
    let noise=try XCTUnwrap(fixture["noise"] as? [[NSNumber]])
    Device.withDefaultDevice(.cpu) {
      var offset=0
      for index in 0..<3 { for substep in [true,false] { for modality in ["video","audio"] {
        let shape=modality == "video" ? [3,4] : [4,4]
        let draw=MLXGuidedTrajectory.seededNoise(index:index,substep:substep,modality:modality,shape:shape)
        for (actual,expected) in zip(draw.asArray(Float.self),noise[offset].map(\.floatValue)) {
          XCTAssertEqual(actual,expected,accuracy:0.000001)
        }
        offset+=1
      } } }
      XCTAssertEqual(offset,12)
    }
  }
}
