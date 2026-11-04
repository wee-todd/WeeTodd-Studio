import XCTest
import MLX
import LTX25Engine
@testable import LTX25MLX

final class MLXPreviewHookTests: XCTestCase {
  func testGuidedPreviewMutationLeavesEulerHQFrozenAudioAndNoiseOrderExact() throws {
    try Device.withDefaultDevice(.cpu) {
      for mode in [MLXGuidedSampling.Mode.guided, .guidedHQ] {
        for frozen in [false, true] {
          let sampling = try MLXGuidedSampling(mode: mode, steps: 2, stg: 0,
            videoRescale: 0, audioRescale: 0, modality: 1, stgBlocks: [],
            distilledAdapterPath: "/models/refinement")
          let schedule = try SamplingSchedule(sigmas: [1, 0.5, 0], eta: 0)
          let initial = ["video": MLXArray([Float(0.2), -0.4], [1, 2]),
            "audio": MLXArray([Float(0.1), 0.3], [1, 2])]
          var draws: [String] = [], previews = 0
          func run(preview: (([String: MLXArray], MLXSamplingRunner.Progress) throws -> Void)?) throws -> [String: MLXArray] {
            try MLXGuidedTrajectory.evaluate(initial, sampling: sampling, schedule: schedule,
              frozenAudio: frozen, noise: { i, sub, name, shape in
                draws.append("\(i)-\(sub)-\(name)"); return MLXArray.ones(shape) * Float(0.2)
              }, predict: { state, _ in state.mapValues { $0 * Float(0.4) } },
              denoisedPreview: preview, progress: { _ in })
          }
          let control = try run(preview: nil), controlDraws = draws
          draws = []
          let result = try run { values, event in
            previews += 1
            XCTAssertEqual(values["video"]?.shape, [1, 2])
            XCTAssertGreaterThan(event.completedSteps, 0)
            for value in values.values { value[.ellipsis] = MLXArray.zeros(value.shape) }
          }
          XCTAssertEqual(previews, mode == .guided ? 2 : 3)
          XCTAssertEqual(draws, controlDraws)
          for name in ["video", "audio"] {
            XCTAssertEqual(control[name]!.asArray(Float.self), result[name]!.asArray(Float.self))
          }
        }
      }
    }
  }

  func testEulerCleanPreviewIsAfterWeightReleaseAndCannotChangeTrajectory() throws {
    let source = MLXSamplingTests(), fixture = try source.fixture()
    let runner = try MLXSamplingRunner(configuration: fixture.configuration, blockCount: 1)
    let inputs = fixture.inputs.mapValues { MLXArray($0) }
    let schedule = try SamplingSchedule(sigmas: [1, 0.5, 0], eta: 0)
    let blocks: MLXDenoiser.BlockProvider = { try source.weight("transformer_blocks.\($0)." + $1, $2) }
    let control = try runner.evaluate(inputs, schedule: schedule, fixedWeights: source.weight, blockWeights: blocks)
    var previews = 0
    let result = try runner.evaluateWithDenoisedPreview(inputs, schedule: schedule, fixedWeights: source.weight, blockWeights: blocks,
      denoisedPreview: { values, _ in
        previews += 1
        XCTAssertEqual(runner.residentWeightBytes, 0)
        for value in values.values { value[.ellipsis] = MLXArray.zeros(value.shape) }
      })
    XCTAssertEqual(previews, 2)
    for name in ["video", "audio"] {
      XCTAssertEqual(control[name]!.asArray(Float.self), result[name]!.asArray(Float.self))
    }
  }
}
