import Foundation
import AdapterRuntime
import LTX25Engine
import LTX25Video

/// Shared two-stage T2AV sampling component, using the same LTXSamplingRunner for
/// both stages. Text and final media decoding remain separate weighted components.
public final class DistilledSamplingRunner {
  private let recipe: DistilledTwoStageRecipe
  private let root: URL
  private let upscaler: LatentUpscaler
  private let adapters: LoRAWeightStack?
  private let lock = NSLock()
  public private(set) var stageSeconds: [String:Double] = [:]
  public private(set) var graphBuilds: [Int:Int] = [:]
  public init(recipe: DistilledTwoStageRecipe,transformerRoot: URL,
    upscalerCheckpoint: URL,statisticsCheckpoint: URL,loras: [LoRAAdapter] = []) throws {
    self.recipe = recipe; root = transformerRoot
    adapters = loras.isEmpty ? nil : try LTXAdapterCompatibility.weightStack(loras)
    for geometry in [recipe.low,recipe.high] {
      let configuration = try Self.configuration(geometry)
      try AVBlockRunner.validateAllocation(configuration: configuration)
      _ = try DenoiserWeights(root: transformerRoot,configuration: configuration,adapters: adapters)
    }
    let manifestURL = transformerRoot.appendingPathComponent("paged_manifest.json")
    let handle = try FileHandle(forReadingFrom: manifestURL)
    defer { try? handle.close() }
    guard let data = try handle.read(upToCount: 1024*1024+1),data.count <= 1024*1024,
      let manifest = try JSONSerialization.jsonObject(with: data) as? [String:Any],
      manifest["source"] as? String == "ltx-2.5-22b-distilled-transformer-bf16.safetensors" else {
      throw BlockError.invalid("Two-stage distilled sampling requires the released distilled transformer.")
    }
    _ = try LatentUpscalePlan(shape: [recipe.low.latentFrames,recipe.low.latentHeight,recipe.low.latentWidth,128])
    upscaler = try LatentUpscaler(checkpoint: upscalerCheckpoint,statisticsCheckpoint: statisticsCheckpoint)
  }
  private static func configuration(_ g: AVGeometry) throws -> AVBlockConfiguration {
    try AVBlockConfiguration(videoTokens: g.videoTokens,audioTokens: g.audioFrames,textTokens: 1024)
  }
  public func evaluate(videoContext: [Float],audioContext: [Float],
    progress: (String,Int,Int) throws -> Void = { _,_,_ in }) throws -> AVLatents {
    guard videoContext.count == 1024*4096,audioContext.count == 1024*2048,
      videoContext.allSatisfy(\.isFinite),audioContext.allSatisfy(\.isFinite),lock.try() else {
      throw BlockError.invalid("Invalid two-stage conditioning or overlapping evaluation.")
    }
    defer { lock.unlock() }
    stageSeconds = [:]; graphBuilds = [:]
    return try TwoStageTrajectory().evaluate(recipe: recipe,sample: { stage,geometry,state,schedule,noise in
      let start = Date()
      let result: AVLatents = try autoreleasepool {
        let configuration = try Self.configuration(geometry)
        let weights = try DenoiserWeights(root: self.root,configuration: configuration,adapters: self.adapters)
        let runner = try LTXSamplingRunner(configuration: configuration)
        let inputs = ["video_latent":state.video,"audio_latent":state.audio,
          "video_text":videoContext,"audio_text":audioContext,
          "video_positions":geometry.videoPositions,"audio_positions":geometry.audioPositions]
        let output = try runner.evaluate(inputs,schedule: schedule,
          fixedWeights: { try weights.readFixed($0,shape: $1) },
          blockWeights: { try weights.readBlock($0,name: $1,shape: $2) },noise: noise,
          stageProgress: { _,event in try progress("stage\(stage):"+event.stage,event.completedBlocks,48) },
          progress: { event in try progress("sampling",(stage == 1 ? 0 : 8)+event.completedSteps,11) })
        self.graphBuilds[stage] = runner.lastGraphBuildCount
        return output
      }
      self.stageSeconds["stage\(stage)"] = Date().timeIntervalSince(start)
      try progress("stage\(stage)_weights_released",stage,2)
      return result
    },upscale: { values,shape in
      let start = Date()
      let result = try self.upscaler.upscale(packed: values,shape: shape,
        progress: { name in try progress("upscale:"+name,0,1) })
      self.stageSeconds["latent_upscale"] = Date().timeIntervalSince(start)
      try progress("upscaler_weights_released",1,1)
      return result
    })
  }
}
