import Foundation
import InferenceContracts
import MLX

/// Stage-local context used by the one existing weighted DiT implementation.
struct H3RegenerateExecutionContext {
  let adapter:H3RegenerateAdapter
  let sharp:H3SharpCurve
  let admission:H3RegenerateAdmission
  let attentionPlan:H3RegenerateAttentionPlan
}

public enum H3RegenerateRunner {
  public struct Result {
    public let video:MLXArray
    public let encodedSilence:MLXArray
    public let recipeFingerprint:String,sourceIdentity:String,sourceAudioIdentity:String,noiseIdentity:String
    public let visibleFrames:Int,paddedFrames:Int,evaluations:Int
    public let qualificationProbe:Bool
    public let denoiseSeconds:Double
    public let attentionPlan:H3RegenerateAttentionPlan
  }

  /// Source content verification/preparation is caller-owned. In particular,
  /// `source` is already on the admitted target grid; a learned-upscale stage
  /// must finish and release its weights before entering this transformer stage.
  public static func run(prepared:H3RegenerateRecipe.Prepared,source:MLXArray,noise:MLXArray,
    encodedSilence:MLXArray,textEmbeddings:MLXArray,textTags:[Int32],qualificationProbe:Bool=false,
    progress:(Int,Int)->Void={_,_ in},blockProgress:(Int,Int,Int)->Void={_,_,_ in}) throws -> Result {
    try H3RegenerateProfile.validateBaseSHA256(prepared.component(.transformer).identity.sourceSHA256)
    try prepared.admission.validateExecution(qualificationProbe:qualificationProbe)
    let admission=prepared.admission
    guard admission.maximumStageBytes <= ProcessInfo.processInfo.physicalMemory else {
      throw H3CheckpointError.invalid("Regenerate stage budget exceeds this Mac’s physical memory.")
    }
    guard textTags.count == admission.textRows,textTags.allSatisfy({ $0 == 0 || $0 == 1 }),
      textEmbeddings.shape == [1,textTags.count,5120],textEmbeddings.dtype.isFloatingPoint,
      source.shape == [1,admission.videoRows,96],noise.shape == source.shape,
      source.dtype == .float32,noise.dtype == .float32,
      encodedSilence.shape == [1,admission.audioRows,32],encodedSilence.dtype == .bfloat16,
      all(isFinite(source)).item(Bool.self),all(isFinite(noise)).item(Bool.self),
      all(isFinite(encodedSilence)).item(Bool.self),all(isFinite(textEmbeddings)).item(Bool.self) else {
      throw H3CheckpointError.invalid("Regenerate prepared source, exact noise, encoded silence or Qwen rows differ from the frozen plan.")
    }
    let layout=try H3PackedLayout(geometry:admission.geometry,textTags:textTags,anchors:[])
    let rowSchedule=try H3RowSchedule(regenerate:layout,schedule:prepared.schedule)
    let adapter=try H3RegenerateAdapter(url:URL(fileURLWithPath:prepared.component(.refiner).path),
      profile:prepared.profile,companions:[.sharpCurve])
    defer { adapter.close() }
    let sharp=try H3SharpCurve(tableURL:URL(fileURLWithPath:prepared.component(.sharpInput).path),
      branchesURL:URL(fileURLWithPath:prepared.component(.sharpCurve).path))
    defer { sharp.close(); Stream.gpu.synchronize();Memory.clearCache() }
    let attentionPlan=try H3RegenerateAttentionPlan.runtime(rows:admission.packedRows,heads:56,headWidth:128,dtype:.bfloat16)
    let context=H3RegenerateExecutionContext(adapter:adapter,sharp:sharp,admission:admission,attentionPlan:attentionPlan)
    let started=Date()
    let sampled=try adapter.withStage(.regenerate) {
      let state=try H3DiTState(checkpointURL:URL(fileURLWithPath:prepared.component(.transformer).path),
        layout:layout,textEmbeddings:textEmbeddings,timestepTable:rowSchedule.table,blockCount:50,
        regenerate:context)
      defer { state.unload() }
      return try H3AVSampler.runRegenerate(predictor:state,schedule:prepared.schedule,
        source:source,noise:noise,encodedSilence:encodedSilence,progress:progress,blockProgress:blockProgress)
    }
    eval(sampled.video)
    return Result(video:sampled.video,encodedSilence:sampled.audio,recipeFingerprint:prepared.fingerprint,
      sourceIdentity:prepared.source.contentSHA256,sourceAudioIdentity:prepared.source.audioSHA256,
      noiseIdentity:prepared.noiseSHA256,visibleFrames:admission.visibleFrames,paddedFrames:admission.paddedFrames,
      evaluations:prepared.schedule.evaluations,qualificationProbe:qualificationProbe,denoiseSeconds:Date().timeIntervalSince(started),
      attentionPlan:attentionPlan)
  }
}
