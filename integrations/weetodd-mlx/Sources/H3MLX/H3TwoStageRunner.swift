import Foundation
import MLX

public struct H3TwoStageRunner {
  public let baseTask:H3TwoStageTask
  public let finalTask:H3TwoStageTask
  public let plan:H3TwoStagePlan
  public let omittedTurboAdapters:[String]
  public let resizeMethod:H3SpatialLatentResizeMethod?
  public let learnedUpscaler:URL?
  public let learnedUpscalerHeaderSHA256:String?

  public init(task:H3TwoStageTask,plan:H3TwoStagePlan,resizeMethod:H3SpatialLatentResizeMethod? = nil,
    learnedUpscaler:URL? = nil,learnedUpscalerHeaderSHA256:String? = nil,
    draftKeyframes:[H3StillReference]? = nil,draftOnlyAdapters:[URL] = []) throws {
    guard task.geometry.width==plan.target.width,task.geometry.height==plan.target.height,task.geometry.frames==plan.target.frames,
      (resizeMethod != nil) != (learnedUpscaler != nil),
      (learnedUpscaler != nil)==(learnedUpscalerHeaderSHA256 != nil) else {
      throw H3CheckpointError.invalid("H3 two-stage task needs its exact final canvas and one explicit latent-upscale method.")
    }
    let explicit=Set(draftOnlyAdapters.map {$0.standardizedFileURL.path})
    guard draftOnlyAdapters.count<=8,explicit.count==draftOnlyAdapters.count,
      explicit.isSubset(of:Set(task.adapters.map {$0.url.standardizedFileURL.path})),
      explicit.isEmpty || plan.target.stillImageMode == .qualityFiveFrames else {
      throw H3CheckpointError.invalid("Explicit draft-only adapters must be unique members of a quality-still LoRA stack.")
    }
    var standard:[H3LoRAAdapter]=[],omitted:[String]=[]
    for adapter in task.adapters {
      if try explicit.contains(adapter.url.standardizedFileURL.path) ||
        H3LoRAFile.isTurboForTwoStage(adapter,requestedSteps:task.requestedSteps,samplingMethod:task.samplingMethod) {omitted.append(adapter.url.path)}
      else {standard.append(adapter)}
    }
    self.baseTask=try task.replacing(geometry:plan.base,adapters:task.adapters,draftKeyframes:draftKeyframes)
    self.finalTask=try task.replacing(geometry:plan.target,adapters:standard)
    self.plan=plan;self.omittedTurboAdapters=omitted;self.resizeMethod=resizeMethod
    self.learnedUpscaler=learnedUpscaler;self.learnedUpscalerHeaderSHA256=learnedUpscalerHeaderSHA256
  }
  public struct Admission {
    public let basePackedRows:Int
    public let finalPackedRows:Int
    public let baseEvaluations:Int
    public let refinementEvaluations:Int
    public var evaluations:Int {baseEvaluations+refinementEvaluations}
  }
  public func preflight() throws -> Admission {
    try Task.checkCancellation()
    let base=try baseTask.preflight(),final=try finalTask.preflight(refinement:plan.refinement)
    if let url=learnedUpscaler,let digest=learnedUpscalerHeaderSHA256 {
      try H3LearnedLatentUpscaler.validateGeometry(source:plan.base,target:plan.target)
      _ = try H3LearnedUpscalerLayout(url:url,expectedHeaderSHA256:digest)
      let convolution=try H3UpscalerConvolutionPlan()
      _ = try convolution.validate(channels:512,kernel:3,bytesPerElement:4)
      _ = try convolution.tileCount(frames:plan.target.videoLatentFrames,height:plan.target.height/16,width:plan.target.width/16)
    }
    return Admission(basePackedRows:base.rows,finalPackedRows:final.rows,baseEvaluations:base.evaluations,refinementEvaluations:final.evaluations)
  }
  public func metadata(_ admission:Admission) throws -> [String:Any] {
    let schedules=try plan.refinement.schedules(requestedSteps:finalTask.requestedSteps)
    return ["version":1,"baseWidth":plan.base.width,"baseHeight":plan.base.height,
      "finalWidth":plan.target.width,"finalHeight":plan.target.height,"draftScale":plan.draftScale,
      "effectiveScaleX":Double(plan.target.width)/Double(plan.base.width),"effectiveScaleY":Double(plan.target.height)/Double(plan.base.height),
      "strength":plan.strength,"baseEvaluations":admission.baseEvaluations,"refinementEvaluations":admission.refinementEvaluations,
      "basePackedRows":admission.basePackedRows,"finalPackedRows":admission.finalPackedRows,
      "startVideoSigma":Double(schedules.video.sigmas[0]),"startAudioSigma":Double(schedules.audio.sigmas[0]),
      "sourceAudioPreserved":true,"intermediateDecode":false,"refinementAdapterPolicy":"standard",
      "omittedTurboAdapters":omittedTurboAdapters,"resizeMethod":resizeMethod?.rawValue ?? "learned_3d",
      "learnedUpscaler":learnedUpscaler?.path as Any? ?? NSNull(),
      "learnedUpscalerHeaderSHA256":learnedUpscalerHeaderSHA256 as Any? ?? NSNull()]
  }
  public func run(onFrame:(Int,Data)throws->Void,onAudio:([Float],Int)throws->Void,
    onLatents:(([Float],[Float])throws->Void)? = nil,
    progress:(String,Int,Int)->Void = {_,_,_ in}) throws -> H3AVOutputDecoder.Result {
    let sampled=try sample(progress:progress)
    try onLatents?(sampled.rows.video,sampled.rows.audio)
    let d=finalTask.decoder
    return try H3AVOutputDecoder.decode(videoRows:sampled.rows.video,audioRows:sampled.rows.audio,geometry:plan.target,
      videoVAE:d.video,audioVAE:d.audio,videoDecodeMemoryMode:d.memory,videoDecodePrecision:d.precision,
      backendReport:sampled.backendReport,onFrame:onFrame,onAudio:onAudio,progress:progress)
  }
  /// Shared latent-only execution lets the image worker decode one still after
  /// refinement, without loading an audio decoder or exporting an intermediate.
  public func sample(progress:(String,Int,Int)->Void = {_,_,_ in},
    onStage:((String,Double)->Void)? = nil) throws -> H3SampledLatents {
    let admitted=try preflight()
    var report:H3BackendReport?
    func stageProgress(_ prefix:String,_ offset:Int,_ stage:String,_ completed:Int,_ total:Int) {
      if stage=="sampling" {progress(stage,completed+offset,admitted.evaluations)}
      else if stage.hasPrefix("sampling_block_"),let index=Int(stage.dropFirst("sampling_block_".count)) {
        progress("sampling_block_\(index+offset)",completed,total)
      } else {progress(prefix+stage,completed,total)}
    }
    defer {Stream.gpu.synchronize();Memory.clearCache()}
    return try H3TwoStageExecution.run(plan:plan,preflight:{},sampleBase:{
      let started=Date()
      let rows=try baseTask.sample {stage,c,t in stageProgress("draft_",0,stage,c,t)}.rows
      onStage?("base",Date().timeIntervalSince(started));return rows
    },upscaleVideo:{video in
      let started=Date()
      defer {onStage?("upscale",Date().timeIntervalSince(started))}
      return try autoreleasepool {
        let rows=MLXArray(video,[1,plan.base.videoRows,96])
        let result:MLXArray
        if let checkpoint=learnedUpscaler,let digest=learnedUpscalerHeaderSHA256 {
          let scaled=try H3LearnedLatentUpscaler.upscale(rows:rows,source:plan.base,target:plan.target,
            checkpointURL:checkpoint,expectedHeaderSHA256:digest,videoVAE:finalTask.decoder.video) {c,t in progress("learned_spatial_upscale",c,t)}
          result=scaled.videoRows
        } else {
          progress("latent_upscale",0,1)
          result=try H3SpatialLatentResize.rows(rows,source:plan.base,target:plan.target,method:resizeMethod!)
        }
        let values=result.asArray(Float.self)
        progress("latent_upscale",1,1)
        return values
      }
    },sampleRefinement:{initial in
      let started=Date()
      defer {onStage?("refinement",Date().timeIntervalSince(started))}
      Stream.gpu.synchronize();Memory.clearCache()
      progress("upscaler_weights_released",1,1)
      let sampled=try finalTask.sample(initial:initial,refinement:plan.refinement) {stage,c,t in
        stageProgress("refine_",admitted.baseEvaluations,stage,c,t)
      }
      report=sampled.backendReport;return sampled.rows
    },decode:{rows in
      H3SampledLatents(video:rows.video,audio:rows.audio,backendReport:report)
    })
  }
}
