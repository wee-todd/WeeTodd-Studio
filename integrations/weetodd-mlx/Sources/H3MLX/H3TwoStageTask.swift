import Foundation

/// The same task-specific samplers used by ordinary Studio and headless jobs.
public enum H3TwoStageTask {
  case text(H3T2VARequest), keyframes(H3FL2VARequest), references(H3Ref2VAStillRequest)
  public var geometry:H3Geometry {switch self {case .text(let r):r.geometry;case .keyframes(let r):r.base.geometry;case .references(let r):r.geometry}}
  public var adapters:[H3LoRAAdapter] {switch self {case .text(let r):r.loRAAdapters;case .keyframes(let r):r.base.loRAAdapters;case .references(let r):r.loRAAdapters}}
  var requestedSteps:Int {switch self {case .text(let r):r.requestedSteps;case .keyframes(let r):r.base.requestedSteps;case .references(let r):r.requestedSteps}}
  var samplingMethod:H3SamplingMethod {switch self {case .text(let r):r.samplingMethod;case .keyframes(let r):r.base.samplingMethod;case .references(let r):r.samplingMethod}}
  var decoder:(video:URL,audio:URL,memory:H3VideoDecodeMemoryMode?,precision:H3VideoDecodePrecision) {
    switch self {
    case .text(let r):return(r.videoVAE,r.audioVAE,r.videoDecodeMemoryMode,r.videoDecodePrecision)
    case .keyframes(let r):return(r.base.videoVAE,r.base.audioVAE,r.base.videoDecodeMemoryMode,r.base.videoDecodePrecision)
    case .references(let r):return(r.videoVAE,r.audioVAE,r.videoDecodeMemoryMode,r.videoDecodePrecision)
    }
  }
  func replacing(geometry:H3Geometry,adapters:[H3LoRAAdapter],draftKeyframes:[H3StillReference]? = nil) throws -> Self {
    func base(_ r:H3T2VARequest) throws -> H3T2VARequest {
      guard r.funControl==nil,r.vdn==nil,r.fastVariant==nil,r.transformerWeightCacheGB==0 else {
        throw H3CheckpointError.invalid("Automatic H3 upscale cannot combine Fun control, VDN, FastH3 or weight cache.")
      }
      return try H3T2VARequest(prompt:r.prompt,width:geometry.width,height:geometry.height,
        durationSeconds:r.durationSeconds,seed:r.seed,requestedSteps:r.requestedSteps,
        transformer:r.transformer,qwenPages:r.qwenPages,tokenizer:r.tokenizer,videoVAE:r.videoVAE,audioVAE:r.audioVAE,
        loRAAdapters:adapters,videoDecodeMemoryMode:r.videoDecodeMemoryMode,videoDecodePrecision:r.videoDecodePrecision,
        samplingMethod:r.samplingMethod,canvasAdmission:geometry.canvasAdmission)
    }
    switch self {
    case .text(let r):return .text(try base(r))
    case .keyframes(let r):
      guard r.attentionPolicy == .dense else {throw H3CheckpointError.invalid("Automatic H3 upscale requires dense attention.")}
      return .keyframes(try H3FL2VARequest(base:base(r.base),vision:r.vision,images:draftKeyframes ?? r.images,
        anchors:r.anchors,referenceNoise:r.referenceNoise,attentionPolicy:r.attentionPolicy))
    case .references(let r):
      guard r.attentionPolicy == .dense,r.transformerBackend == .mlx,r.transformerWeightCacheGB==0 else {
        throw H3CheckpointError.invalid("Automatic H3 upscale requires dense MLX reference sampling without a weight cache.")
      }
      return .references(try H3Ref2VAStillRequest(prompt:r.prompt,mediaReferences:r.references,
        width:geometry.width,height:geometry.height,durationSeconds:min(Double(geometry.frames)/24,15),
        seed:r.seed,requestedSteps:r.requestedSteps,transformer:r.transformer,qwenPages:r.qwenPages,qwenVision:r.qwenVision,
        tokenizer:r.tokenizer,videoVAE:r.videoVAE,audioVAE:r.audioVAE,loRAAdapters:adapters,
        videoDecodeMemoryMode:r.videoDecodeMemoryMode,videoDecodePrecision:r.videoDecodePrecision,
        samplingMethod:r.samplingMethod,referenceNoise:r.referenceNoise,canvasAdmission:geometry.canvasAdmission))
    }
  }
  func preflight(refinement:H3JointRefinement? = nil) throws -> (rows:Int,evaluations:Int) {
    switch self {
    case .text(let r):let a=try H3T2VARunner.preflight(r,refinement:refinement);return(a.packedRows,a.evaluations)
    case .keyframes(let r):let a=try H3FL2VARunner.preflight(r,refinement:refinement);return(a.packedRows,a.evaluations)
    case .references(let r):let a=try H3Ref2VAStillRunner.preflight(r,refinement:refinement);return(a.packedRows,a.evaluations)
    }
  }
  func sample(initial:H3JointLatentArtifact.Rows? = nil,refinement:H3JointRefinement? = nil,
    progress:(String,Int,Int)->Void) throws -> H3SampledLatents {
    switch self {
    case .text(let r):try H3T2VARunner.sample(r,initialRows:initial,refinement:refinement,progress:progress)
    case .keyframes(let r):try H3FL2VARunner.sample(r,initialRows:initial,refinement:refinement,progress:progress)
    case .references(let r):try H3Ref2VAStillRunner.sample(r,initialRows:initial,refinement:refinement,progress:progress)
    }
  }
}
