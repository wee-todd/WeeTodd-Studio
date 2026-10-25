/// Stage ordering shared by real execution and small model-free contract tests.
public enum H3TwoStageExecution {
  public static func run<Output>(plan: H3TwoStagePlan,
    preflight: () throws -> Void,
    sampleBase: () throws -> H3JointLatentArtifact.Rows,
    upscaleVideo: ([Float]) throws -> [Float],
    sampleRefinement: (H3JointLatentArtifact.Rows) throws -> H3JointLatentArtifact.Rows,
    decode: (H3JointLatentArtifact.Rows) throws -> Output,
    checkCancellation: () throws -> Void = {try Task.checkCancellation()}) throws -> Output {
    try preflight()
    try checkCancellation()
    var base: H3JointLatentArtifact.Rows? = try sampleBase()
    try base!.validate(geometry: plan.base)
    try checkCancellation()
    let audio = base!.audio
    let enlarged = try upscaleVideo(base!.video)
    base = nil
    let initialized = H3JointLatentArtifact.Rows(video: enlarged, audio: audio)
    try initialized.validate(geometry: plan.target)
    try checkCancellation()
    let refined = try sampleRefinement(initialized)
    try refined.validate(geometry: plan.target)
    try checkCancellation()
    return try decode(.init(video: refined.video, audio: audio))
  }
}
