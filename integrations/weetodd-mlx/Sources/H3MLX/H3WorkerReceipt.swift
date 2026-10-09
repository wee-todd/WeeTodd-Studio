import Foundation

/// Scalar snapshot of transformer dispatch, independent of weighted stage ownership.
/// Counts describe actual selected projection outputs and completed VSA consumers.
public struct H3BackendReport: Sendable, Equatable {
  public let samplingAllocationPool: H3SamplingAllocationReport?
  public let transformerWeightCache:H3TransformerCacheReport?
  public let sol: H3SolReport?
  public let nativeBlocks: H3NativeBlockReport?
  public let preparedWindowSize: Int?
  public let eligibleProjectionCalls: Int
  public let mppProjectionCalls: Int
  public let knownFallbackProjectionCalls: Int
  public let firstUseReferenceProjectionCalls: Int
  public let verifiedMPPSignatures: Int
  public let rejectedMPPSignatures: Int
  public let vsaOriginalRowsIndexedCalls: Int
  public let vsaGroupedSparseCalls: Int
  public let vsaDenseCalls: Int

  public init(transformerWeightCache:H3TransformerCacheReport? = nil, preparedWindowSize: Int? = nil, eligibleProjectionCalls: Int = 0,
    mppProjectionCalls: Int = 0, knownFallbackProjectionCalls: Int = 0,
    firstUseReferenceProjectionCalls: Int = 0, verifiedMPPSignatures: Int = 0,
    rejectedMPPSignatures: Int = 0, vsaOriginalRowsIndexedCalls: Int = 0,
    vsaGroupedSparseCalls: Int = 0, vsaDenseCalls: Int = 0,
    sol: H3SolReport? = nil, nativeBlocks: H3NativeBlockReport? = nil,
    samplingAllocationPool: H3SamplingAllocationReport? = nil) {
    self.transformerWeightCache = transformerWeightCache
    self.samplingAllocationPool = samplingAllocationPool
    self.sol = sol
    self.nativeBlocks = nativeBlocks
    self.preparedWindowSize = preparedWindowSize
    self.eligibleProjectionCalls = eligibleProjectionCalls
    self.mppProjectionCalls = mppProjectionCalls
    self.knownFallbackProjectionCalls = knownFallbackProjectionCalls
    self.firstUseReferenceProjectionCalls = firstUseReferenceProjectionCalls
    self.verifiedMPPSignatures = verifiedMPPSignatures
    self.rejectedMPPSignatures = rejectedMPPSignatures
    self.vsaOriginalRowsIndexedCalls = vsaOriginalRowsIndexedCalls
    self.vsaGroupedSparseCalls = vsaGroupedSparseCalls
    self.vsaDenseCalls = vsaDenseCalls
  }

  public var metadata: [String: Any] {
    var result: [String: Any] = ["preparedWindowSize": preparedWindowSize ?? 0,
      "preparedWindowPolicy": preparedWindowSize == nil ? "unprepared" : "bounded",
      "projectionBackend": nativeBlocks != nil ? "nnc_experimental" : (mppProjectionCalls > 0 ? "mlx_with_verified_mpp" : "mlx"),
      "mppEligibleProjectionCalls": eligibleProjectionCalls,
      "mppProjectionCalls": mppProjectionCalls,
      "mppKnownFallbackProjectionCalls": knownFallbackProjectionCalls,
      "mppFirstUseReferenceProjectionCalls": firstUseReferenceProjectionCalls,
      "mppVerifiedSignatures": verifiedMPPSignatures,
      "mppRejectedSignatures": rejectedMPPSignatures,
      "mppVerificationPolicy": "first-use geometry bitwise BF16 equality; empirical eligibility sample, not a universal proof for later activations or weights; full-media qualification is separate",
      "mppCallCountScope": "eligible projection dispatches; known fallback excludes first-use reference outputs; ineligible projections use MLX and are outside these counts",
      "vsaOriginalRowsIndexedCalls": vsaOriginalRowsIndexedCalls,
      "vsaGroupedSparseCalls": vsaGroupedSparseCalls,
      "vsaDenseCalls": vsaDenseCalls,
      "vsaCallCountScope": "completed VSA consumers; ordinary dense transformer attention is outside these counts"]
    if let transformerWeightCache { result["transformerWeightCache"] = transformerWeightCache.metadata }
    if let samplingAllocationPool { result["samplingAllocationPool"] = samplingAllocationPool.metadata }
    if let sol { result["solAttention"] = sol.metadata }
    if let nativeBlocks {
      result["nativeBlocks"] = nativeBlocks.metadata
      result["allocationCounterScope"] = "MLX allocation counters cover the parent; their scope excludes the NNC child; physical observations are reported separately"
    }
    return result
  }
}

/// Terminal handoff to Studio and headless hosts. Keep the original request
/// identity on the final event as well as in the saved metadata.
public enum H3WorkerReceipt {
  /// Decode callbacks count raw frames, including continuation context. Both
  /// previews and stage callbacks publish the cropped frame clock and share
  /// the previous fraction so interleaved events cannot move progress back.
  public static func progress(stage: String, completed: Int, total: Int,
    evaluations: Int, publishedFrames: Int, overlapFrames: Int,
    previousFraction: Double) -> (completed: Int, total: Int, fraction: Double) {
    let previous = previousFraction.isFinite ? max(0, min(0.995, previousFraction)) : 0
    var count = completed, limit = total
    if stage == "video_decode" {
      limit = max(0, publishedFrames)
      count = min(limit, max(0, max(0, completed) - max(0, overlapFrames)))
    }
    let part = limit > 0 ? Double(min(limit, max(0, count))) / Double(limit) : 0
    var next = previous
    if let sampling = H3WorkerProgress.fraction(stage: stage, completed: completed,
      total: total, evaluations: evaluations) { next = sampling }
    else if stage == "control_video_encode" { next = 0.01 + 0.01 * part }
    else if stage == "text" { next = 0.02 + 0.04 * part }
    else if stage == "reference_video_weights_released" { next = 0.08 }
    else if stage == "video_decode" { next = 0.84 + 0.13 * part }
    else if stage == "audio_decode" { next = 0.97 + 0.02 * part }
    return (count, limit, max(previous, min(0.995, next)))
  }

  public static func renderResult(video: URL, metadata: [String: Any],
    jobID: UUID) -> [String: Any] {
    var result:[String:Any]=["video": video.path, "metadata": metadata,
      "jobID": jobID.uuidString, "nativeRuntime": "swift-mlx",
      "productionQualified": false]
    if let manifest=metadata["continuationManifest"] as? String,
      let digest=metadata["continuationManifestSHA256"] as? String,
      let payload=metadata["continuationPayloadSHA256"] as? String,
      let frames=metadata["frames"] as? Int,frames>0,
      let rate=metadata["fps"] as? NSNumber,rate.doubleValue.isFinite,rate.doubleValue>0 {
      let fps=rate.doubleValue
      result["continuation_artifact"]=["manifest":manifest,"manifest_sha256":digest,
        "payload_sha256":payload,"payload_filename":"latents.f32"]
      result["usable_source_in"]=0.0;result["usable_duration"]=Double(frames)/fps
      result["use_complete_duration"]=true
    }
    if metadata["task"] as? String == "motion_fidelity",
      let frames=metadata["frames"] as? Int,frames>0,
      let rate=metadata["fps"] as? NSNumber,rate.doubleValue.isFinite,rate.doubleValue>0 {
      result["usable_source_in"]=0.0;result["usable_duration"]=Double(frames)/rate.doubleValue
      result["use_complete_duration"]=true
    }
    return result
  }
}
