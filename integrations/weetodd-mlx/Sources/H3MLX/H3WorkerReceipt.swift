import Foundation

/// Terminal handoff to Studio and headless hosts. Keep the original request
/// identity on the final event as well as in the saved metadata.
public enum H3WorkerReceipt {
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
