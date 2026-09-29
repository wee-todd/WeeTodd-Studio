import Foundation

/// Terminal handoff to Studio and headless hosts. Keep the original request
/// identity on the final event as well as in the saved metadata.
public enum H3WorkerReceipt {
  public static func renderResult(video: URL, metadata: [String: Any],
    jobID: UUID) -> [String: Any] {
    ["video": video.path, "metadata": metadata,
      "jobID": jobID.uuidString, "nativeRuntime": "swift-mlx",
      "productionQualified": false]
  }
}
