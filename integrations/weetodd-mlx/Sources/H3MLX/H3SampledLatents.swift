/// Materialized complete AV rows after all transformer weights are released.
/// Latent-only sampling owns no decoder and publishes no media.
public struct H3SampledLatents {
  public let rows: H3JointLatentArtifact.Rows
  public let backendReport: H3BackendReport?
  public init(video: [Float], audio: [Float], backendReport: H3BackendReport? = nil) {
    rows = .init(video: video, audio: audio); self.backendReport = backendReport
  }
}
