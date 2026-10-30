import Foundation

public struct VideoProcessingAcceptanceToken:Sendable {
  public let projectID:UUID,clipID:UUID,clipFingerprint:String
  public init(project:StudioProject,clipID:UUID) throws {
    guard let clip=project.clips.first(where:{$0.id==clipID}) else{throw StudioError.invalid("The processing destination clip no longer exists.")}
    self.projectID=project.id;self.clipID=clipID
    let encoder=JSONEncoder();encoder.outputFormatting=[.sortedKeys]
    clipFingerprint=NativeHeadlessJob.hash(try encoder.encode(clip))
  }
  func verify(project:StudioProject,clipID:UUID) throws {
    let current=try Self(project:project,clipID:clipID)
    guard current.projectID==projectID,current.clipID==self.clipID,current.clipFingerprint==clipFingerprint else {
      throw StudioError.invalid("The destination document or clip changed. The completed sequence remains in its cache.")
    }
  }
}

/// Accepted restored takes retain the editorial interval and point back to the immutable original.
public enum VideoProcessingAcceptance {
  public static func apply(_ result: NativeVideoProcessingResult, to clipID: UUID,
    project: inout StudioProject,expected:VideoProcessingAcceptanceToken? = nil) throws {
    try expected?.verify(project:project,clipID:clipID)
    guard let index = project.clips.firstIndex(where: { $0.id == clipID }),
      project.clips[index].videoProcessing == result.recipe else {
      throw StudioError.invalid("The clip processing recipe changed. The completed upscale remains in its cache.")
    }
    let clip = project.clips[index]
    let selected=URL(fileURLWithPath:clip.sourcePath).resolvingSymlinksInPath().path
    let prior=clip.versions.last(where:{$0.path==clip.sourcePath})?.processingLineage?.source
    guard selected==result.source.path || prior==result.source else {
      throw StudioError.invalid("The selected source take changed. The completed sequence remains in its cache.")
    }
    let original = clip.versions.last { URL(fileURLWithPath: $0.path).resolvingSymlinksInPath().path == result.source.path }
    let lineage = try result.lineage(sourceVersionID: original?.id)
    if let existing = clip.versions.last(where: { $0.processingLineage == lineage && $0.path == result.receipt.video.path }) {
      project.clips[index].sourcePath = existing.path
      return
    }
    var version = original ?? RenderVersion(path: result.source.path, seed: clip.seed, prompt: clip.prompt,
      recipePath: "", usableSourceIn: clip.sourceIn, usableDuration: clip.duration)
    version.id = UUID(); version.created = Date(); version.path = result.receipt.video.path
    version.processingLineage = lineage
    // Latents belong to the generation, not the restored RGB movie.
    version.jointLatentArtifact = nil; version.continuationArtifact = nil
    if let scene = version.sceneMembers {
      version.sceneTakeID = project.clips.flatMap(\.versions).first {
        $0.path == version.path && $0.sceneMembers == scene
      }?.sceneTakeID ?? UUID()
    }
    project.clips[index].versions.append(version); project.clips[index].sourcePath = version.path
    if !project.assets.contains(where: { $0.path == version.path }) {
      var asset = MediaAsset(name: result.receipt.label + " · " + clip.name,
        kind: .video, path: version.path, scope: .project)
      asset.width = result.receipt.width; asset.height = result.receipt.height
      asset.fps = result.receipt.fps.value
      asset.duration = Double(result.receipt.frames) / asset.fps; asset.upscaleReceipt = result.receiptPath
      project.assets.append(asset)
    }
  }
}
