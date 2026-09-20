import Foundation

/// Captures the source frame being edited, independently of the Director's current playhead.
public struct RippleImageContext: Codable, Equatable {
  public var clipID: UUID
  public var referenceID: UUID
  public var sourcePath: String
  public var sourceIn: Double
  public var duration: Double
  public var frameRate: Double
  public var frame: Int
  public var originalPath: String
  public init(clipID: UUID, draft: RippleDraft, reference: RippleReference) {
    self.clipID = clipID; referenceID = reference.id
    sourcePath = draft.sourcePath; sourceIn = draft.sourceIn; duration = draft.duration
    frameRate = draft.frameRate; frame = reference.frame; originalPath = reference.originalPath
  }
  public func matches(_ clip: Clip) -> Bool {
    guard clip.id == clipID, let draft = clip.rippleDraft,
      draft.sourceMatches(clip), draft.sourcePath == sourcePath, draft.sourceIn == sourceIn,
      draft.duration == duration, draft.frameRate == frameRate,
      let reference = draft.references.first(where: { $0.id == referenceID }) else { return false }
    return reference.frame == frame && reference.originalPath == originalPath
  }
}
