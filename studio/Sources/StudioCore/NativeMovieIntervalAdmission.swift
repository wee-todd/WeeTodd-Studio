import Foundation

/// New source-movie trim fields are never interpreted as ordinary reference,
/// keyframe or audio-driver controls. Presence, including zero, is explicit intent.
public enum NativeMovieIntervalAdmission {
  public static func rejectInOrdinaryRequest(_ request:[String:Any]) throws {
    guard let raw=request["project"],let id=request["clipID"] as? String else { return }
    let project=try JSONDecoder().decode(StudioProject.self,from:JSONSerialization.data(withJSONObject:raw))
    guard let clip=project.clips.first(where:{ $0.id.uuidString.caseInsensitiveCompare(id) == .orderedSame }) else { return }
    let members=project.isContinuousSceneMember(clip) ? try project.continuousSceneMembers(for:clip):[clip]
    guard members.allSatisfy({ $0.attachments.allSatisfy { $0.sourceStartSeconds==nil && $0.sourceDurationSeconds==nil } }) else {
      throw StudioError.invalid("Source movie interval fields apply only to the explicit native movie upscale task; ordinary generation cannot ignore them.")
    }
  }
}
