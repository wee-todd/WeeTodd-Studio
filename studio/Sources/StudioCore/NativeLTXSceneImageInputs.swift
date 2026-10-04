import Foundation

/// Metadata-only scene image producer. Original clips remain unchanged; only
/// their independent transport copies omit images while common settings resolve.
enum NativeLTXSceneImageInputs {
  static let roles:Set<MediaRole>=[.first,.last,.keyframe]
  static func requiresVersionTwo(_ members:[Clip]) -> Bool {
    members.contains { clip in
      clip.generationSelection?.ltx25Keyframes != nil ||
        clip.attachments.contains { [.last,.keyframe].contains($0.role) } ||
        clip.attachments.filter { roles.contains($0.role) }.count>1
    }
  }
  static func baseProject(_ project:StudioProject,members:[Clip]) throws -> StudioProject {
    var result=project
    for member in members {
      try Task.checkCancellation()
      let images=member.attachments.filter { roles.contains($0.role) }
      let audio=member.attachments.contains { $0.role == .audioDriver }
      let task=member.generationSelection?.task ?? member.inferredTask
      guard ["t2v","i2v","fflf","a2v"].contains(task),audio == (task == "a2v"),
        images.isEmpty ? ["t2v","a2v"].contains(task) : ["i2v","fflf","a2v"].contains(task) else {
        throw StudioError.invalid("Scene task must match its image and continuous source-audio inputs.")
      }
      if let policy=member.generationSelection?.ltx25Keyframes {
        guard policy.experimentalEnabled,policy.generatedCount==0 else {
          throw StudioError.invalid("Continuous scenes accept timed images, not generated keyframe slots.")
        }
      }
      guard let index=result.clips.firstIndex(where: { $0.id==member.id }) else {
        throw StudioError.invalid("A scene member is missing from the project.")
      }
      result.clips[index].continuity=ClipContinuity(mode:"independent")
      result.clips[index].attachments.removeAll { roles.contains($0.role) }
      var selection=result.clips[index].generationSelection ?? GenerationSelection()
      selection.task=audio ? "a2v":"t2v";selection.ltx25Keyframes=nil
      result.clips[index].generationSelection=selection
    }
    return result
  }
  static func make(members:[Clip],assets:[MediaAsset],segmentStarts:[Int],segmentFrames:[Int],fps:Double) throws -> [[String:Any]] {
    guard members.count==segmentStarts.count,members.count==segmentFrames.count,
      !members.isEmpty,fps.isFinite,(1...120).contains(fps),
      segmentStarts.allSatisfy({ (0...3600).contains($0) }),segmentFrames.allSatisfy({ (1...3600).contains($0) }),
      segmentStarts.last!+segmentFrames.last!<=3600,
      segmentStarts[0]==0,
      zip(segmentStarts,segmentFrames).dropLast().enumerated().allSatisfy({ index,pair in
        pair.0+pair.1==segmentStarts[index+1]
      }) else { throw StudioError.invalid("Scene images require an exact quantized member frame plan.") }
    var byFrame:[Int:[String:Any]]=[:],ids=Set<String>()
    for (index,member) in members.enumerated() {
      for attachment in member.attachments where roles.contains(attachment.role) {
        try Task.checkCancellation()
        guard let asset=assets.last(where: { $0.id==attachment.assetID }),asset.kind == .image,
          attachment.strength.isFinite,(0...1).contains(attachment.strength),
          attachment.time.isFinite,attachment.time>=0,attachment.time<=member.duration,
          attachment.time*fps<Double(Int.max),asset.path.hasPrefix("/"),
          asset.path.utf8.count<=4096,!asset.path.utf8.contains(0) else {
          throw StudioError.invalid("Scene keyframes require linked images, finite strengths and bounded times.")
        }
        let url=URL(fileURLWithPath:asset.path).standardizedFileURL.resolvingSymlinksInPath()
        let properties=try url.resourceValues(forKeys:[.isRegularFileKey,.fileSizeKey])
        guard properties.isRegularFile == true,let bytes=properties.fileSize,bytes>0,bytes<=64*1024*1024,
          FileManager.default.isReadableFile(atPath:url.path) else {
          throw StudioError.invalid("Relink the scene image as a readable regular file of at most 64 MiB.")
        }
        let local=attachment.role == .last ? segmentFrames[index]-1 :
          attachment.role == .first ? 0 : Int((attachment.time*fps).rounded(.toNearestOrEven))
        guard (0..<segmentFrames[index]).contains(local) else {
          throw StudioError.invalid("Timed scene images must lie within their resolved member range; use Last frame for the terminal endpoint.")
        }
        let frame=segmentStarts[index]+local,id=member.id.uuidString+":"+attachment.id.uuidString
        guard ids.insert(id).inserted else { throw StudioError.invalid("Scene attachment identities must be unique.") }
        if let prior=byFrame[frame] {
          guard prior["path"] as? String==url.path,prior["strength"] as? Double==attachment.strength else {
            throw StudioError.invalid("Conflicting scene images target the same global frame.")
          }
          continue
        }
        byFrame[frame]=["id":id,"kind":"image","role":"keyframe","path":url.path,
          "strength":attachment.strength,"frame_index":frame]
      }
    }
    guard byFrame.count<=32 else { throw StudioError.invalid("LTX scenes support at most 32 global image anchors.") }
    return byFrame.keys.sorted().map { byFrame[$0]! }
  }
}
