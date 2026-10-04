import Foundation

extension ProjectStorage {
  public static func collectJointLatentArtifact(_ artifact:H3JointLatentArtifact,to directory:URL) throws -> H3JointLatentArtifact {
    try artifact.verify()
    let fm=FileManager.default
    guard !fm.fileExists(atPath:directory.path) else {throw StudioError.invalid("Choose a new directory for the complete H3 latent artifact.")}
    try fm.createDirectory(at:directory,withIntermediateDirectories:false)
    var published=false
    defer {if !published {try? fm.removeItem(at:directory)}}
    let manifest=directory.appendingPathComponent("joint-manifest.json")
    try fm.copyItem(at:URL(fileURLWithPath:artifact.manifest),to:manifest)
    try fm.copyItem(at:URL(fileURLWithPath:artifact.payloadPath),to:directory.appendingPathComponent(artifact.payloadFilename))
    var copied=artifact;copied.manifest=manifest.path
    try copied.verify();try artifact.verify();published=true;return copied
  }
  /// Call off the UI actor. Copy each complete artifact once, retaining every
  /// take/current/saved selection reference and its sibling payload identity.
  public static func collectH3CreativeMedia(_ original:StudioProject,to media:URL) throws -> StudioProject {
    var project=original,copied:[String:String]=[:]
    func collect(_ artifact:H3JointLatentArtifact) throws -> H3JointLatentArtifact {
      var value=artifact
      if let known=copied[artifact.manifest] {value.manifest=known;return value}
      let directory=media.appendingPathComponent(UUID().uuidString+"-h3-latents")
      value=try collectJointLatentArtifact(artifact,to:directory)
      value.manifest="Media/"+directory.lastPathComponent+"/joint-manifest.json"
      copied[artifact.manifest]=value.manifest;return value
    }
    for i in project.clips.indices {
      for j in project.clips[i].versions.indices {
        if let source=project.clips[i].versions[j].jointLatentArtifact {
          project.clips[i].versions[j].jointLatentArtifact=try collect(source)
        }
      }
      if let source=project.clips[i].generationSelection?.h3Joint?.refinement?.source {
        project.clips[i].generationSelection?.h3Joint?.refinement?.source=try collect(source)
      }
      if var saved=project.clips[i].savedNativeGenerations {
        for key in Array(saved.keys) {
          if let source=saved[key]?.selection?.h3Joint?.refinement?.source {
            saved[key]?.selection?.h3Joint?.refinement?.source=try collect(source)
          }
        }
        project.clips[i].savedNativeGenerations=saved
      }
    }
    return project
  }
}
