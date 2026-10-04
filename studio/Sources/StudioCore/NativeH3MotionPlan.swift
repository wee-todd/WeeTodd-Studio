import CryptoKit
import Darwin
import Foundation

/// Editor admission for task-bound native latent continuation: T2VA v2 and FL2VA v3.
/// Model identity and latent-row decoding remain owned by the shared worker.
struct NativeH3MotionPlan {
  let contract: [String:Any]
  let duration: Double
  let publishedFrames: Int
  let dependency: [String:Any]
  let warnings: [String]

  private static func read(_ path:String,limit:Int) throws -> Data {
    let descriptor=Darwin.open(path,O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK)
    guard descriptor>=0 else { throw StudioError.invalid("Relink the native H3 continuation artifact.") }
    let handle=FileHandle(fileDescriptor:descriptor,closeOnDealloc:true);defer { try? handle.close() }
    var status=stat()
    guard fstat(descriptor,&status)==0,status.st_mode & S_IFMT == S_IFREG,
      status.st_size>0,status.st_size<=limit else { throw StudioError.invalid("Invalid bounded H3 continuation artifact.") }
    let bytes=try handle.read(upToCount:limit+1) ?? Data()
    guard bytes.count==status.st_size else { throw StudioError.invalid("H3 continuation artifact changed while reading.") }
    return bytes
  }
  private static func hash(_ bytes:Data) -> String {
    SHA256.hash(data:bytes).map { String(format:"%02x",$0) }.joined()
  }
  init?(project:StudioProject,clip:Clip) throws {
    let saving=project.shouldSaveContinuityContext(for:clip),motion=clip.continuityMode == "motion"
    guard saving || motion else { return nil }
    let fl2va=["i2v","fflf"].contains(clip.inferredTask),reference=["ref2va","a2v"].contains(clip.inferredTask)
    let task=fl2va ? "fl2va" : reference ? "ref2va" : "t2va"
    let roles:Set<MediaRole>=fl2va ? [.first,.last,.keyframe,.lora]
      : clip.inferredTask == "a2v" ? [.audioDriver,.first,.last,.keyframe,.lora]
      : reference ? [.reference,.lora] : [.lora]
    guard ["t2v","t2va","i2v","fflf","ref2va","a2v"].contains(clip.inferredTask),
      clip.attachments.allSatisfy({ roles.contains($0.role) }),
      clip.duration.isFinite,(2.5...(saving && !motion ? 362.0/24 : 15)).contains(clip.duration),
      (32...1920).contains(clip.generationWidth),(32...1920).contains(clip.generationHeight),
      clip.generationWidth%32 == 0,clip.generationHeight%32 == 0 else {
      throw StudioError.invalid("Swift H3 motion context requires text-to-video, timed-image FL2VA or Ref2VA/A2V, matching canvases, and compatible task attachments.")
    }
    var context=22,sourcePath:String?,sourceHash:String?,dependency:[String:Any]=[:]
    if motion {
      let issues=project.continuityIssues(for:clip)
      guard issues.isEmpty,let source=try project.continuitySource(for:clip),
        let artifact=source.activeRenderVersion?.continuationArtifact,
        artifact.payloadFilename == "latents.f32" else {
        throw StudioError.invalid(issues.first ?? "Render and accept a Swift H3 source with Save motion context first; Python context files are incompatible.")
      }
      guard artifact.manifest.hasPrefix("/"),!artifact.manifest.utf8.contains(0) else {
        throw StudioError.invalid("Relink the native H3 continuation manifest.")
      }
      let bytes=try Self.read(artifact.manifest,limit:1024*1024)
      guard Self.hash(bytes)==artifact.manifestSHA256,
        let manifest=try JSONSerialization.jsonObject(with:bytes) as? [String:Any],
        Set(manifest.keys).subtracting(["task"])==["format","contextFrames","width","height","generatedFrames","publishedFrames",
          "overlapFrames","identity","payloadBytes","payloadSHA256"],
        manifest["format"] as? String == "weetodd-h3-swift-continuation-v2",
        (manifest["task"] == nil || manifest["task"] as? String != nil),
        (manifest["task"] as? String ?? "t2va") == task,
        let count=manifest["contextFrames"] as? Int,[5,22,39,56].contains(count),
        manifest["width"] as? Int == clip.generationWidth,manifest["height"] as? Int == clip.generationHeight,
        let generated=manifest["generatedFrames"] as? Int,(60...362).contains(generated),generated%17 == 5,
        let published=manifest["publishedFrames"] as? Int,
        let overlap=manifest["overlapFrames"] as? Int,generated-overlap==published,
        let payloadBytes=manifest["payloadBytes"] as? Int,
        let payloadHash=manifest["payloadSHA256"] as? String,payloadHash==artifact.payloadSHA256 else {
        throw StudioError.invalid("Swift H3 source context identity, canvas, or timing changed. Prepare the source again.")
      }
      let expected=(((count-5)/17)*5+2)*(clip.generationWidth/32)*(clip.generationHeight/32)*96*4
        + 2*Int((Double(count)/24*40).rounded(.toNearestOrEven))*32*4
      guard payloadBytes==expected,expected<=64*1024*1024 else { throw StudioError.invalid("Swift H3 source payload geometry changed.") }
      let payload=try Self.read(URL(fileURLWithPath:artifact.manifest).deletingLastPathComponent().appendingPathComponent(artifact.payloadFilename).path,limit:64*1024*1024)
      guard payload.count==expected,Self.hash(payload)==payloadHash else {
        throw StudioError.invalid("Swift H3 source latent payload changed. Prepare the source again.")
      }
      context=count;sourcePath=artifact.manifest;sourceHash=artifact.manifestSHA256
      dependency=["sourceClipID":source.id.uuidString,"sourcePath":source.sourcePath,
        "sourceTakeID":source.activeRenderVersion!.id.uuidString,"sourceContext":artifact.manifest,
        "sourceManifestSHA256":artifact.manifestSHA256,"sourcePayloadSHA256":payloadHash]
    }
    let requested=Int((min(clip.duration,15)*24).rounded(.toNearestOrEven)),overlap=motion ? context : 0
    var generated=requested+overlap
    while generated%17 != 5 { generated+=1 }
    guard generated<=362,requested>context else {
      throw StudioError.invalid("H3 continuation plus context exceeds the 362-frame window. Shorten the visible clip.")
    }
    let published=motion && !saving ? requested : generated-overlap
    duration=motion && saving ? Double(published)/24 : min(clip.duration,15)
    publishedFrames=published
    var fields:[String:Any]=["version":reference ? 4 : fl2va ? 3 : 2,"context_frames":context,"save_context":saving]
    if let sourcePath,let sourceHash { fields["source_context"]=sourcePath;fields["source_manifest_sha256"]=sourceHash }
    contract=fields
    dependency["mode"]=motion ? "motion" : "independent";dependency["engine"]="h3"
    dependency["saveContext"]=saving;dependency["contextFrames"]=context
    dependency["publishedFrames"]=published;self.dependency=dependency
    warnings=saving && abs(Double(published)/24-clip.duration)>0.5/24
      ? ["Saving H3 motion context keeps the complete terminal frame grid: \(published) frames (\(String(format:"%.3f",Double(published)/24)) seconds at 24 fps). The accepted clip uses this duration."] : []
  }
}
