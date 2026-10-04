import AVFoundation
import Darwin
import Foundation
import LTX25Engine

/// Durable native chunks. The executor runs each missing chunk exactly once;
/// only verified completed media are reused. Failure never publishes a partial
/// chunk and never erases previously completed chunks.
public enum MLXMovieChunkCoordinator {
  public struct Binding:Codable,Equatable,Sendable {
    public let contractSHA256:String,sourceSHA256:String,rgbSHA256:String,audioSHA256:String,componentIdentitySHA256:String,workerSHA256:String
    public init(request:MLXMovieUpscaleRequest,audioSHA256:String,componentIdentitySHA256:String,workerSHA256:String) throws {
      guard [audioSHA256,componentIdentitySHA256,workerSHA256].allSatisfy(MLXMovieUpscaleRequest.validSHA) else {
        throw LTXError.invalid("Movie resume needs exact publication, component-header and renderer identities.")
      }
      contractSHA256=request.contractSHA256;sourceSHA256=request.source.sha256;rgbSHA256=request.source.rgbSHA256
      self.audioSHA256=audioSHA256;self.componentIdentitySHA256=componentIdentitySHA256;self.workerSHA256=workerSHA256
    }
  }
  public struct Receipt:Codable,Sendable {
    public let version:Int,binding:Binding,index:Int,startFrame:Int,endFrame:Int,paddedFrames:Int,width:Int,height:Int
    public let fps:Double,videoSHA256:String
  }
  public struct Completed:Sendable { public let video:URL,receipt:Receipt,reused:Bool }
  public static func execute(request:MLXMovieUpscaleRequest,chunks:[MLXMovieUpscalePlan.Chunk],binding:Binding,
    directory:URL,progress:(String,Int,Int)throws->Void={ _,_,_ in },
    render:(MLXMovieUpscalePlan.Chunk,Int,URL) async throws -> Void) async throws -> [Completed] {
    guard binding.contractSHA256==request.contractSHA256,binding.sourceSHA256==request.source.sha256,
      binding.rgbSHA256==request.source.rgbSHA256,directory.isFileURL,
      !chunks.isEmpty,chunks.count<=20408,chunks.first?.startFrame==0,chunks.last?.endFrame==request.plan.frames,
      chunks.allSatisfy({ $0.frames>0 && (try? MLXMovieUpscalePlan.paddedFrameCount($0.frames)) == $0.paddedFrames }),
      zip(chunks,chunks.dropFirst()).allSatisfy({ $0.endFrame==$1.startFrame }) else {
      throw LTXError.invalid("Movie chunks need exact disjoint source coverage and the frozen execution identity.")
    }
    try request.validateMediaSources();try Task.checkCancellation()
    let sourcePaths=[request.source.path,request.source.rgbPath]+request.referenceImageSHA256.keys.sorted()+(request.audioSource.map { [$0.path] } ?? [])
    let sourceIdentities=try sourcePaths.map { try MLXMovieFiles.Identity(URL(fileURLWithPath:$0)) }
    func checkSources() throws {
      try Task.checkCancellation()
      guard try sourcePaths.map({ try MLXMovieFiles.Identity(URL(fileURLWithPath:$0)) })==sourceIdentities else {
        throw LTXError.invalid("Movie frozen source changed during chunk execution.")
      }
    }
    let fm=FileManager.default
    if fm.fileExists(atPath:directory.path) {
      guard request.resume else { throw LTXError.invalid("Movie chunks already exist; explicitly select resume.") }
      let values=try directory.resourceValues(forKeys:[.isDirectoryKey,.isSymbolicLinkKey])
      guard values.isDirectory==true,values.isSymbolicLink != true else { throw LTXError.invalid("Movie chunk root must be a real directory.") }
    } else { try fm.createDirectory(at:directory,withIntermediateDirectories:true) }
    let lock=directory.appendingPathComponent(".execution.lock"),fd=Darwin.open(lock.path,O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC,0o600)
    guard fd>=0 else { throw LTXError.invalid("Another movie chunk execution owns this output, or an interrupted lock needs inspection.") }
    Darwin.close(fd);defer { try? fm.removeItem(at:lock) }
    var completed:[Completed]=[]
    for (index,chunk) in chunks.enumerated() {
      try checkSources()
      let name=String(format:"chunk-%06d",index),output=directory.appendingPathComponent(name),video=output.appendingPathComponent("video.mp4")
      let receiptURL=output.appendingPathComponent("chunk.json")
      if fm.fileExists(atPath:output.path) {
        guard request.resume else { throw LTXError.invalid("Movie chunk reuse requires explicit resume.") }
        let values=try output.resourceValues(forKeys:[.isDirectoryKey,.isSymbolicLinkKey])
        guard values.isDirectory==true,values.isSymbolicLink != true else { throw LTXError.invalid("Movie chunk escapes its output directory.") }
        let bytes=try MLXMovieFiles.metadata(receiptURL,maximumBytes:65536)
        let receipt=try JSONDecoder().decode(Receipt.self,from:bytes)
        guard receipt.version==1,receipt.binding==binding,receipt.index==index,
          receipt.startFrame==chunk.startFrame,receipt.endFrame==chunk.endFrame,receipt.paddedFrames==chunk.paddedFrames,
          receipt.width==request.plan.size.outputWidth,receipt.height==request.plan.size.outputHeight,receipt.fps==request.plan.fps else {
          throw LTXError.invalid("Completed movie chunk belongs to a different source, renderer or generation recipe.")
        }
        try await validate(video,frames:chunk.frames,width:receipt.width,height:receipt.height,fps:receipt.fps,sha256:receipt.videoSHA256)
        completed.append(Completed(video:video,receipt:receipt,reused:true))
        try progress("movie_chunk_reused",index+1,chunks.count)
        continue
      }
      let staging=directory.appendingPathComponent(".chunk-"+UUID().uuidString)
      try fm.createDirectory(at:staging,withIntermediateDirectories:false)
      do {
        try progress("movie_chunk_started",index,chunks.count)
        try await render(chunk,index,staging);try Task.checkCancellation()
        let stagedVideo=staging.appendingPathComponent("video.mp4"),sha=try MLXMovieFiles.digest(stagedVideo)
        try await validate(stagedVideo,frames:chunk.frames,width:request.plan.size.outputWidth,height:request.plan.size.outputHeight,fps:request.plan.fps,sha256:sha)
        try checkSources()
        let receipt=Receipt(version:1,binding:binding,index:index,startFrame:chunk.startFrame,endFrame:chunk.endFrame,
          paddedFrames:chunk.paddedFrames,width:request.plan.size.outputWidth,height:request.plan.size.outputHeight,fps:request.plan.fps,videoSHA256:sha)
        let encoder=JSONEncoder();encoder.outputFormatting=[.sortedKeys]
        try encoder.encode(receipt).write(to:staging.appendingPathComponent("chunk.json"),options:.withoutOverwriting)
        try fm.moveItem(at:staging,to:output)
        completed.append(Completed(video:video,receipt:receipt,reused:false))
        try progress("movie_chunk_completed",index+1,chunks.count)
      } catch { try? fm.removeItem(at:staging);throw error }
    }
    try checkSources()
    return completed
  }
  private static func validate(_ url:URL,frames:Int,width:Int,height:Int,fps:Double,sha256:String) async throws {
    guard MLXMovieUpscaleRequest.validSHA(sha256),try MLXMovieFiles.digest(url)==sha256 else {
      throw LTXError.invalid("Completed movie chunk bytes changed.")
    }
    let clock=try await MLXMovieFiles.videoClock(url,maximumFrames:frames)
    guard clock.width==width,clock.height==height,clock.times.count==frames,abs(clock.fps-fps)<0.001,
      abs(clock.times[0])<0.05/fps,try await AVURLAsset(url:url).loadTracks(withMediaType:.audio).isEmpty else {
      throw LTXError.invalid("Completed movie chunk is not the exact silent source interval/canvas/cadence.")
    }
  }
}
