import Foundation
import InferenceContracts
import LTX25Engine

/// Model-free CFR source preparation. The transformed visible RGB stays on
/// disk; causal padding is made separately for each actual execution chunk.
public struct MLXMovieSourceVideo:Sendable {
  public struct Prepared:Sendable {
    public let rgb24:URL,sha256:String,plan:MLXMovieUpscalePlan,sourceFrameRange:Range<Int>
    public let resizePolicy:String
  }
  public let source:URL,sha256:String,plan:MLXMovieUpscalePlan,startSeconds:Double
  public init(source:URL,sha256:String,plan:MLXMovieUpscalePlan,startSeconds:Double=0) throws {
    _ = try NativeMediaSource(path:source.path,sha256:sha256)
    guard source.isFileURL,startSeconds.isFinite,startSeconds>=0 else { throw LTXError.invalid("Movie source needs a frozen absolute path and interval.") }
    self.source=source;self.sha256=sha256;self.plan=plan;self.startSeconds=startSeconds
  }
  public func preflight() async throws -> Range<Int> {
    try NativeMediaSource(path:source.path,sha256:sha256).verify()
    let clock=try await MLXMovieFiles.videoClock(source)
    let first=clock.times[0],position=((startSeconds-first)*plan.fps).rounded(.toNearestOrEven)
    guard position.isFinite,position>=0,position<Double(clock.times.count) else { throw LTXError.invalid("Movie selected start is outside its admitted source frames.") }
    let start=Int(position)
    guard clock.width==plan.size.sourceWidth,clock.height==plan.size.sourceHeight,
      abs(clock.fps-plan.fps)<0.001,start>=0,start<=clock.times.count-plan.frames,
      abs(clock.times[start]-startSeconds)*plan.fps<0.05 else {
      throw LTXError.invalid("Movie source dimensions, cadence or exact selected frame interval changed.")
    }
    return start..<start+plan.frames
  }
  public func prepare(ffmpeg:URL,directory:URL) async throws -> Prepared {
    let range=try await preflight(),fm=FileManager.default
    guard directory.isFileURL,directory.path.hasPrefix("/"),!fm.fileExists(atPath:directory.path),
      plan.size.width<=2048,plan.size.height<=2048 else { throw LTXError.invalid("Movie preparation needs a new directory and an admitted 2× canvas.") }
    try fm.createDirectory(at:directory.deletingLastPathComponent(),withIntermediateDirectories:true)
    let staging=directory.deletingLastPathComponent().appendingPathComponent(".movie-rgb-"+UUID().uuidString)
    try fm.createDirectory(at:staging,withIntermediateDirectories:false);defer { try? fm.removeItem(at:staging) }
    let rgb=staging.appendingPathComponent("visible.rgb24"),size=plan.size
    let operation=size.resized ? "scale=\(size.width):\(size.height):flags=lanczos" :
      "crop=\(size.width):\(size.height):\(size.cropLeft):\(size.cropTop)"
    try MLXMovieFiles.run(ffmpeg,["-v","error","-nostdin","-n","-i",source.path,"-map","0:v:0","-vf",
      "trim=start_frame=\(range.lowerBound):end_frame=\(range.upperBound),setpts=PTS-STARTPTS,format=rgb24,"+operation,
      "-vsync","0","-frames:v",String(plan.frames),"-an","-f","rawvideo","-pix_fmt","rgb24",rgb.path],log:staging.appendingPathComponent("video.log"))
    guard (try MLXMovieFiles.Identity(rgb)).bytes == Int64(plan.frames)*Int64(size.width)*Int64(size.height)*3 else {
      throw LTXError.invalid("Processed movie RGB differs from the visible source geometry.")
    }
    let digest=try MLXMovieFiles.digest(rgb)
    try NativeMediaSource(path:source.path,sha256:sha256).verify();try Task.checkCancellation()
    try fm.moveItem(at:staging,to:directory)
    return Prepared(rgb24:directory.appendingPathComponent("visible.rgb24"),sha256:digest,plan:plan,
      sourceFrameRange:range,resizePolicy:"native-ffmpeg-rgb8-lanczos-or-centered-crop-v1")
  }
  public static func sceneCuts(rgb24:URL,plan:MLXMovieUpscalePlan) throws -> [Int] {
    let before=try MLXMovieFiles.Identity(rgb24),frameBytes=plan.size.width*plan.size.height*3
    guard plan.size.width<=2048,plan.size.height<=2048,before.bytes==Int64(plan.frames)*Int64(frameBytes) else {
      throw LTXError.invalid("Scene inspection RGB differs from its frozen complete source.")
    }
    let input=try FileHandle(forReadingFrom:rgb24);defer { try? input.close() }
    let yStep=max(1,(plan.size.height+63)/64),xStep=max(1,(plan.size.width+63)/64)
    var previous:[Float]=[],differences:[Float]=[]
    for _ in 0..<plan.frames {
      try Task.checkCancellation();let data=try input.read(upToCount:frameBytes) ?? Data()
      guard data.count==frameBytes else { throw LTXError.invalid("Scene inspection ended inside a source frame.") }
      var current:[Float]=[]
      data.withUnsafeBytes { raw in
        let bytes=raw.bindMemory(to:UInt8.self)
        for y in stride(from:0,to:plan.size.height,by:yStep) { for x in stride(from:0,to:plan.size.width,by:xStep) {
          let i=(y*plan.size.width+x)*3
          current.append((Float(bytes[i])*0.299+Float(bytes[i+1])*0.587+Float(bytes[i+2])*0.114)/255)
        } }
      }
      if !previous.isEmpty { differences.append(zip(previous,current).reduce(Float(0)) { $0+abs($1.0-$1.1) }/Float(current.count)) }
      previous=current
    }
    guard before == (try MLXMovieFiles.Identity(rgb24)) else { throw LTXError.invalid("Movie RGB changed during scene inspection.") }
    return try MLXMovieUpscalePlan.sceneCuts(adjacentLuminanceDifferences:differences)
  }
}
