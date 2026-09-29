import Foundation
import Darwin

/// Ordered RGB24 frames with bounded pipe backpressure and cancellable writes.
/// Publishes a video-only file after FFmpeg succeeds; audio is muxed separately.
public final class RawVideoWriter {
  private let process:Process
  private let input:FileHandle
  private let log:FileHandle
  private let temporary:URL
  private let output:URL
  private let frameBytes:Int
  private let expectedFrames:Int
  private var written=0
  private var closed=false
  public init(ffmpeg:URL,output:URL,width:Int,height:Int,frames:Int,fps:Double) throws {
    guard ffmpeg.isFileURL,FileManager.default.isExecutableFile(atPath:ffmpeg.path),output.isFileURL,
      !FileManager.default.fileExists(atPath:output.path),(2...16384).contains(width),(2...16384).contains(height),
      width%2==0,height%2==0,(1...100000).contains(frames),fps.isFinite,(1...240).contains(fps) else {
      throw MediaOutputError.invalid("Invalid raw video encoder configuration.")
    }
    self.output=output;frameBytes=width*height*3;expectedFrames=frames
    temporary=output.deletingLastPathComponent().appendingPathComponent(".raw-video-"+UUID().uuidString+".mp4")
    let logURL=temporary.appendingPathExtension("log")
    let fd=Darwin.open(logURL.path,O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC,mode_t(0o600))
    guard fd>=0 else { throw MediaOutputError.invalid("Cannot create encoder log.") }
    log=FileHandle(fileDescriptor:fd,closeOnDealloc:true)
    let pipe=Pipe();input=pipe.fileHandleForWriting
    process=Process();process.executableURL=ffmpeg
    process.arguments=["-v","error","-nostdin","-n","-f","rawvideo","-pixel_format","rgb24",
      "-video_size","\(width)x\(height)","-framerate",String(fps),"-i","pipe:0","-an",
      "-c:v","libx264","-crf","18","-pix_fmt","yuv420p",temporary.path]
    process.standardInput=pipe;process.standardOutput=log;process.standardError=log
    do {
      let flags=fcntl(input.fileDescriptor,F_GETFL)
      guard flags>=0,fcntl(input.fileDescriptor,F_SETFL,flags|O_NONBLOCK)==0,
        fcntl(input.fileDescriptor,F_SETNOSIGPIPE,1)==0 else { throw MediaOutputError.invalid("Cannot configure encoder pipe.") }
      try Task.checkCancellation();try process.run()
      try pipe.fileHandleForReading.close()
    } catch { cancel();throw error }
  }
  deinit { cancel() }
  public func append(_ rgb:Data,frame:Int) throws {
    guard !closed,frame==written,written<expectedFrames,rgb.count==frameBytes else {
      throw MediaOutputError.invalid("Raw frames must be complete and sequential.")
    }
    try rgb.withUnsafeBytes { bytes in
      var offset=0
      while offset<bytes.count {
        try Task.checkCancellation()
        guard process.isRunning else { throw MediaOutputError.invalid("Video encoder exited before receiving all frames.") }
        let amount=Darwin.write(input.fileDescriptor,bytes.baseAddress!.advanced(by:offset),min(65536,bytes.count-offset))
        if amount>0 { offset += amount;continue }
        if amount<0 && errno==EINTR { continue }
        guard amount<0 && (errno==EAGAIN || errno==EWOULDBLOCK) else { throw MediaOutputError.invalid("Video encoder pipe failed.") }
        var descriptor=pollfd(fd:input.fileDescriptor,events:Int16(POLLOUT),revents:0)
        let ready=Darwin.poll(&descriptor,1,50)
        if ready<0 && errno != EINTR { throw MediaOutputError.invalid("Cannot wait for video encoder.") }
      }
    }
    written += 1
  }
  public func finish() throws {
    guard !closed,written==expectedFrames else { throw MediaOutputError.invalid("Incomplete raw video stream.") }
    try input.close()
    while process.isRunning { try Task.checkCancellation();usleep(10000) }
    process.waitUntilExit();try Task.checkCancellation()
    guard process.terminationStatus==0,(try temporary.resourceValues(forKeys:[.fileSizeKey]).fileSize ?? 0)>0 else {
      throw MediaOutputError.invalid("FFmpeg failed to encode the raw video stream.")
    }
    try FileManager.default.moveItem(at:temporary,to:output)
    closed=true;try? log.close()
    try? FileManager.default.removeItem(at:temporary.appendingPathExtension("log"))
  }
  public func cancel() {
    guard !closed else { return }
    closed=true;try? input.close()
    if process.isRunning {
      process.terminate();usleep(100000)
      if process.isRunning { kill(process.processIdentifier,SIGKILL) }
    }
    if process.processIdentifier>0 { process.waitUntilExit() }
    try? log.close()
    try? FileManager.default.removeItem(at:temporary)
    try? FileManager.default.removeItem(at:temporary.appendingPathExtension("log"))
  }
}
