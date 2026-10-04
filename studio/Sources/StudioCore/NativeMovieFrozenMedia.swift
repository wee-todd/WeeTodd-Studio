import AVFoundation
import CryptoKit
import Darwin
import Foundation


enum NativeMovieFrozenMedia {
  struct Identity:Codable,Equatable,Sendable {
    let path:String,device:UInt64,inode:UInt64,bytes:Int64,mtimeSeconds:Int64,mtimeNanoseconds:Int64,ctimeSeconds:Int64,ctimeNanoseconds:Int64
    init(_ url:URL) throws {
      let fd=Darwin.open(url.path,O_RDONLY|O_CLOEXEC|O_NOFOLLOW|O_NONBLOCK)
      guard fd>=0 else { throw StudioError.invalid("Cannot open regular movie media.") };defer { Darwin.close(fd) }
      var status=stat();guard fstat(fd,&status)==0,status.st_mode&S_IFMT==S_IFREG,status.st_size>0 else {
        throw StudioError.invalid("Movie media must be a nonempty regular file.")
      }
      path=url.path;device=UInt64(status.st_dev);inode=UInt64(status.st_ino);bytes=status.st_size
      mtimeSeconds=Int64(status.st_mtimespec.tv_sec);mtimeNanoseconds=Int64(status.st_mtimespec.tv_nsec)
      ctimeSeconds=Int64(status.st_ctimespec.tv_sec);ctimeNanoseconds=Int64(status.st_ctimespec.tv_nsec)
    }
  }
  static func digest(_ url:URL) throws -> String {
    let before=try Identity(url),fd=Darwin.open(url.path,O_RDONLY|O_CLOEXEC|O_NOFOLLOW|O_NONBLOCK)
    guard fd>=0 else { throw StudioError.invalid("Cannot hash regular movie media.") }
    let input=FileHandle(fileDescriptor:fd,closeOnDealloc:true);defer { try? input.close() }
    var hash=SHA256()
    while let bytes=try input.read(upToCount:1024*1024),!bytes.isEmpty { try Task.checkCancellation();hash.update(data:bytes) }
    guard before == (try Identity(url)) else { throw StudioError.invalid("Movie media changed during hashing.") }
    return hash.finalize().map { String(format:"%02x",$0) }.joined()
  }
  static func run(_ executable:URL,_ arguments:[String],log:URL) throws {
    guard FileManager.default.isExecutableFile(atPath:executable.path) else { throw StudioError.invalid("Movie media needs executable FFmpeg.") }
    let fd=Darwin.open(log.path,O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC,0o600)
    guard fd>=0 else { throw StudioError.invalid("Movie media needs a writable new log.") }
    let handle=FileHandle(fileDescriptor:fd,closeOnDealloc:true);defer { try? handle.close() }
    let process=Process();process.executableURL=executable;process.arguments=arguments
    process.standardOutput=handle;process.standardError=handle
    try Task.checkCancellation();try process.run()
    defer {
      if process.isRunning { process.terminate();usleep(100000);if process.isRunning { kill(process.processIdentifier,SIGKILL) } }
      process.waitUntilExit()
    }
    while process.isRunning { try Task.checkCancellation();usleep(10000) }
    guard process.terminationStatus==0 else { throw StudioError.invalid("Movie FFmpeg step failed; retained log: "+log.path) }
  }
  static func videoClock(_ url:URL,maximumFrames:Int=1_000_000) async throws -> (width:Int,height:Int,fps:Double,times:[Double]) {
    let before=try Identity(url),asset=AVURLAsset(url:url)
    let tracks=try await asset.loadTracks(withMediaType:.video)
    guard tracks.count==1 else { throw StudioError.invalid("Movie source needs exactly one video track.") }
    let track=tracks[0],natural=try await track.load(.naturalSize),transform=try await track.load(.preferredTransform)
    let bounds=CGRect(origin:.zero,size:natural).applying(transform)
    let rate=Double(try await track.load(.nominalFrameRate))
    guard rate.isFinite,(1...60).contains(rate),bounds.width.isFinite,bounds.height.isFinite,
      bounds.width>=32,bounds.height>=32,bounds.width<=8192,bounds.height<=8192 else {
      throw StudioError.invalid("Movie track dimensions or constant frame cadence are invalid.")
    }
    let reader=try AVAssetReader(asset:asset)
    // Compressed samples suffice for timing; no full movie pixel allocation.
    let output=AVAssetReaderTrackOutput(track:track,outputSettings:nil);output.alwaysCopiesSampleData=false
    guard reader.canAdd(output) else { throw StudioError.invalid("Cannot inspect movie timing.") }
    reader.add(output);guard reader.startReading() else { throw StudioError.invalid("Cannot start movie timing inspection.") }
    defer { reader.cancelReading() }
    var times:[Double]=[]
    while let sample=output.copyNextSampleBuffer() {
      try Task.checkCancellation()
      let count=CMSampleBufferGetNumSamples(sample)
      if count==0 {
        // AVAssetReader also emits edit/drain/empty-media control buffers.
        // They contain no frames; counting them rejects ordinary MOV/MP4.
        guard CMSampleBufferIsValid(sample),CMSampleBufferDataIsReady(sample),
          CMSampleBufferGetImageBuffer(sample)==nil,
          (CMSampleBufferGetDataBuffer(sample).map({ CMBlockBufferGetDataLength($0) }) ?? 0) == 0 else {
          throw StudioError.invalid("Movie empty control buffer unexpectedly contains image payload.")
        }
        continue
      }
      guard count>0,count<=maximumFrames-times.count else { throw StudioError.invalid("Movie timing exceeds its bounded frame inspection.") }
      // Raw media PTS can precede or follow an MP4 edit-list mapping. Output
      // timing expresses the actual visible movie timeline (e.g. B-frame lead).
      var needed=0
      guard CMSampleBufferGetOutputSampleTimingInfoArray(sample,entryCount:0,arrayToFill:nil,entriesNeededOut:&needed)==noErr,
        needed==1 || needed==count else { throw StudioError.invalid("Movie sample has ambiguous output timing.") }
      var timing=[CMSampleTimingInfo](repeating:CMSampleTimingInfo(),count:needed)
      let status=timing.withUnsafeMutableBufferPointer {
        CMSampleBufferGetOutputSampleTimingInfoArray(sample,entryCount:needed,arrayToFill:$0.baseAddress,entriesNeededOut:nil)
      }
      guard status==noErr else { throw StudioError.invalid("Movie sample has no visible output timing.") }
      for index in 0..<count {
        let info=timing[needed==1 ? 0 : index]
        let time=info.presentationTimeStamp.seconds+(needed==1 && count>1 ? Double(index)*info.duration.seconds : 0)
        guard time.isFinite else { throw StudioError.invalid("Movie sample has no finite output presentation time.") }
        times.append(time)
      }
    }
    guard reader.status == .completed,!times.isEmpty else { throw StudioError.invalid("Movie frame inspection did not complete.") }
    times.sort()
    let first=times[0]
    guard times.enumerated().allSatisfy({ abs(($0.element-first)*rate-Double($0.offset))<0.05 }),
      before == (try Identity(url)) else { throw StudioError.invalid("Movie has variable frame cadence or changed during inspection.") }
    return (Int(bounds.width.rounded()),Int(bounds.height.rounded()),rate,times)
  }
}
