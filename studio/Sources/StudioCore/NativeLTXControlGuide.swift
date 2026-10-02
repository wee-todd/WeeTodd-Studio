import AVFoundation
import CoreImage
import CryptoKit
import Darwin
import Foundation

/// Preprocessed control movies are decoded and resampled one frame at a time.
/// Union's guide is half the stage-one canvas, hence one quarter of the final canvas.
enum NativeLTXControlGuide {
  static func prepare(source: URL, destination: URL, width: Int, height: Int,
    frames: Int, fps: Double, editorialDuration: Double) async throws -> String {
    guard (128...4096).contains(width),(128...4096).contains(height),width % 128 == 0,height % 128 == 0,
      (9...2401).contains(frames),(frames-1)%8 == 0,fps.isFinite,(1...120).contains(fps),
      editorialDuration.isFinite,(0.01...20).contains(editorialDuration) else {
      throw StudioError.invalid("Invalid Union guide canvas or timing.")
    }
    func signature() throws -> [Int64] {
      var status=stat()
      guard lstat(source.path,&status)==0,status.st_mode & S_IFMT == S_IFREG,
        status.st_size>0,status.st_size<=4*1024*1024*1024 else {
        throw StudioError.invalid("Choose a regular control movie under 4 GiB.")
      }
      return [Int64(status.st_dev),Int64(bitPattern:UInt64(status.st_ino)),status.st_size,
        Int64(status.st_mtimespec.tv_sec),Int64(status.st_mtimespec.tv_nsec),
        Int64(status.st_ctimespec.tv_sec),Int64(status.st_ctimespec.tv_nsec)]
    }
    let initial=try signature(),asset=AVURLAsset(url:source)
    guard let track=try await asset.loadTracks(withMediaType:.video).first else {
      throw StudioError.invalid("The Union guide needs a video track.")
    }
    let duration=try await asset.load(.duration).seconds,transform=try await track.load(.preferredTransform)
    guard duration.isFinite,duration+0.001>=editorialDuration else {
      throw StudioError.invalid("The Union guide is shorter than the requested clip. Choose a longer guide or shorten the clip.")
    }
    let w=width/4,h=height/4,frameBytes=w*h*3,totalBytes=frameBytes*frames
    let free=try FileManager.default.attributesOfFileSystem(forPath:destination.deletingLastPathComponent().path)[.systemFreeSize] as? NSNumber
    guard let free,free.int64Value>=Int64(totalBytes)+64*1024*1024 else {
      throw StudioError.invalid("Free disk space is insufficient for the frozen Union guide.")
    }
    let reader=try AVAssetReader(asset:asset)
    let output=AVAssetReaderTrackOutput(track:track,outputSettings:[kCVPixelBufferPixelFormatTypeKey as String:kCVPixelFormatType_32BGRA])
    output.alwaysCopiesSampleData=false
    guard reader.canAdd(output) else { throw StudioError.invalid("Cannot read the Union guide.") }
    reader.add(output)
    reader.timeRange=CMTimeRange(start:.zero,duration:CMTime(seconds:min(duration,Double(frames)/fps),preferredTimescale:1_000_000_000))
    guard reader.startReading() else { throw reader.error ?? StudioError.invalid("Cannot decode the Union guide.") }
    defer { reader.cancelReading() }
    let descriptor=Darwin.open(destination.path,O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC,0o600)
    guard descriptor>=0 else { throw StudioError.invalid("Cannot create the frozen Union guide.") }
    let file=FileHandle(fileDescriptor:descriptor,closeOnDealloc:true)
    var complete=false
    defer { try? file.close();if !complete { try? FileManager.default.removeItem(at:destination) } }
    let context=CIContext(options:[.cacheIntermediates:false]),color=CGColorSpace(name:CGColorSpace.sRGB)
    func rgb(_ sample:CMSampleBuffer) throws -> Data {
      guard let pixels=CMSampleBufferGetImageBuffer(sample) else { throw StudioError.invalid("Cannot decode a Union guide frame.") }
      let image=CIImage(cvPixelBuffer:pixels).transformed(by:transform),extent=image.extent
      guard extent.width.isFinite,extent.height.isFinite,extent.width>0,extent.height>0 else { throw StudioError.invalid("Invalid Union guide frame dimensions.") }
      let scale=max(CGFloat(w)/extent.width,CGFloat(h)/extent.height)
      let scaled=image.transformed(by:CGAffineTransform(scaleX:scale,y:scale))
      let outputWidth=CGFloat(w),outputHeight=CGFloat(h)
      let cropX=scaled.extent.midX-outputWidth/2,cropY=scaled.extent.midY-outputHeight/2
      let crop=CGRect(x:cropX,y:cropY,width:outputWidth,height:outputHeight)
      let centered=scaled.cropped(to:crop).transformed(by:CGAffineTransform(translationX:-crop.minX,y:-crop.minY))
      var rgba=[UInt8](repeating:0,count:w*h*4)
      rgba.withUnsafeMutableBytes { context.render(centered,toBitmap:$0.baseAddress!,rowBytes:w*4,
        bounds:CGRect(x:0,y:0,width:w,height:h),format:.RGBA8,colorSpace:color) }
      var bytes=Data(count:frameBytes)
      bytes.withUnsafeMutableBytes { raw in
        let target=raw.bindMemory(to:UInt8.self)
        for pixel in 0..<w*h { for channel in 0..<3 { target[pixel*3+channel]=rgba[pixel*4+channel] } }
      }
      return bytes
    }
    var pending=output.copyNextSampleBuffer(),previousTime = -Double.infinity,current:Data?,hash=SHA256()
    for index in 0..<frames {
      try Task.checkCancellation()
      let targetTime=Double(index)/fps
      while let sample=pending {
        try Task.checkCancellation()
        let time=CMSampleBufferGetPresentationTimeStamp(sample).seconds
        guard time.isFinite,time>=previousTime else { throw StudioError.invalid("Union guide timestamps are invalid or unordered.") }
        if current != nil && time>targetTime+1e-7 { break }
        current=try autoreleasepool { try rgb(sample) };previousTime=time
        pending=autoreleasepool { output.copyNextSampleBuffer() }
      }
      guard let current else { throw StudioError.invalid("The Union guide contains no decoded video frames.") }
      try file.write(contentsOf:current);hash.update(data:current)
    }
    if reader.status == .failed { throw reader.error ?? StudioError.invalid("Union guide decoding failed.") }
    guard try signature()==initial else { throw StudioError.invalid("The control movie changed during preparation.") }
    try Task.checkCancellation();try file.close();complete=true
    return hash.finalize().map { String(format:"%02x",$0) }.joined()
  }
}
