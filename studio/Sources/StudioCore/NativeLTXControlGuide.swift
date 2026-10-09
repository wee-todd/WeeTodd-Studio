import AVFoundation
import CoreImage
import CryptoKit
import Darwin
import Foundation
import ImageIO

/// Preprocessed control movies are decoded and resampled one frame at a time.
/// Each guide uses its trained downscale relative to the active sampling canvas.
enum NativeLTXControlGuide {
  static func prepare(source: URL, destination: URL, width: Int, height: Int,
    frames: Int, fps: Double, editorialDuration: Double, referenceDownscale:Int=2,
    stageDownscale:Int=2) async throws -> String {
    guard (64...4096).contains(width),(64...4096).contains(height),[1,2].contains(referenceDownscale),[1,2].contains(stageDownscale),width % (32*stageDownscale*referenceDownscale) == 0,height % (32*stageDownscale*referenceDownscale) == 0,
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
    let w=width/(stageDownscale*referenceDownscale),h=height/(stageDownscale*referenceDownscale),frameBytes=w*h*3,totalBytes=frameBytes*frames
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
  static func prepareSheet(source:URL,destination:URL,width:Int,height:Int,frames:Int) throws -> String {
    guard (32...4096).contains(width),(32...4096).contains(height),width%32 == 0,height%32 == 0,
      (121...2401).contains(frames),(frames-1)%8 == 0,
      let input=CGImageSourceCreateWithURL(source as CFURL,[kCGImageSourceShouldCache:false] as CFDictionary),
      let properties=CGImageSourceCopyPropertiesAtIndex(input,0,nil) as? [CFString:Any],
      let w=properties[kCGImagePropertyPixelWidth] as? Int,let h=properties[kCGImagePropertyPixelHeight] as? Int,
      w>0,h>0,w<=16384,h<=16384,Int64(w)*Int64(h)<=100_000_000,
      let image=CGImageSourceCreateThumbnailAtIndex(input,0,[kCGImageSourceCreateThumbnailFromImageAlways:true,
        kCGImageSourceCreateThumbnailWithTransform:true,kCGImageSourceThumbnailMaxPixelSize:max(width,height),
        kCGImageSourceShouldCacheImmediately:true] as CFDictionary) else {
      throw StudioError.invalid("Ingredients needs a readable bounded reference image and at least121 frames.")
    }
    try Task.checkCancellation()
    let ci=CIImage(cgImage:image),extent=ci.extent
    let scale=min(CGFloat(width)/extent.width,CGFloat(height)/extent.height)
    let scaled=ci.transformed(by:CGAffineTransform(scaleX:scale,y:scale))
    let outputWidth=CGFloat(width),outputHeight=CGFloat(height)
    let canvas=CGRect(x:0,y:0,width:outputWidth,height:outputHeight)
    let offsetX=outputWidth/2-scaled.extent.midX,offsetY=outputHeight/2-scaled.extent.midY
    let fitted=scaled.transformed(by:CGAffineTransform(translationX:offsetX,y:offsetY))
    let black=CIImage(color:CIColor(red:0,green:0,blue:0,alpha:1)).cropped(to:canvas)
    let centered=fitted.composited(over:black).cropped(to:canvas)
    var rgba=[UInt8](repeating:0,count:width*height*4)
    let context=CIContext(options:[.cacheIntermediates:false])
    rgba.withUnsafeMutableBytes { context.render(centered,toBitmap:$0.baseAddress!,rowBytes:width*4,
      bounds:CGRect(x:0,y:0,width:width,height:height),format:.RGBA8,colorSpace:CGColorSpace(name:CGColorSpace.sRGB)) }
    var rgb=Data(count:width*height*3)
    rgb.withUnsafeMutableBytes { bytes in
      let target=bytes.bindMemory(to:UInt8.self)
      for pixel in 0..<width*height { for channel in 0..<3 { target[pixel*3+channel]=rgba[pixel*4+channel] } }
    }
    let free=try FileManager.default.attributesOfFileSystem(forPath:destination.deletingLastPathComponent().path)[.systemFreeSize] as? NSNumber
    guard let free,free.int64Value>=Int64(rgb.count)*Int64(frames)+64*1024*1024 else {
      throw StudioError.invalid("Free disk space is insufficient for the Ingredients guide.")
    }
    let fd=Darwin.open(destination.path,O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC,0o600)
    guard fd>=0 else { throw StudioError.invalid("Cannot freeze the Ingredients guide.") }
    let file=FileHandle(fileDescriptor:fd,closeOnDealloc:true);var complete=false
    defer { try? file.close();if !complete { try? FileManager.default.removeItem(at:destination) } }
    var digest=SHA256()
    for _ in 0..<frames { try Task.checkCancellation();try file.write(contentsOf:rgb);digest.update(data:rgb) }
    try file.close();complete=true
    return digest.finalize().map { String(format:"%02x",$0) }.joined()
  }
  static func prepareAudio(source:URL,destination:URL,duration:Double) async throws -> String {
    guard duration.isFinite,(0.01...20).contains(duration),!FileManager.default.fileExists(atPath:destination.path) else {
      throw StudioError.invalid("CrossView source audio needs a valid new publication interval.")
    }
    var initial=stat()
    guard stat(source.path,&initial)==0,initial.st_mode & S_IFMT == S_IFREG,
      initial.st_size>0,initial.st_size<=4*1024*1024*1024 else {
      throw StudioError.invalid("CrossView source must be a regular movie under4GiB.")
    }
    let asset=AVURLAsset(url:source)
    guard let track=try await asset.loadTracks(withMediaType:.audio).first,
      try await asset.load(.duration).seconds+0.001>=duration,
      let description=try await track.load(.formatDescriptions).first,
      let original=CMAudioFormatDescriptionGetStreamBasicDescription(description) else {
      throw StudioError.invalid("CrossView's original source movie needs an audio track covering the selected duration.")
    }
    let rate=original.pointee.mSampleRate,channels=Int(original.pointee.mChannelsPerFrame)
    guard rate.isFinite,rate.rounded()==rate,(8000...192000).contains(rate),(1...8).contains(channels) else {
      throw StudioError.invalid("Unsupported CrossView source audio format.")
    }
    let frameBytes=channels*4,targetFrames=Int((duration*rate).rounded(.toNearestOrEven))
    let reader=try AVAssetReader(asset:asset)
    let output=AVAssetReaderTrackOutput(track:track,outputSettings:[AVFormatIDKey:kAudioFormatLinearPCM,
      AVLinearPCMBitDepthKey:32,AVLinearPCMIsFloatKey:true,AVLinearPCMIsBigEndianKey:false,
      AVLinearPCMIsNonInterleaved:false])
    output.alwaysCopiesSampleData=false
    guard reader.canAdd(output) else { throw StudioError.invalid("Cannot decode CrossView source audio.") }
    reader.add(output);reader.timeRange=CMTimeRange(start:.zero,duration:CMTime(seconds:duration,preferredTimescale:1_000_000_000))
    guard reader.startReading() else { throw reader.error ?? StudioError.invalid("Cannot read CrossView source audio.") }
    defer { reader.cancelReading() }
    let fd=Darwin.open(destination.path,O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC,0o600)
    guard fd>=0 else { throw StudioError.invalid("Cannot freeze CrossView source audio.") }
    let file=FileHandle(fileDescriptor:fd,closeOnDealloc:true);var complete=false
    defer { try? file.close();if !complete { try? FileManager.default.removeItem(at:destination) } }
    func header(_ frames:Int)->Data {
      var result=Data()
      func text(_ string:String) { result.append(string.data(using:.ascii)!) }
      func word<T:FixedWidthInteger>(_ value:T) { var little=value.littleEndian;withUnsafeBytes(of:&little) { result.append(contentsOf:$0) } }
      let bytes=frames*frameBytes
      text("RIFF");word(UInt32(36+bytes));text("WAVEfmt ");word(UInt32(16));word(UInt16(3))
      word(UInt16(channels));word(UInt32(rate));word(UInt32(Int(rate)*frameBytes));word(UInt16(frameBytes));word(UInt16(32))
      text("data");word(UInt32(bytes));return result
    }
    try file.write(contentsOf:header(0));var written=0
    let zero=Data(repeating:0,count:4096*frameBytes)
    while written<targetFrames,let sample=autoreleasepool(invoking:{ output.copyNextSampleBuffer() }) {
      try Task.checkCancellation()
      guard let block=CMSampleBufferGetDataBuffer(sample),
        let format=CMSampleBufferGetFormatDescription(sample),let pcm=CMAudioFormatDescriptionGetStreamBasicDescription(format),
        pcm.pointee.mFormatID==kAudioFormatLinearPCM,pcm.pointee.mSampleRate==rate,
        pcm.pointee.mChannelsPerFrame==UInt32(channels),pcm.pointee.mBitsPerChannel==32,
        pcm.pointee.mFormatFlags & kAudioFormatFlagIsFloat != 0,
        pcm.pointee.mFormatFlags & (kAudioFormatFlagIsNonInterleaved | kAudioFormatFlagIsBigEndian) == 0 else {
        throw StudioError.invalid("CrossView PCM decoder returned an incompatible format.")
      }
      let length=CMBlockBufferGetDataLength(block),samples=CMSampleBufferGetNumSamples(sample)
      let pts=CMSampleBufferGetPresentationTimeStamp(sample).seconds
      guard samples>0,length==samples*frameBytes,length<=16*1024*1024,pts.isFinite,(-60...86400).contains(pts) else {
        throw StudioError.invalid("CrossView PCM buffer timing or size is invalid.")
      }
      let start=Int((pts*rate).rounded(.toNearestOrEven)),skip=max(0,written-start)
      while written<min(start,targetFrames) {
        let count=min(4096,min(start,targetFrames)-written)
        try file.write(contentsOf:zero.prefix(count*frameBytes));written+=count
      }
      let count=min(max(0,samples-skip),targetFrames-written)
      if count>0 {
        var bytes=Data(count:count*frameBytes)
        let status=bytes.withUnsafeMutableBytes { CMBlockBufferCopyDataBytes(block,atOffset:skip*frameBytes,
          dataLength:count*frameBytes,destination:$0.baseAddress!) }
        guard status==kCMBlockBufferNoErr else { throw StudioError.invalid("Cannot read CrossView PCM samples.") }
        try file.write(contentsOf:bytes);written+=count
      }
    }
    if reader.status == .failed { throw reader.error ?? StudioError.invalid("CrossView audio decode failed.") }
    guard written>0 else { throw StudioError.invalid("CrossView source contains no audio samples.") }
    var final=stat()
    guard stat(source.path,&final)==0,final.st_dev==initial.st_dev,final.st_ino==initial.st_ino,
      final.st_size==initial.st_size,final.st_mtimespec.tv_sec==initial.st_mtimespec.tv_sec,
      final.st_mtimespec.tv_nsec==initial.st_mtimespec.tv_nsec else {
      throw StudioError.invalid("CrossView source changed during audio preparation.")
    }
    try file.seek(toOffset:0);try file.write(contentsOf:header(written));try file.close()
    let input=try FileHandle(forReadingFrom:destination);defer { try? input.close() };var digest=SHA256()
    while let bytes=try input.read(upToCount:1024*1024),!bytes.isEmpty { try Task.checkCancellation();digest.update(data:bytes) }
    complete=true;return digest.finalize().map { String(format:"%02x",$0) }.joined()
  }

}
