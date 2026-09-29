import Foundation
import Darwin
import CoreGraphics
import CoreImage
import ImageIO
import LTX25Engine

/// Bounded local image preparation: EXIF orientation, sRGB, optional training
/// CRF round-trip, then Lanczos cover resize and center crop. No Python process.
public enum MLXReferenceImage {
  public static let policy="native-srgb-orientation-crf-lanczos-center-v1"
  struct Prepared {
    let file:URL
    let width:Int,height:Int
    func pixels() throws -> [Float] {
      let data=try Data(contentsOf:file,options:.mappedIfSafe)
      guard data.count == width*height*3*MemoryLayout<Float>.size else {
        throw LTXError.invalid("Prepared reference pixels are incomplete.")
      }
      return data.withUnsafeBytes { Array($0.bindMemory(to:Float.self)) }
    }
  }
  /// Complete all unweighted image work before loading Gemma. Prepared RGB is
  /// disk-backed and each bounded buffer is read only when its VAE stage runs.
  static func prepareStages(_ references:[MLXImageReference],sizes:[(width:Int,height:Int)],ffmpeg:URL,directory:URL,
    progress:(Int,String) throws -> Void = { _,_ in }) throws -> [[Prepared]] {
    var cached:[String:Prepared]=[:],stages:[[Prepared]]=[]
    for (index,size) in sizes.enumerated() {
      var stage:[Prepared]=[]
      for reference in references {
        try Task.checkCancellation();try progress(index,reference.role);try Task.checkCancellation()
        let key="\(size.width)x\(size.height):\(reference.crf):\(reference.path)"
        if let reused=cached[key] { stage.append(reused);continue }
        let file=directory.appendingPathComponent("pixels-"+UUID().uuidString+".f32")
        try autoreleasepool {
          let rgb=try prepare(URL(fileURLWithPath:reference.path),width:size.width,height:size.height,
            crf:reference.crf,ffmpeg:ffmpeg,temporaryParent:directory)
          try rgb.withUnsafeBytes { try Data($0).write(to:file,options:.withoutOverwriting) }
        }
        let prepared=Prepared(file:file,width:size.width,height:size.height)
        cached[key]=prepared;stage.append(prepared)
      }
      stages.append(stage)
    }
    return stages
  }
  private static func source(_ url:URL) throws -> CGImageSource {
    guard url.isFileURL else { throw LTXError.invalid("Reference must be a local image.") }
    let fd=Darwin.open(url.path,O_RDONLY | O_CLOEXEC | O_NONBLOCK)
    guard fd>=0 else { throw LTXError.invalid("Cannot open reference image.") }
    let handle=FileHandle(fileDescriptor:fd,closeOnDealloc:true);defer { try? handle.close() }
    var info=stat()
    guard fstat(fd,&info)==0,info.st_mode & S_IFMT == S_IFREG,info.st_size>0,info.st_size<=64*1024*1024 else {
      throw LTXError.invalid("Reference must be a regular image file of at most64MiB.")
    }
    let data=try handle.read(upToCount:64*1024*1024+1) ?? Data()
    guard data.count<=64*1024*1024,let image=CGImageSourceCreateWithData(data as CFData,[kCGImageSourceShouldCache:false] as CFDictionary) else {
      throw LTXError.invalid("Cannot read the reference image.")
    }
    return image
  }
  public static func inspect(_ url:URL) throws {
    let source=try source(url)
    _ = try dimensions(source)
  }
  private static func dimensions(_ source:CGImageSource) throws -> (Int,Int) {
    guard CGImageSourceGetCount(source)==1,
      let props=CGImageSourceCopyPropertiesAtIndex(source,0,nil) as? [CFString:Any],
      let width=props[kCGImagePropertyPixelWidth] as? Int,let height=props[kCGImagePropertyPixelHeight] as? Int,
      (1...8192).contains(width),(1...8192).contains(height),width*height<=32*1024*1024 else {
      throw LTXError.invalid("Reference requires a single image, at most8192pixels per side and32megapixels.")
    }
    return (width,height)
  }
  private static func decoded(_ url:URL) throws -> CGImage {
    let src=try source(url),size=try dimensions(src)
    let options:[CFString:Any]=[kCGImageSourceCreateThumbnailFromImageAlways:true,
      kCGImageSourceCreateThumbnailWithTransform:true,kCGImageSourceThumbnailMaxPixelSize:max(size.0,size.1),
      kCGImageSourceShouldCacheImmediately:true]
    guard let image=CGImageSourceCreateThumbnailAtIndex(src,0,options as CFDictionary) else { throw LTXError.invalid("Cannot decode reference pixels.") }
    return image
  }
  private static func rgb(_ image:CGImage) throws -> Data {
    let w=image.width,h=image.height
    var bytes=Data(count:w*h*4)
    try bytes.withUnsafeMutableBytes { raw in
      guard let context=CGContext(data:raw.baseAddress,width:w,height:h,bitsPerComponent:8,bytesPerRow:w*4,
        space:CGColorSpace(name:CGColorSpace.sRGB)!,bitmapInfo:CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue) else {
        throw LTXError.invalid("Cannot allocate reference RGB surface.")
      }
      context.setFillColor(red:1,green:1,blue:1,alpha:1);context.fill(CGRect(x:0,y:0,width:w,height:h))
      context.draw(image,in:CGRect(x:0,y:0,width:w,height:h))
    }
    var result=Data(count:w*h*3)
    bytes.withUnsafeBytes { source in result.withUnsafeMutableBytes { target in
      let input=source.bindMemory(to:UInt8.self),output=target.bindMemory(to:UInt8.self)
      for i in 0..<(w*h) { output[i*3]=input[i*4];output[i*3+1]=input[i*4+1];output[i*3+2]=input[i*4+2] }
    } }
    return result
  }
  public static func prepare(_ url:URL,width:Int,height:Int,crf:Int,ffmpeg:URL,temporaryParent:URL) throws -> [Float] {
    _ = try MLXImageEncodePlan(width:width,height:height)
    guard (0...51).contains(crf) else { throw LTXError.invalid("Reference CRF must be0...51.") }
    try Task.checkCancellation()
    let temp=temporaryParent.appendingPathComponent("reference-"+UUID().uuidString)
    let fm=FileManager.default;try fm.createDirectory(at:temp,withIntermediateDirectories:false)
    defer { try? fm.removeItem(at:temp) }
    let original=try decoded(url)
    var image=original
    if crf>0 {
      guard ffmpeg.isFileURL,fm.isExecutableFile(atPath:ffmpeg.path) else { throw LTXError.invalid("Reference CRF requires an executable FFmpeg.") }
      let raw=temp.appendingPathComponent("source.rgb"),encoded=temp.appendingPathComponent("source.mp4"),png=temp.appendingPathComponent("decoded.png")
      try rgb(original).write(to:raw,options:.withoutOverwriting)
      let padW=original.width+(original.width%2),padH=original.height+(original.height%2)
      try process(ffmpeg,args:["-v","error","-nostdin","-n","-f","rawvideo","-pix_fmt","rgb24","-s","\(original.width)x\(original.height)","-r","1","-i",raw.path,
        "-vf","pad=\(padW):\(padH):0:0:black","-c:v","libx264","-preset","veryfast","-crf",String(crf),"-frames:v","1",encoded.path],log:temp.appendingPathComponent("encode.log"))
      try process(ffmpeg,args:["-v","error","-nostdin","-n","-i",encoded.path,"-frames:v","1","-vf","crop=\(original.width):\(original.height):0:0",png.path],log:temp.appendingPathComponent("decode.log"))
      image=try decoded(png)
    }
    let scale=max(Double(width)/Double(image.width),Double(height)/Double(image.height))
    let w=ceil(Double(image.width)*scale),h=ceil(Double(image.height)*scale)
    guard w*h<=64*1024*1024 else { throw LTXError.invalid("Reference aspect ratio exceeds resize budget.") }
    let ci=CIImage(cgImage:image),vertical=h/Double(image.height)
    let resized=ci.applyingFilter("CILanczosScaleTransform",parameters:[kCIInputScaleKey:vertical,kCIInputAspectRatioKey:(w/Double(image.width))/vertical])
    let cropX=floor((w-Double(width))/2)
    let cropY=h-Double(height)-floor((h-Double(height))/2)
    let crop=CGRect(x:cropX,y:cropY,width:Double(width),height:Double(height))
    let context=CIContext(options:[.workingColorSpace:CGColorSpace(name:CGColorSpace.sRGB)!, .cacheIntermediates:false])
    guard let result=context.createCGImage(resized,from:crop,format:.RGBA8,colorSpace:CGColorSpace(name:CGColorSpace.sRGB)) else {
      throw LTXError.invalid("Cannot resize reference image.")
    }
    try Task.checkCancellation()
    return try rgb(result).map { Float($0)/255*2-1 }
  }
  private static func process(_ executable:URL,args:[String],log:URL) throws {
    let fd=Darwin.open(log.path,O_CREAT | O_EXCL | O_WRONLY | O_CLOEXEC,0o600)
    guard fd>=0 else { throw LTXError.invalid("Cannot create reference preparation log.") }
    let handle=FileHandle(fileDescriptor:fd,closeOnDealloc:true);defer { try? handle.close() }
    let process=Process();process.executableURL=executable;process.arguments=args
    process.standardOutput=handle;process.standardError=handle
    try Task.checkCancellation();try process.run()
    defer {
      if process.isRunning { process.terminate();usleep(100000);if process.isRunning { kill(process.processIdentifier,SIGKILL) } }
      process.waitUntilExit()
    }
    let deadline=Date().addingTimeInterval(120)
    while process.isRunning {
      try Task.checkCancellation()
      guard Date()<deadline else { throw LTXError.invalid("Reference preprocessing timed out.") }
      usleep(10000)
    }
    guard process.terminationStatus==0 else { throw LTXError.invalid("Reference preprocessing failed (exit\(process.terminationStatus)).") }
  }
}
