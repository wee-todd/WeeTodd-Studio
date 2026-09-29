import Foundation
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
import LTX25Engine

/// Fractions describe work phases, not an ETA. Completion is emitted only after publication.
public struct MLXStudioProgress {
  private var fraction=0.0,steps=0
  public init() {}
  public mutating func event(stage:String,completed:Int,total:Int) -> [String:Any] {
    let part=total > 0 ? min(1,max(0,Double(completed)/Double(total))) : 0
    var next=fraction,message=stage.replacingOccurrences(of:"_",with:" ")
    if stage == "audio_encode" { next=0.005+0.015*part;message="Encoding source audio · \(completed)/\(total)" }
    else if stage == "audio_encoder_weights_released" { next=0.02;message="Source audio ready" }
    else if stage.hasPrefix("text:") { next=0.02+0.04*part;message="Encoding prompt · \(completed)/\(total)" }
    else if stage.hasPrefix("reference_") { next=0.07;message="Preparing reference images · \(stage)" }
    else if stage == "guide_encode" || stage == "anchor_encode" {
      next=0.07+0.02*part;message="Encoding Ripple references · \(completed)/\(total)"
    }
    else if stage == "sampling" { steps=completed;next=0.1+0.72*part;message="Sampling · \(completed)/\(total) steps" }
    else if stage.hasPrefix("ripple:") {
      next=0.1+0.72*min(1,(Double(steps)+part)/8)
      message="Sampling Ripple · step \(min(steps+1,8))/8 · block \(completed)/\(total)"
    }
    else if stage.hasPrefix("stage1:") || stage.hasPrefix("stage2:") {
      next=0.1+0.72*min(1,(Double(steps)+part)/11)
      message="Sampling · step \(min(steps+1,11))/11 · block \(completed)/\(total)"
    } else if stage.hasPrefix("upscale") { message="Upscaling latents for refinement" }
    else if stage == "video_layers" { next=0.83;message="Decoding video · layer \(completed)/\(total)" }
    else if stage == "video_decode" { next=0.84+0.12*part;message="Decoding video · frame \(completed)/\(total)" }
    else if stage == "source_audio_published" { next=0.97;message="Publishing original source audio" }
    else if stage.hasPrefix("audio") { next=0.97;message="Decoding audio" }
    else if stage == "ready_to_publish" { next=0.995;message="Publishing synchronized movie" }
    fraction=max(fraction,min(0.995,next))
    return ["event":"progress","message":message,"fraction":fraction,"stage":stage,"completed":completed,"total":total]
  }
}

/// One 640px preview, derived from the already-decoded frame; no second VAE execution.
public enum MLXStudioPreview {
  public static func write(rgb:Data,width:Int,height:Int,to url:URL) throws {
    guard (1...4096).contains(width),(1...4096).contains(height),rgb.count == width*height*3 else {
      throw LTXError.invalid("Invalid decoded preview dimensions.")
    }
    let scale=min(1,640/Double(max(width,height)))
    let w=max(1,Int(Double(width)*scale)),h=max(1,Int(Double(height)*scale))
    guard let provider=CGDataProvider(data:rgb as CFData),let source=CGImage(width:width,height:height,
      bitsPerComponent:8,bitsPerPixel:24,bytesPerRow:width*3,space:CGColorSpaceCreateDeviceRGB(),
      bitmapInfo:CGBitmapInfo(rawValue:0),provider:provider,decode:nil,shouldInterpolate:true,intent:.defaultIntent),
      let context=CGContext(data:nil,width:w,height:h,bitsPerComponent:8,bytesPerRow:w*4,
        space:CGColorSpaceCreateDeviceRGB(),bitmapInfo:CGImageAlphaInfo.premultipliedLast.rawValue) else {
      throw LTXError.invalid("Cannot create bounded preview.")
    }
    context.interpolationQuality = .medium
    context.draw(source,in:CGRect(x:0,y:0,width:w,height:h))
    guard let image=context.makeImage() else { throw LTXError.invalid("Cannot resize preview.") }
    let bytes=NSMutableData()
    guard let destination=CGImageDestinationCreateWithData(bytes,UTType.png.identifier as CFString,1,nil) else {
      throw LTXError.invalid("Cannot encode preview.")
    }
    CGImageDestinationAddImage(destination,image,nil)
    guard CGImageDestinationFinalize(destination) else { throw LTXError.invalid("Cannot finish preview.") }
    try (bytes as Data).write(to:url,options:.atomic)
  }
}
