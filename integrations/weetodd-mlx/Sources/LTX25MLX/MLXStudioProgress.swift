import Foundation
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
import LTX25Engine

/// Fractions describe work phases, not an ETA. Completion is emitted only after publication.
public struct MLXStudioProgress {
  private var fraction=0.0,steps=0,sceneSteps=0
  private let sceneWindowCount:Int
  private let temporalRounds:Int
  private var samplingSteps:Int
  private var temporalRound=0,temporalTiles=1,temporalTilesCompleted=0
  public init(sceneWindowCount:Int=0,temporalRounds:Int=0,samplingSteps:Int=11) {
    self.sceneWindowCount=max(0,sceneWindowCount)
    self.temporalRounds=max(0,temporalRounds)
    self.samplingSteps=max(1,samplingSteps)
  }
  public mutating func event(stage:String,completed:Int,total:Int) -> [String:Any] {
    let part=total > 0 ? min(1,max(0,Double(completed)/Double(total))) : 0
    var next=fraction,message=stage.replacingOccurrences(of:"_",with:" ")
    func sceneIndex(_ prefix:String) -> Int? {
      guard stage.hasPrefix(prefix),sceneWindowCount>0,
        let value=Int(stage.dropFirst(prefix.count).split(separator:":").first ?? ""),
        (1...sceneWindowCount).contains(value) else { return nil }
      return value
    }
    if stage.hasPrefix("scene_reference_prepare:") {
      message="Preparing scene opening image"
    } else if stage.hasPrefix("scene_reference_encode:") {
      next=0.08;message="Encoding scene opening image · \(completed)/\(total)"
    } else if stage == "scene_reference_weights_released" {
      next=0.08;message="Scene opening image ready"
    } else if let index=sceneIndex("scene_text_") {
      next=0.01+0.06*(Double(index-1)+part)/Double(sceneWindowCount)
      message="Encoding scene prompt \(index)/\(sceneWindowCount)"
    } else if stage == "scene_text_weights_released" {
      next=0.08;message="Scene prompts ready"
    } else if stage == "scene_window_released" && sceneWindowCount>0 {
      next=0.08+0.74*min(1,Double(completed)/Double(sceneWindowCount))
      message="Scene window \(completed)/\(sceneWindowCount) sampled"
    } else if let index=sceneIndex("scene_window_") {
      if stage.hasSuffix(":sampling") {
        sceneSteps=completed
        next=0.08+0.74*(Double(index-1)+min(1,Double(completed)/11))/Double(sceneWindowCount)
      } else if stage.contains(":stage1:") || stage.contains(":stage2:") {
        let progress=min(1,(Double(sceneSteps)+part)/11)
        next=0.08+0.74*(Double(index-1)+progress)/Double(sceneWindowCount)
      }
      message="Sampling scene window \(index)/\(sceneWindowCount) · \(completed)/\(total)"
    } else if stage == "audio_encode" { next=0.005+0.015*part;message="Encoding source audio · \(completed)/\(total)" }
    else if stage == "audio_encoder_weights_released" { next=0.02;message="Source audio ready" }
    else if stage.hasPrefix("text:") { next=0.02+0.04*part;message="Encoding prompt · \(completed)/\(total)" }
    else if stage.hasPrefix("msr_reference_prepare") { next=0.07;message="Preparing MSR image \(completed+1)/\(max(total,1))" }
    else if stage.hasPrefix("msr_guide_encode:") { next=0.07+0.02*part;message="Encoding MSR \(stage.split(separator:":").last ?? "") · \(completed)/\(total)" }
    else if stage == "msr_guide_ready" { next=0.07+0.02*part;message="MSR image \(completed)/\(total) ready" }
    else if stage.hasPrefix("reference_") { next=0.07;message="Preparing reference images · \(stage)" }
    else if stage == "guide_encode" || stage == "anchor_encode" || stage == "ingredients_guide_encode" {
      next=0.07+0.02*part
      message=stage == "ingredients_guide_encode"
        ? "Encoding Ingredients sheet · \(completed)/\(total)"
        : "Encoding Ripple references · \(completed)/\(total)"
    }
    else if stage == "sampling" {
      if total > 0 { samplingSteps=total }
      steps=completed;next=0.1+(temporalRounds>0 ? 0.52 : 0.72)*part
      message="Sampling · \(completed)/\(total) steps"
    }
    else if stage.hasPrefix("ripple:") || stage.hasPrefix("ingredients:") || stage.hasPrefix("msr:") {
      next=0.1+0.72*min(1,(Double(steps)+part)/8)
      let name=stage.hasPrefix("ingredients:") ? "Ingredients" : stage.hasPrefix("msr:") ? "MSR" : "Ripple"
      message="Sampling \(name) · step \(min(steps+1,8))/8 · block \(completed)/\(total)"
    }
    else if stage.hasPrefix("stage1:") || stage.hasPrefix("stage2:") || stage.hasPrefix("single_stage:") {
      next=0.1+(temporalRounds>0 ? 0.52 : 0.72)*min(1,(Double(steps)+part)/Double(samplingSteps))
      message="Sampling · step \(min(steps+1,samplingSteps))/\(samplingSteps) · block \(completed)/\(total)"
    } else if stage == "temporal_upscaler_weights_released" && temporalRounds>0 {
      temporalRound=min(temporalRounds,max(1,completed))
      next=0.62+0.2*Double(temporalRound-1)/Double(temporalRounds)
      message="Preparing temporal round \(temporalRound)/\(temporalRounds)"
    } else if stage == "temporal_tiles" && temporalRounds>0 {
      temporalTiles=max(1,total);temporalTilesCompleted=0
      message="Sampling temporal round \(temporalRound)/\(temporalRounds)"
    } else if (stage == "temporal_sampling" || stage == "temporal_tile_complete") && temporalRounds>0 {
      if stage == "temporal_tile_complete" { temporalTilesCompleted=min(temporalTiles,completed) }
      let tileProgress=stage == "temporal_tile_complete" ? 0 : part
      let roundProgress=(Double(temporalTilesCompleted)+tileProgress)/Double(temporalTiles)
      next=0.62+0.2*(Double(max(0,temporalRound-1))+min(1,roundProgress))/Double(temporalRounds)
      message="Sampling temporal round \(temporalRound)/\(temporalRounds) · tile \(min(temporalTiles,temporalTilesCompleted+1))/\(temporalTiles)"
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
  public static func shouldEmit(index:Int,total:Int,secondsSinceLast:TimeInterval) -> Bool {
    guard total>0,index>=0,index<total else { return false }
    return index == 0 || index == total-1 || secondsSinceLast >= 1
  }
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
