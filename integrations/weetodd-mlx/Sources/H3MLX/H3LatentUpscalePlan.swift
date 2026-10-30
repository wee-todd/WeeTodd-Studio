import Foundation
import MLX

public enum H3LatentUpscalerMathProfile: String, Codable, Sendable {
  case nativeF32Geometric, publisherBF16Arithmetic
  func scale(sourceWidth:Int,sourceHeight:Int,targetWidth:Int,targetHeight:Int) -> Double {
    let w=Double(targetWidth)/Double(sourceWidth),h=Double(targetHeight)/Double(sourceHeight)
    return self == .publisherBF16Arithmetic ? (w+h)/2:sqrt(w*h)
  }
  var dtype:DType { self == .publisherBF16Arithmetic ? .bfloat16:.float32 }
  func normalize(_ input:MLXArray,mean:MLXArray,std:MLXArray) -> MLXArray {
    (input.asType(dtype)-mean.asType(dtype))/std.asType(dtype)
  }
  func denormalize(_ input:MLXArray,mean:MLXArray,std:MLXArray) -> MLXArray {
    (input.asType(dtype)*std.asType(dtype)+mean.asType(dtype)).asType(.float32)
  }

}

public struct H3LatentUpscalePlan: Sendable {
  public enum Mode:String,Codable,Sendable { case directPublisher, multiHopNative }
  public struct Grid:Equatable,Sendable { public let width:Int,height:Int }
  public let mode:Mode
  public let grids:[Grid]
  public let normalization="h3_sampler_normalized_v1"
  public var isPublisherDirect:Bool { mode == .directPublisher }
  public init(sourceWidth:Int,sourceHeight:Int,targetWidth:Int,targetHeight:Int,mode:Mode) throws {
    guard [sourceWidth,sourceHeight,targetWidth,targetHeight].allSatisfy({ (32...4096).contains($0) && $0.isMultiple(of:32) }),
      targetWidth >= sourceWidth,targetHeight >= sourceHeight else {
      throw H3CheckpointError.invalid("Latent upscale grids require exact 32-pixel dimensions and cannot shrink an axis.")
    }
    var grids=[Grid(width:sourceWidth,height:sourceHeight)]
    var w=sourceWidth,h=sourceHeight
    while w != targetWidth || h != targetHeight {
      if mode == .directPublisher, (targetWidth > 2*w || targetHeight > 2*h) {
        throw H3CheckpointError.invalid("Direct publisher upscaling exceeds its two-times trained grid range; select an explicit multihop plan.")
      }
      w=min(2*w,targetWidth);h=min(2*h,targetHeight)
      grids.append(Grid(width:w,height:h))
    }
    self.mode=mode;self.grids=grids
  }
}
