import Foundation

/// The F32 source encoder is a separate memory profile from the refiner and
/// native F16 control encoder. It retains a17-frame full spatial window.
public enum H3PublisherEncoderAdmission {
  public static func requiredPeakBytes(width:Int,height:Int,paddedFrames:Int) throws -> UInt64 {
    guard (32...2048).contains(width),(32...2048).contains(height),width.isMultiple(of:32),height.isMultiple(of:32),
      width*height <= H3Geometry.maximumCanvasPixels,(5...362).contains(paddedFrames),(paddedFrames-5).isMultiple(of:17) else {
      throw H3CheckpointError.invalid("Publisher F32 source encoder requires its exact bounded source grid and padded clock.")
    }
    // Twelve live full-resolution128-channel frame equivalents cover residual,
    // normalization, activation, causal-convolution terms and allocator overlap.
    // Lower-resolution wider layers fit this envelope. The fixed allowance
    // covers weights and posterior/output rows. This exceeds the measured
    // 1280×736 source's94.19GB physical peak; it is not a tiled-encoder claim.
    return UInt64(width)*UInt64(height)*12*17*128*4 + 2*1024*1024*1024
  }
  @discardableResult
  public static func validate(width:Int,height:Int,paddedFrames:Int,maximumStageBytes:UInt64,physicalMemory:UInt64) throws -> UInt64 {
    let required=try requiredPeakBytes(width:width,height:height,paddedFrames:paddedFrames)
    guard required <= maximumStageBytes,required <= physicalMemory/4*3 else {
      throw H3CheckpointError.invalid("Publisher F32 source encoding at \(width)×\(height) requires an explicit stage allowance of at least \(required)bytes and sufficient physical memory.")
    }
    return required
  }
}
