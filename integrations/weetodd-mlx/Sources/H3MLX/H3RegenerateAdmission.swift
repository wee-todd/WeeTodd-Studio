import Foundation

/// Regeneration-only planning includes every packed modality. The bounds keep
/// the largest full hidden tensor below signed-32 element addressing. Full QKV
/// and attended tensors are never allocated at this bound.
public struct H3RegenerateAdmission: Sendable {
  public static let maximumPackedRows=390_000
  public static func silenceSampleCount(paddedFrames:Int) throws -> Int {
    guard (5...362).contains(paddedFrames),(paddedFrames-5).isMultiple(of:17) else {
      throw H3CheckpointError.invalid("Encoded silence requires the exact regeneration frame clock.")
    }
    return Int((Double(paddedFrames)/24*32000).rounded(.toNearestOrEven))
  }
  public let visibleWidth:Int,visibleHeight:Int,workingWidth:Int,workingHeight:Int,cropX:Int,cropY:Int
  public let visibleFrames:Int,paddedFrames:Int,latentFrames:Int,videoRows:Int,audioRows:Int,textRows:Int,packedRows:Int
  public let maximumStageBytes:UInt64,estimatedStageBytes:UInt64,temporaryBytes:UInt64
  public var requiresHighRowQualification:Bool { packedRows > 64_000 }

  public init(width:Int,height:Int,visibleFrames:Int,paddedFrames:Int,fps:Double,textRows:Int,
    maximumStageBytes:UInt64,availableTemporaryBytes:UInt64) throws {
    guard fps == 24 else {
      throw H3CheckpointError.invalid("H3 regeneration requires 24 FPS. Preserve this source with FlashVSR, or explicitly choose a 24 FPS conversion before preparation.")
    }
    guard (32...4096).contains(width),(32...4096).contains(height),
      (5...362).contains(paddedFrames),(paddedFrames-5).isMultiple(of:17),
      visibleFrames > 0,visibleFrames <= paddedFrames,paddedFrames-visibleFrames <= 16,
      (1...4096).contains(textRows),maximumStageBytes > 0,maximumStageBytes <= 192*1024*1024*1024 else {
      throw H3CheckpointError.invalid("Invalid regenerate canvas, visible interval, text count or stage budget.")
    }
    let w=(width+31)/32*32,h=(height+31)/32*32
    let frames=(paddedFrames-5)/17*5+2
    let video=frames*(w/32)*(h/32)
    let audio=(try Self.silenceSampleCount(paddedFrames:paddedFrames)+799)/800*2
    let packed=video+audio+textRows
    guard packed <= Self.maximumPackedRows,packed*5376 < Int(Int32.max) else {
      throw H3CheckpointError.invalid("Regenerate packed hidden rows exceed the bounded indexing plan.")
    }
    // Full hidden/residual/modulation consumers, separate head-group Q/K/V,
    // query windows, streamed base/adapter weights and allocator allowance.
    let estimate=UInt64(packed)*UInt64((12*5376+3*7168)*2)+12*1024*1024*1024
    let temporary=UInt64(w)*UInt64(h)*UInt64(paddedFrames)*12
      + UInt64(video)*96*4*3 + 512*1024*1024
    guard estimate <= maximumStageBytes,temporary <= availableTemporaryBytes else {
      throw H3CheckpointError.invalid("Insufficient admitted stage memory or temporary disk for regeneration.")
    }
    self.visibleWidth=width;self.visibleHeight=height;workingWidth=w;workingHeight=h
    cropX=(w-width)/2;cropY=(h-height)/2
    self.visibleFrames=visibleFrames;self.paddedFrames=paddedFrames;latentFrames=frames
    videoRows=video;audioRows=audio;self.textRows=textRows;packedRows=packed
    self.maximumStageBytes=maximumStageBytes;estimatedStageBytes=estimate;temporaryBytes=temporary
  }
  public func validateExecution(qualificationProbe:Bool=false) throws {
    guard !requiresHighRowQualification || qualificationProbe else {
      throw H3CheckpointError.invalid("This regenerate grid requires a recorded large-row indexing, memory and timing qualification before ordinary execution.")
    }
  }
  public var geometry:H3Geometry {
    get throws { try H3Geometry(regenerateWidth:workingWidth,height:workingHeight,paddedFrames:paddedFrames) }
  }
}
