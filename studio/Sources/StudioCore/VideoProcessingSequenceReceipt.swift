import Foundation

public enum VideoProcessingStagePayload: Codable, Sendable {
  case flashVSR(FlashVSRReceipt)
  case h3Regenerate(H3RegenerateReceipt)
  case pixelCQ(NativePixelCQReceipt)
  public var kind:String {
    switch self {case .flashVSR:return "flashvsr-turbo";case .h3Regenerate:return "h3-regenerate";case .pixelCQ:return "native-pixel-cq"}
  }
  public var files:[FlashVSRFile] {
    switch self {
    case .flashVSR(let r):return [r.video,r.restored]+[r.audio,r.sourcePreview].compactMap{$0}
    case .h3Regenerate(let r):return [r.video,r.restored,r.workerResult,r.workerJob,r.refinedLatents]+[r.audio].compactMap{$0}
    case .pixelCQ(let r):return [r.video,r.workerMetadata,r.workerRecipe]+[r.restored].compactMap{$0}
    }
  }
  public var elapsedSeconds:Double {
    switch self {case .flashVSR(let r):return r.elapsedSeconds;case .h3Regenerate(let r):return r.elapsedSeconds;case .pixelCQ(let r):return r.elapsedSeconds}
  }
}
/// One completed stage is independently reusable. Downstream recipes and
/// delivery codecs are intentionally absent from this immutable evidence.
public struct VideoProcessingStageReceipt: Codable, Sendable {
  public var format="weetodd-video-processing-stage-v2"
  public let key:String,settingsFingerprint:String,inputRGBSHA256:String,outputRGBSHA256:String
  public let input:FlashVSRFile,restored:FlashVSRFile,media:FlashVSRMediaInfo
  public let engineReceipt:FlashVSRFile,payload:VideoProcessingStagePayload
}
public struct VideoProcessingSequenceReceipt: Codable, Sendable {
  public struct Stage:Codable,Sendable {
    public let id:String,receiptFile:FlashVSRFile,receipt:VideoProcessingStageReceipt
  }
  public var format="weetodd-video-processing-sequence-result-v2"
  public let requestFingerprint:String,recipe:VideoProcessingRecipe,source:FlashVSRFile,sourceMedia:FlashVSRMediaInfo
  public let stages:[Stage],video:FlashVSRFile,restored:FlashVSRFile,audio:FlashVSRFile?
  public let sourcePCMHash:String?,outputPCMHash:String?,media:FlashVSRMediaInfo
  public var elapsedSeconds:Double {stages.reduce(0){$0+$1.receipt.payload.elapsedSeconds}}
}
