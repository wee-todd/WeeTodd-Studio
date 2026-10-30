import Foundation

/// The chain publishes its own delivery receipt; stage receipts remain immutable cache evidence.
public struct VideoProcessingChainReceipt: Codable, Sendable {
  public let format: String
  public let requestFingerprint: String
  public let recipe: VideoProcessingRecipe
  public let source: FlashVSRFile
  public let native: NativePixelCQReceipt
  public let flashVSR: FlashVSRReceipt
  public let nativeReceiptFile: FlashVSRFile
  public let flashReceiptFile: FlashVSRFile
  public let video: FlashVSRFile
  public let sourceAudioSHA256: String?
  public let outputAudioSHA256: String?
  public let sourcePCMHash: String?
  public let outputPCMHash: String?
}

/// Common acceptance metadata without pretending native LTX inference is a FlashVSR backend.
public enum VideoProcessingReceipt: Sendable {
  case flashVSR(FlashVSRReceipt)
  case pixelCQ(NativePixelCQReceipt)
  case h3Regenerate(H3RegenerateReceipt)
  case chain(VideoProcessingChainReceipt)
  case sequence(VideoProcessingSequenceReceipt)
  public var chain:VideoProcessingChainReceipt? {if case .chain(let r)=self{return r};return nil}
  public var sequence:VideoProcessingSequenceReceipt? {if case .sequence(let r)=self{return r};return nil}
  public var flashVSR:FlashVSRReceipt? {
    switch self {case .flashVSR(let r):return r;case .chain(let r):return r.flashVSR;case .sequence(let r):if case .flashVSR(let p)=r.stages.last?.receipt.payload{return p};return nil;default:return nil}
  }
  public var native:NativePixelCQReceipt? {
    switch self {case .pixelCQ(let r):return r;case .chain(let r):return r.native;case .sequence(let r):if case .pixelCQ(let p)=r.stages.last?.receipt.payload{return p};return nil;default:return nil}
  }
  public var h3Regenerate:H3RegenerateReceipt? {
    switch self {case .h3Regenerate(let r):return r;case .sequence(let r):if case .h3Regenerate(let p)=r.stages.last?.receipt.payload{return p};return nil;default:return nil}
  }
  public var video:FlashVSRFile {
    switch self {case .flashVSR(let r):return r.video;case .pixelCQ(let r):return r.video;case .h3Regenerate(let r):return r.video;case .chain(let r):return r.video;case .sequence(let r):return r.video}
  }
  public var restored:FlashVSRFile {sequence?.restored ?? h3Regenerate?.restored ?? flashVSR?.restored ?? native?.restored ?? video}
  public var audio:FlashVSRFile? {
    if let sequence{return sequence.audio};if let chain{return chain.native.media.hasAudio ? chain.source:nil};if let native{return native.media.hasAudio ? native.video:nil};return h3Regenerate?.audio ?? flashVSR?.audio
  }
  public var elapsedSeconds:Double {
    switch self {case .flashVSR(let r):return r.elapsedSeconds;case .pixelCQ(let r):return r.elapsedSeconds;case .h3Regenerate(let r):return r.elapsedSeconds;case .chain(let r):return r.native.elapsedSeconds+r.flashVSR.elapsedSeconds;case .sequence(let r):return r.elapsedSeconds}
  }
  public var requestFingerprint:String {
    switch self {case .flashVSR(let r):return r.requestFingerprint;case .pixelCQ(let r):return r.requestFingerprint;case .h3Regenerate(let r):return r.requestFingerprint;case .chain(let r):return r.requestFingerprint;case .sequence(let r):return r.requestFingerprint}
  }
  public var width:Int {sequence?.media.width ?? h3Regenerate?.settings.targetWidth ?? flashVSR?.plan.request.settings.width ?? native!.media.width}
  public var height:Int {sequence?.media.height ?? h3Regenerate?.settings.targetHeight ?? flashVSR?.plan.request.settings.height ?? native!.media.height}
  public var fps:FlashVSRFrameRate {sequence?.media.fps ?? h3Regenerate?.fps ?? flashVSR?.plan.media.fps ?? native!.media.fps}
  public var frames:Int {sequence?.media.frames ?? h3Regenerate?.frames ?? flashVSR?.plan.frames ?? native!.media.frames}
  public var label:String {
    if let sequence{return sequence.recipe.steps.map(\.label).joined(separator:" → ")}
    if h3Regenerate != nil{return "H3 Regenerate"}
    return (chain == nil ? "":"Native Pixel + CQ → ")+(flashVSR.map{"Turbo " + $0.plan.request.settings.decoder.label} ?? "Native Pixel Spatial + CQ")
  }
  public static func read(from url:URL) throws -> Self {
    let info=try url.resourceValues(forKeys:[.isRegularFileKey,.fileSizeKey])
    guard info.isRegularFile==true,let size=info.fileSize,(1...8*1024*1024).contains(size) else{throw StudioError.invalid("Invalid bounded processing receipt.")}
    let data=try Data(contentsOf:url),object=try JSONSerialization.jsonObject(with:data) as? [String:Any],decoder=JSONDecoder()
    switch object?["format"] as? String {
    case "weetodd-video-processing-sequence-result-v2":
      let result=try decoder.decode(VideoProcessingSequenceReceipt.self,from:data);try result.recipe.validate()
      guard result.recipe.isSequence,result.stages.count==result.recipe.steps.count,zip(result.stages,result.recipe.steps).allSatisfy({$0.id==$1.id&&$0.receipt.payload.kind==$1.kind}) else{throw StudioError.invalid("Processing sequence receipt does not match its typed stages.")}
      return .sequence(result)
    case "weetodd-video-processing-chain-result-v1":return .chain(try decoder.decode(VideoProcessingChainReceipt.self,from:data))
    case "weetodd-video-upscale-result-v1":return .flashVSR(try decoder.decode(FlashVSRReceipt.self,from:data))
    case "weetodd-h3-regenerate-result-v1":return .h3Regenerate(try decoder.decode(H3RegenerateReceipt.self,from:data))
    case nil where object?["workerMetadata"] != nil:return .pixelCQ(try decoder.decode(NativePixelCQReceipt.self,from:data))
    default:throw StudioError.invalid("Unknown processing receipt version.")
    }
  }
}
