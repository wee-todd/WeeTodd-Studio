import Foundation
import InferenceContracts

/// Conservative owned-buffer admission, separate from measured run peaks. It
/// includes original-source staging, all DiT carries and exact Wan tile state.
public struct FlashVSRMemoryPlan:Codable,Equatable,Sendable {
  public let stageBytes:[String:UInt64]
  public let peakEstimatedActiveBytes:UInt64
  public let scratchBytes:UInt64
  public let temporaryBytes:UInt64
  public let inputBytes:UInt64,outputBytes:UInt64
  public let featureGroupBytes:UInt64,latentGroupBytes:UInt64
  public let wanCarryDiskBytes:UInt64
  public init(recipe:NativeFlashVSRRecipe)throws {
    try recipe.validate()
    func product(_ numbers:[UInt64])throws->UInt64 {
      try numbers.reduce(1) { partial,value in
        let result=partial.multipliedReportingOverflow(by:value)
        guard !result.overflow else { throw FlashVSRError.invalid("FlashVSR memory/disk geometry overflows.") };return result.partialValue
      }
    }
    func sum(_ numbers:[UInt64])throws->UInt64 {
      try numbers.reduce(0) { partial,value in
        let result=partial.addingReportingOverflow(value)
        guard !result.overflow else { throw FlashVSRError.invalid("FlashVSR memory/disk allowance overflows.") };return result.partialValue
      }
    }
    let pixels=try product([UInt64(recipe.workingHeight),UInt64(recipe.workingWidth)])
    inputBytes=recipe.inputBytes;outputBytes=recipe.outputBytes
    featureGroupBytes=try product([pixels,48]);latentGroupBytes=try product([pixels,2])
    // The reader, normalized source windows and convolution workspace are
    // bounded by one four-frame microstep. Source-format copies coexist with
    // the cast learned parameter only during an individual weight import.
    let projection=try sum([2*1_151_384_272,product([pixels,1024])])
    let cacheWindows:UInt64=recipe.variant == .turbo ? 40:90
    // 128-token windows, 1536 channels, K+V, BF16. One spatial window covers
    // 128×128 prepared source pixels. Every layer's retained windows are counted.
    let ditCarry=try product([pixels/16384,cacheWindows,128,1536,2,2])
    // The active loader holds shared BF16 tensors and imports one block at a
    // time. Bound two live BF16 blocks for iteration overlap plus the largest
    // original F32 import buffer; whole-checkpoint residency is not executed.
    var sharedSource:UInt64=0,blockSources:[String:UInt64]=[:],largestSource:UInt64=0
    for (name,shape) in FlashVSRCheckpoint.expectedShapes(component:.transformer) {
      let bytes=try product(shape.map(UInt64.init)+[4]);largestSource=max(largestSource,bytes)
      if name.hasPrefix("blocks.") {
        let block=String(name.split(separator:".")[1])
        blockSources[block]=try sum([blockSources[block] ?? 0,bytes])
      } else { sharedSource=try sum([sharedSource,bytes]) }
    }
    let streamedWeights=try sum([sharedSource/2,blockSources.values.max() ?? 0,largestSource])
    let transformer=try sum([streamedWeights,ditCarry,product([pixels,1024]),64*1024*1024])
    let decoder:UInt64,wanReplacementBytes:UInt64
    if recipe.decoder == .tiny {
      wanCarryDiskBytes=0;wanReplacementBytes=0
      // Includes nine MemBlock histories, up to four expanded feature frames,
      // source/color buffers and a conservative 3×3 convolution workspace.
      decoder=try sum([2*189_018_333,product([pixels,9752])])
    } else {
      let h=recipe.workingHeight/8,w=recipe.workingWidth/8,th=min(h,30),tw=min(w,52)
      func count(_ length:Int,_ tile:Int,_ stride:Int)->Int {
        let tail=length-tile;return tail/stride+1+(tail%stride==0 ? 0:1)
      }
      let tiles=try product([UInt64(count(h,th,15)),UInt64(count(w,tw,26))])
      let oneTile=try product([UInt64(th),UInt64(tw),151328,2])
      wanCarryDiskBytes=try product([tiles,oneTile]);wanReplacementBytes=oneTile
      // State is on disk between tiles. Only one tile's state/workspace and a
      // full-canvas <=4-frame RGB group are live in the decoder stage.
      decoder=try sum([2*507_609_880,product([oneTile,4]),product([pixels,256])])
    }
    stageBytes=["projection":projection,"transformer":transformer,"decoder":decoder]
    peakEstimatedActiveBytes=stageBytes.values.max()!
    let payload=try product([UInt64(recipe.groupCount),try sum([featureGroupBytes,latentGroupBytes,8192])])
    let raw=try sum([payload,wanCarryDiskBytes,wanReplacementBytes,outputBytes])
    // Header/journal/private-state replacement reserve, above exact payloads.
    scratchBytes=try sum([raw,raw/10,64*1024*1024]);temporaryBytes=scratchBytes
  }
  public func validate(maximumActiveBytes:UInt64,availableDiskBytes:UInt64,physicalMemoryBytes:UInt64)throws {
    let physicalAllowance=physicalMemoryBytes-physicalMemoryBytes/4
    guard peakEstimatedActiveBytes<=maximumActiveBytes,peakEstimatedActiveBytes<=physicalAllowance else {
      throw FlashVSRError.invalid("Selected native FlashVSR model/decoder needs an estimated \(peakEstimatedActiveBytes) active bytes; reduce canvas or explicitly admit a supported memory budget (physical reserve retained).")
    }
    guard scratchBytes<=availableDiskBytes else { throw FlashVSRError.invalid("Native FlashVSR needs \(scratchBytes) free bytes for immutable features, latents, selected decoder carry and raw output.") }
  }
}
