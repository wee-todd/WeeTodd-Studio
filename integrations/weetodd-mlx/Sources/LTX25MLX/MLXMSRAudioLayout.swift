import Foundation
import MLX
import LTX25Engine

/// Clean speaker references precede target audio. Sparse numbered image slots
/// retain absolute five-second negative windows; absent slots do not collapse time.
struct MLXMSRAudioLayout:Sendable {
  let slotIDs:[Int],lengths:[Int],targetFrames:Int,prefixFrames:Int
  let positions:[Float]
  var audioTokens:Int { prefixFrames+targetFrames }
  init(slotIDs:[Int],lengths:[Int],imageSlotCount:Int,targetFrames:Int) throws {
    guard !slotIDs.isEmpty,slotIDs.count<=2,slotIDs == Array(Set(slotIDs)).sorted(),
      slotIDs.allSatisfy({ (1...2).contains($0) && $0<=imageSlotCount }),
      (1...5).contains(imageSlotCount),lengths.count==slotIDs.count,
      lengths.allSatisfy({ (1...125).contains($0) }), (1...1501).contains(targetFrames) else {
      throw LTXError.invalid("MSR V2 voices require unique image slots 1–2 and at most 125 clean tokens each.")
    }
    func start(_ i:Int)->Float { Float(max(0,i*4-3))/100 }
    var positions:[Float]=[]
    for (slot,count) in zip(slotIDs,lengths) {
      let lastEnd=start(count)
      let shift = -Float(imageSlotCount-slot)*5-0.04-lastEnd
      for i in 0..<count { positions.append((start(i)+start(i+1))/2+shift) }
    }
    positions += (0..<targetFrames).map { (start($0)+start($0+1))/2 }
    self.slotIDs=slotIDs;self.lengths=lengths;self.targetFrames=targetFrames
    prefixFrames=lengths.reduce(0,+);self.positions=positions
  }
  static func resolve(_ request:MLXDistilledRequest,target:AVGeometry) throws -> Self? {
    guard let msr=request.msr,!msr.audioReferences.isEmpty else { return nil }
    let lengths=try msr.audioReferences.map {
      min(125,try MLXAudioMelPlan(samples:Int(($0.effectiveDurationSeconds*16000).rounded(.toNearestOrEven))).latentFrames)
    }
    return try Self(slotIDs:msr.audioReferences.map(\.imageSlot),lengths:lengths,
      imageSlotCount:msr.references.count,targetFrames:target.audioFrames)
  }
  func prepare(generated:MLXArray,references:[MLXArray]) throws
    -> (latent:MLXArray,condition:MLXAudioDenoiseCondition) {
    guard generated.dtype == .float32,generated.shape == [targetFrames,128],
      references.count == lengths.count,zip(references,lengths).allSatisfy({ $0.0.dtype == .float32 && $0.0.shape == [$0.1,128] }),
      ([generated]+references).allSatisfy({ MLX.isFinite($0).all().item(Bool.self) }) else {
      throw LTXError.invalid("MSR voice tokens differ from the admitted speaker layout.")
    }
    let latent=concatenated(references+[generated],axis:0)
    let clean=concatenated(references+[MLXArray.zeros([targetFrames,128])],axis:0)
    eval(latent,clean)
    return (latent,try MLXAudioDenoiseCondition(clean:clean,
      mask:Array(repeating:0,count:prefixFrames)+Array(repeating:1,count:targetFrames)))
  }
  func target(_ sampled:MLXArray) throws -> MLXArray {
    guard sampled.shape == [audioTokens,128] else { throw LTXError.invalid("MSR target audio trim differs from its prefix layout.") }
    return sampled[prefixFrames..<audioTokens]
  }
}
