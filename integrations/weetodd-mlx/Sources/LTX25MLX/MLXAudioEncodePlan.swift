import LTX25Engine

/// Owned activation/array allowance for the staged causal audio encoder.
/// Driver allocations and allocator caches are outside this conservative bound.
public struct MLXAudioEncodePlan:Sendable {
  public let melFrames:Int,latentFrames:Int,ownedBufferBytes:Int
  public init(melFrames:Int,maximumMelFrames:Int=2011,maximumOwnedBufferBytes:Int=2*1024*1024*1024) throws {
    latentFrames=try MLXAudioEncoder.latentFrames(melFrames:melFrames,maximumMelFrames:maximumMelFrames)
    self.melFrames=melFrames
    // Full-rate conv-in is the largest activation. Reserve input, causal pad,
    // normalization/SiLU, residual, convolution output and layout copies;
    // three full largest-layer weight/cast slabs are included independently.
    let largestActivation=melFrames*64*128*4
    let largestWeight=512*512*3*3*4
    ownedBufferBytes=8*largestActivation+3*largestWeight+2*melFrames*64*4
    guard maximumOwnedBufferBytes>0,ownedBufferBytes<=maximumOwnedBufferBytes else {
      throw LTXError.invalid("Audio encoder exceeds its admitted owned-array workspace before convolution.")
    }
  }
}
