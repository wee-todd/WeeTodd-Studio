import Foundation

/// A complete clip has one token state and one set of five global tokens per
/// native block. Window attention is bounded; temporal output chunks are admitted
/// only with their complete dependency halo and native-block alignment.
public enum H3FastDecoderWindows {
  public struct Window:Sendable { public let tokens:[Int],globalBlocks:[Int] }
  public struct Plan:Sendable {
    public let frames:Int,height:Int,width:Int,layer:Int
    public let windows:[Window],inversePermutation:[Int]
    public var nativeBlocks:Int { (frames+4)/5 }
    public var tokenCount:Int { frames*height*width }
    public var spatialShift:Int { layer%2 == 0 ? 0 : 8 }
    public var temporalShift:Int { layer%2 == 0 ? 0 : 2 }
  }
  public static func plan(frames:Int,height:Int,width:Int,layer:Int) throws -> Plan {
    guard (1...10_000).contains(frames),(1...512).contains(height),(1...512).contains(width),(0..<24).contains(layer),frames*height*width <= 4_194_304 else { throw H3CheckpointError.invalid("Fast H3 window geometry exceeds its bounded admission.") }
    let s=layer%2 == 0 ? 0 : 8,st=layer%2 == 0 ? 0 : 2
    let nt=(frames+st+4)/5,ny=(height+s+15)/16,nx=(width+s+15)/16
    var windows=[Window](),inverse=[Int](repeating:0,count:frames*height*width),cursor=0
    for time in 0..<nt {
      let first=max(0,time*5-st),last=min(frames,(time+1)*5-st)
      let blocks=Array((first/5)...((last-1)/5))
      for row in 0..<ny { for col in 0..<nx {
        var tokens=[Int]()
        for t in first..<last { for y in max(0,row*16-s)..<min(height,(row+1)*16-s) { for x in max(0,col*16-s)..<min(width,(col+1)*16-s) {
          let index=(t*height+y)*width+x;tokens.append(index);inverse[index]=cursor;cursor += 1
        } } }
        if !tokens.isEmpty { windows.append(Window(tokens:tokens,globalBlocks:blocks)) }
      } }
    }
    guard cursor == frames*height*width else { throw H3CheckpointError.invalid("Fast H3 windows do not cover the complete state.") }
    return Plan(frames:frames,height:height,width:width,layer:layer,windows:windows,inversePermutation:inverse)
  }
  public static func retainedFrames(latentFrames:Int,latentOffset:Int=0) throws -> [Int] {
    guard (1...10_000).contains(latentFrames),latentOffset >= 0,latentOffset <= 10_000-latentFrames else { throw H3CheckpointError.invalid("Invalid fast H3 temporal length.") }
    return (0..<(latentFrames*4)).filter { frame in (frame/4+latentOffset)%5 != 0 || frame%4 == 3 }
  }
  /// Reverse the actual alternating window/global dependency graph. Initial
  /// globals are constants; at each preceding layer a global depends on every
  /// spatial token of its own native block. This is not a one-neighbor heuristic.
  public static func requiredContext(output:Range<Int>,totalFrames:Int,layers:Int=24) throws -> Range<Int> {
    guard (1...10_000).contains(totalFrames),!output.isEmpty,output.lowerBound >= 0,output.upperBound <= totalFrames,(1...24).contains(layers) else { throw H3CheckpointError.invalid("Invalid fast H3 output interval.") }
    var tokens=Set(output),globals=Set<Int>()
    for layer in (0..<layers).reversed() {
      let shift=layer%2 == 0 ? 0 : 2
      var previousTokens=tokens,previousGlobals=globals
      for t in tokens {
        let window=(t+shift)/5,first=max(0,window*5-shift),last=min(totalFrames,(window+1)*5-shift)
        previousTokens.formUnion(first..<last)
        previousGlobals.formUnion((first/5)...((last-1)/5))
      }
      for block in globals { previousTokens.formUnion((block*5)..<min(totalFrames,(block+1)*5)) }
      tokens=previousTokens;globals=previousGlobals
    }
    let first=tokens.min()!/5*5,last=min(totalFrames,((tokens.max()!+5)/5)*5)
    return first..<last
  }
  public static func validateContext(available:Range<Int>,output:Range<Int>,totalFrames:Int,layers:Int=24) throws {
    let required=try requiredContext(output:output,totalFrames:totalFrames,layers:layers)
    guard available.lowerBound >= 0,available.upperBound <= totalFrames,available.lowerBound%5 == 0,
      available.upperBound%5 == 0 || available.upperBound == totalFrames,
      available.lowerBound <= required.lowerBound,available.upperBound >= required.upperBound else {
      throw H3CheckpointError.invalid("Fast H3 output needs its full noncausal native-block halo; independent five-latent decoding is not equivalent.")
    }
  }
}
