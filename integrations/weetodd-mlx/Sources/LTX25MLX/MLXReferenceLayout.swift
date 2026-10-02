import Foundation
import MLX
import LTX25Engine

/// One aligned frame replacement plus at most one appended last-frame reference.
/// This layout needs no inter-reference attention mask. Multiple appended
/// references, IC/MSR and arbitrary frame positions have separate contracts.
public struct MLXReferenceLayout:Sendable {
  public let geometry:AVGeometry
  public let firstStrength:Float
  public let lastStrength:Float?
  public let firstFrame:Int
  public let lastFrame:Int
  public var frameTokens:Int { geometry.latentHeight*geometry.latentWidth }
  public var videoTokens:Int { geometry.videoTokens+(lastStrength == nil ? 0 : frameTokens) }
  public var positions:[Float] {
    var result=geometry.videoPositions
    if lastStrength != nil {
      let time=Float(Double(lastFrame)+0.5)/Float(geometry.fps)
      for h in 0..<geometry.latentHeight { for w in 0..<geometry.latentWidth {
        result += [time,Float(h*32+16),Float(w*32+16)]
      } }
    }
    return result
  }
  public init(geometry:AVGeometry,firstStrength:Float,lastStrength:Float?,firstFrame:Int=0,
    lastFrame:Int?=nil) throws {
    let finalFrame=lastFrame ?? geometry.frames-1
    guard firstStrength.isFinite,(0...1).contains(firstStrength),
      lastStrength.map({ $0.isFinite && (0...1).contains($0) }) ?? true,
      lastStrength == nil || geometry.frames>1,
      firstFrame>=0,firstFrame<geometry.frames,firstFrame%8==0,
      lastFrame == nil || lastStrength != nil,
      finalFrame>=0,finalFrame<geometry.frames,finalFrame%8==0,
      lastStrength == nil || finalFrame>firstFrame,
      firstFrame == 0 || lastStrength == nil else {
      throw LTXError.invalid("Invalid aligned image frame, endpoint strengths or frame count.")
    }
    self.geometry=geometry;self.firstStrength=firstStrength;self.lastStrength=lastStrength
    self.firstFrame=firstFrame;self.lastFrame=finalFrame
    guard videoTokens<=131072 else { throw LTXError.invalid("Reference tokens exceed video admission.") }
  }
  public func validate(first:MLXArray,last:MLXArray?) throws {
    guard (last != nil) == (lastStrength != nil) else { throw LTXError.invalid("Endpoint images differ from the admitted task.") }
    for value in [first,last].compactMap({ $0 }) {
      guard value.dtype == .float32,value.shape == [frameTokens,128],MLX.isFinite(value).all().item(Bool.self) else {
        throw LTXError.invalid("Reference latent differs from this stage's image geometry.")
      }
    }
  }
  public func prepare(generated:MLXArray,first:MLXArray,last:MLXArray?) throws -> (latent:MLXArray,condition:MLXVideoDenoiseCondition) {
    try validate(first:first,last:last)
    guard generated.dtype == .float32,generated.shape == [geometry.videoTokens,128],MLX.isFinite(generated).all().item(Bool.self) else {
      throw LTXError.invalid("Invalid generated reference-stage state.")
    }
    // Match the existing distilled recipe: scalar noise blend occurs before
    // conditioning; anchors replace/append clean tokens at both resolutions.
    // Strength controls the denoise mask, not an extra initial interpolation.
    let start=firstFrame/8*frameTokens
    var parts:[MLXArray]=[],clean:[MLXArray]=[]
    if start>0 {
      parts.append(generated[0..<start]);clean.append(MLXArray.zeros([start,128]))
    }
    parts.append(first.reshaped(first.shape));clean.append(first.reshaped(first.shape))
    let end=start+frameTokens
    if end<geometry.videoTokens {
      parts.append(generated[end..<geometry.videoTokens])
      clean.append(MLXArray.zeros([geometry.videoTokens-end,128]))
    }
    var mask=[Float](repeating:1,count:start)
    mask += [Float](repeating:1-firstStrength,count:frameTokens)
    mask += [Float](repeating:1,count:geometry.videoTokens-end)
    if let last,let strength=lastStrength {
      parts.append(last.reshaped(last.shape));clean.append(last.reshaped(last.shape))
      mask += [Float](repeating:1-strength,count:frameTokens)
    }
    let latent=concatenated(parts,axis:0),anchors=concatenated(clean,axis:0)
    eval(latent,anchors)
    return (latent,try MLXVideoDenoiseCondition(clean:anchors,mask:mask))
  }
}
