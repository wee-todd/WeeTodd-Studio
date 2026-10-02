import Foundation
import MLX
import LTX25Engine

/// The DFR canvas follows the released 24/32-pixel-frame seam policy. The
/// public frame count is retained separately so a padded tail can be trimmed.
public struct MLXDFRCanvas: Sendable {
  public let requestedFrames: Int
  public let frames: Int
  public let segmentFrames: Int
  public let slotFrames: [Int]

  public init(frames requestedFrames: Int) throws {
    guard (2...4097).contains(requestedFrames), (requestedFrames - 1) % 8 == 0 else {
      throw LTXError.invalid("DFR needs bounded 8n+1 output frames.")
    }
    let content = requestedFrames - 1
    let segment = [24, 32].min {
      let left = ($0 - content % $0) % $0
      let right = ($1 - content % $1) % $1
      return left == right ? $0 > $1 : left < right
    }!
    let padded = content + (segment - content % segment) % segment
    guard padded + 1 <= 4097 else { throw LTXError.invalid("DFR padded canvas exceeds its frame limit.") }
    self.requestedFrames = requestedFrames
    frames = padded + 1
    segmentFrames = segment
    slotFrames = stride(from: segment, through: padded, by: segment).map { $0 }
  }
}

/// Appended DFR tokens use a single ordered layout: generated video, optional
/// lower-resolution clean reference, then generated keyframe slots. The last
/// rows are the only rows that receive the checkpoint's learned slot marker.
public struct MLXDFRLayout: Sendable {
  public let geometry: AVGeometry
  public let slotFrames: [Int]
  public let reference: AVGeometry?
  public let endpointTokens: Int
  public let referenceTokens: Int
  public let slotTokens: Int
  public let videoTokens: Int
  public let positions: [Float]
  private let endpoints: MLXReferenceLayout?

  public init(geometry: AVGeometry, slotFrames: [Int], reference: AVGeometry? = nil,
    firstStrength: Float? = nil, lastStrength: Float? = nil,
    lastFrame:Int?=nil) throws {
    guard !slotFrames.isEmpty, slotFrames == Array(Set(slotFrames)).sorted(),
      slotFrames.allSatisfy({ $0 > 0 && $0 <= geometry.frames - 1 && $0 % 8 == 0 }),
      reference == nil || (reference!.frames == geometry.frames && reference!.fps == geometry.fps &&
        reference!.width * 2 == geometry.width && reference!.height * 2 == geometry.height) else {
      throw LTXError.invalid("DFR slots or half-resolution reference do not match the canvas.")
    }
    self.geometry = geometry
    self.slotFrames = slotFrames
    self.reference = reference
    guard (firstStrength != nil || lastStrength == nil),
      (lastFrame == nil || lastStrength != nil) else {
      throw LTXError.invalid("DFR last image needs a first image.")
    }
    endpoints = try firstStrength.map { try MLXReferenceLayout(geometry:geometry,
      firstStrength:$0,lastStrength:lastStrength,lastFrame:lastFrame) }
    endpointTokens = endpoints?.videoTokens ?? geometry.videoTokens
    referenceTokens = reference?.videoTokens ?? 0
    slotTokens = slotFrames.count * geometry.latentHeight * geometry.latentWidth
    videoTokens = endpointTokens + referenceTokens + slotTokens
    guard videoTokens <= 131_072 else { throw LTXError.invalid("DFR token layout exceeds video admission.") }
    var result = endpoints?.positions ?? geometry.videoPositions
    result.reserveCapacity(videoTokens * 3)
    if let reference {
      let source = reference.videoPositions
      for offset in stride(from: 0, to: source.count, by: 3) {
        result += [source[offset], source[offset + 1] * 2, source[offset + 2] * 2]
      }
    }
    for frame in slotFrames {
      let time = Float(Double(frame) + 0.5) / Float(geometry.fps)
      for h in 0..<geometry.latentHeight {
        for w in 0..<geometry.latentWidth {
          result += [time, Float(h * 32 + 16), Float(w * 32 + 16)]
        }
      }
    }
    positions = result
  }

  public func prepare(generated: MLXArray, first:MLXArray? = nil,last:MLXArray? = nil,
    reference cleanReference: MLXArray? = nil,
    slots initialSlots: MLXArray? = nil) throws -> (latent: MLXArray, condition: MLXVideoDenoiseCondition) {
    guard generated.dtype == .float32, generated.shape == [geometry.videoTokens, 128],
      MLX.isFinite(generated).all().item(Bool.self),
      (first != nil) == (endpoints != nil),
      (cleanReference != nil) == (reference != nil),
      cleanReference.map({ $0.dtype == .float32 && $0.shape == [referenceTokens, 128] &&
        MLX.isFinite($0).all().item(Bool.self) }) ?? true,
      initialSlots.map({ $0.dtype == .float32 && $0.shape == [slotTokens, 128] &&
        MLX.isFinite($0).all().item(Bool.self) }) ?? true else {
      throw LTXError.invalid("DFR generated, reference or slot latents differ from admitted layout.")
    }
    let anchored: (latent:MLXArray,condition:MLXVideoDenoiseCondition)
    if let endpoints,let first {
      anchored=try endpoints.prepare(generated:generated,first:first,last:last)
    } else {
      guard last == nil else { throw LTXError.invalid("DFR last image has no first image.") }
      anchored=(generated,try MLXVideoDenoiseCondition(
        clean:MLXArray.zeros([geometry.videoTokens,128]),
        mask:[Float](repeating:1,count:geometry.videoTokens)))
    }
    let slots = initialSlots ?? .zeros([slotTokens, 128])
    let components = [anchored.latent, cleanReference, slots].compactMap { $0 }
    let latent = concatenated(components, axis: 0)
    let clean = concatenated([anchored.condition.clean, cleanReference,
      MLXArray.zeros([slotTokens, 128])].compactMap { $0 }, axis: 0)
    let mask = anchored.condition.mask +
      [Float](repeating: 0, count: referenceTokens) + [Float](repeating: 1, count: slotTokens)
    eval(latent, clean)
    return (latent, try MLXVideoDenoiseCondition(clean: clean, mask: mask))
  }

  /// Generated slots are denoised tokens. The pipeline noiser starts them at
  /// the stage sigma, just like the main canvas; a zero/unnoised slot produces
  /// invalid carry-forward keyframe planes even when the main video looks fine.
  func noiseSlots(_ latent:MLXArray,noise:MLXArray,sigma:Float) throws -> MLXArray {
    guard latent.dtype == .float32,latent.shape == [videoTokens,128],
      noise.dtype == .float32,noise.shape == [slotTokens,128],
      sigma.isFinite,(0...1).contains(sigma) else {
      throw LTXError.invalid("DFR generated slot initialization differs from its stage layout.")
    }
    let start=videoTokens-slotTokens
    let seeded=latent[start..<videoTokens]*(1-sigma)+noise*sigma
    return concatenated([latent[0..<start],seeded],axis:0)
  }
}
