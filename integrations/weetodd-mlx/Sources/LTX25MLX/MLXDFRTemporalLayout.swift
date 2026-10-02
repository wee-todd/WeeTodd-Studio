import Foundation
import MLX
import LTX25Engine

/// A temporal tile has generated video, optional clean image/keyframe anchors,
/// then generated midpoint slots. Only the trailing slot rows get the learned
/// keyframe marker. Positions use the clamped conditioning clock.
struct MLXDFRTemporalLayout {
  struct Anchor {
    let frame: Int
    let latent: MLXArray
    let strength: Float
    let replace: Bool
  }
  let geometry: AVGeometry
  let slots: [Int]
  let anchors: [Anchor]
  let videoTokens: Int
  let slotTokens: Int
  let positions: [Float]

  init(geometry: AVGeometry, slots: [Int], anchors: [Anchor]) throws {
    guard !slots.isEmpty, slots == Array(Set(slots)).sorted(),
      slots.allSatisfy({ $0 > 0 && $0 < geometry.frames && $0 % 8 == 0 }),
      anchors.map(\.frame) == Array(Set(anchors.map(\.frame))).sorted(),
      anchors.allSatisfy({ $0.frame >= 0 && $0.frame < geometry.frames && $0.frame % 8 == 0 &&
        $0.strength.isFinite && (0...1).contains($0.strength) &&
        (!$0.replace || $0.frame == 0) }) else {
      throw LTXError.invalid("Temporal DFR slots or anchors differ from the aligned tile.")
    }
    let frameTokens = geometry.latentHeight * geometry.latentWidth
    guard anchors.allSatisfy({ $0.latent.dtype == .float32 &&
      $0.latent.shape == [frameTokens, 128] }) else {
      throw LTXError.invalid("Temporal DFR anchor latent differs from tile geometry.")
    }
    self.geometry = geometry
    self.slots = slots
    self.anchors = anchors
    slotTokens = slots.count * frameTokens
    videoTokens = geometry.videoTokens + anchors.filter({ !$0.replace }).count * frameTokens + slotTokens
    guard videoTokens <= 131_072 else { throw LTXError.invalid("Temporal DFR tile exceeds video token admission.") }
    var coordinates = geometry.videoPositions
    func append(_ frame: Int) {
      let time = Float(Double(frame) + 0.5) / Float(geometry.fps)
      for h in 0..<geometry.latentHeight {
        for w in 0..<geometry.latentWidth {
          coordinates += [time, Float(h * 32 + 16), Float(w * 32 + 16)]
        }
      }
    }
    for anchor in anchors where !anchor.replace { append(anchor.frame) }
    for slot in slots { append(slot) }
    positions = coordinates
  }

  func prepare(generated: MLXArray, initialSlots: MLXArray,
    pinnedPrefix: MLXArray? = nil) throws
    -> (latent: MLXArray, condition: MLXVideoDenoiseCondition) {
    let frameTokens = geometry.latentHeight * geometry.latentWidth
    guard generated.dtype == .float32, generated.shape == [geometry.videoTokens, 128],
      initialSlots.dtype == .float32, initialSlots.shape == [slotTokens, 128] else {
      throw LTXError.invalid("Temporal DFR generated or midpoint latent has incorrect rows.")
    }
    let pinnedTokens = pinnedPrefix?.shape.first ?? 0
    guard pinnedTokens <= geometry.videoTokens,
      pinnedTokens % frameTokens == 0,
      pinnedPrefix.map({ $0.dtype == .float32 && $0.shape == [pinnedTokens,128] }) ?? true,
      pinnedPrefix == nil || !anchors.contains(where: \.replace) else {
      throw LTXError.invalid("Temporal DFR pinned prefix differs from tile geometry.")
    }
    var base = generated
    var cleanBase = MLXArray.zeros([geometry.videoTokens, 128])
    var mask = [Float](repeating: 1, count: geometry.videoTokens)
    if let pinnedPrefix {
      base = concatenated([pinnedPrefix,generated[pinnedTokens..<geometry.videoTokens]],axis:0)
      cleanBase = concatenated([pinnedPrefix,cleanBase[pinnedTokens..<geometry.videoTokens]],axis:0)
      for index in 0..<pinnedTokens { mask[index] = 0 }
    }
    if let first = anchors.first(where: \.replace) {
      base = concatenated([first.latent, generated[frameTokens..<geometry.videoTokens]], axis: 0)
      cleanBase = concatenated([first.latent, cleanBase[frameTokens..<geometry.videoTokens]], axis: 0)
      for index in 0..<frameTokens { mask[index] = 1 - first.strength }
    }
    var parts = [base], clean = [cleanBase]
    for anchor in anchors where !anchor.replace {
      parts.append(anchor.latent)
      clean.append(anchor.latent)
      mask += [Float](repeating: 1 - anchor.strength, count: frameTokens)
    }
    parts.append(initialSlots)
    clean.append(.zeros([slotTokens, 128]))
    mask += [Float](repeating: 1, count: slotTokens)
    let latent = concatenated(parts, axis: 0), reference = concatenated(clean, axis: 0)
    eval(latent, reference)
    return (latent, try MLXVideoDenoiseCondition(clean: reference, mask: mask))
  }
}

enum MLXDFRFrozenAudio {
  static func tile(_ source: MLXArray, pixelStart: Int, frames: Int,
    playbackFPS: Double, sourceSeconds: Double) throws -> (latent: MLXArray, positions: MLXArray) {
    guard source.dtype == .float32, source.shape.count == 2, source.shape[1] == 128,
      source.shape[0] > 0, pixelStart >= 0, frames > 0,
      sourceSeconds.isFinite, sourceSeconds > 0 else {
      throw LTXError.invalid("Temporal DFR needs packed stage-one audio and finite timing.")
    }
    let fps = try MLXDFRTemporalPlan.conditioningFPS(playbackFPS)
    let count = Int(ceil(Double(frames) / fps * 25))
    guard count > 0 && count <= 1501 else { throw LTXError.invalid("Temporal DFR audio window is invalid.") }
    let sourceCount = source.shape[0], samples = source.asArray(Float.self)
    let start = Double(pixelStart) / playbackFPS / sourceSeconds * Double(sourceCount)
    let span = Double(frames) / playbackFPS / sourceSeconds * Double(sourceCount)
    var output = [Float](repeating: 0, count: count * 128)
    for index in 0..<count {
      let position = min(max(start + Double(index) * span / Double(count), 0), Double(sourceCount - 1))
      let low = Int(floor(position)), high = min(low + 1, sourceCount - 1)
      let fraction = Float(position - Double(low))
      for channel in 0..<128 {
        output[index * 128 + channel] = samples[low * 128 + channel] * (1 - fraction)
          + samples[high * 128 + channel] * fraction
      }
    }
    let positions = (0..<count).map { index -> Float in
      let begin = Float(max(0, index * 4 - 3)) * 0.01
      let end = Float(index * 4 + 1) * 0.01
      return (begin + end) / 2
    }
    return (MLXArray(output, [count, 128]), MLXArray(positions, [count, 1]))
  }
}
