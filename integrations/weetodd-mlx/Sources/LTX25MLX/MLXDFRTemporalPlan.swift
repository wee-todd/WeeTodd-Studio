import Foundation
import LTX25Engine

public struct MLXDFRTemporalTile: Sendable, Equatable {
  public let pixelStart: Int
  public let pixelEnd: Int
  public let latentStart: Int
  public let latentEnd: Int
  public let anchorFrames: [Int]
  public let slotFrames: [Int]
  public let dropLatentPrefix: Int

  public var frames: Int { pixelEnd - pixelStart + 1 }
  public var latentFrames: Int { latentEnd - latentStart + (dropLatentPrefix > 0 ? 1 : 0) }
}

/// Each tile owns a disjoint latent interval. A later tile starts on the
/// generated midpoint plane before its seam and pins the already sampled
/// cells through that seam; those prefix cells are dropped during stitching.
public enum MLXDFRTemporalPlan {
  public static func tiles(seams: [Int], frames: Int, maximumTiles: Int) throws -> [MLXDFRTemporalTile] {
    guard !seams.isEmpty, seams == Array(Set(seams)).sorted(), seams.last == frames - 1,
      (2...4097).contains(frames), maximumTiles > 0 else {
      throw LTXError.invalid("Temporal DFR needs ordered seams ending on its output frame.")
    }
    let boundaries = [0] + seams
    let spans = zip(boundaries, boundaries.dropFirst()).map { $1 - $0 }
    guard spans.allSatisfy({ $0 >= 16 && $0 % 8 == 0 }) else {
      throw LTXError.invalid("Temporal DFR seam intervals need at least two aligned latent steps.")
    }
    let count = min(maximumTiles, spans.count)
    let base = spans.count / count, extra = spans.count % count
    var start = 0
    var result: [MLXDFRTemporalTile] = []
    for index in 0..<count {
      let end = start + base + (index < extra ? 1 : 0)
      let seam = boundaries[start], endFrame = boundaries[end]
      let slots = (start..<end).map { (boundaries[$0] + boundaries[$0 + 1]) / 2 }
      let allSlots = (0..<spans.count).map { (boundaries[$0] + boundaries[$0 + 1]) / 2 }
      let plane = index == 0 ? 0 : allSlots.filter { $0 < seam }.max()!
      let latentStart = index == 0 ? 0 : plane / 8 + 1
      let drop = index == 0 ? 0 : 1 + (seam - plane) / 8
      let anchors = ((start + 1)...end).map { boundaries[$0] }
      result.append(MLXDFRTemporalTile(pixelStart: plane, pixelEnd: endFrame,
        latentStart: latentStart, latentEnd: endFrame / 8 + 1,
        anchorFrames: anchors, slotFrames: slots, dropLatentPrefix: drop))
      start = end
    }
    return result
  }

  public static func outputFrames(inputFrames: Int, rounds: Int) throws -> Int {
    guard (2...4097).contains(inputFrames), (inputFrames - 1) % 8 == 0,
      (0...2).contains(rounds) else {
      throw LTXError.invalid("Temporal DFR requires 8n+1 frames and zero to two rounds.")
    }
    let frames = (inputFrames - 1) * (1 << rounds) + 1
    guard frames <= 4097 else { throw LTXError.invalid("Temporal DFR output exceeds its frame limit.") }
    return frames
  }

  public static func conditioningFPS(_ playbackFPS: Double) throws -> Double {
    guard playbackFPS.isFinite, playbackFPS > 0, playbackFPS <= 120 else {
      throw LTXError.invalid("Temporal DFR playback FPS is invalid.")
    }
    return playbackFPS > 30 ? 60 : playbackFPS
  }

  /// Conservative header-only admission for every temporal tile. Endpoint
  /// references may replace a generated slot, so counting them separately is
  /// an upper bound on both tokens and per-token modulation workspace.
  public static func admissionConfigurations(geometry: AVGeometry,
    requestedFrames: Int, slots originalSlots: [Int], rounds: Int,
    endpointCount: Int) throws -> [AVBlockConfiguration] {
    guard (0...2).contains(endpointCount),
      requestedFrames <= geometry.frames else {
      throw LTXError.invalid("Temporal DFR endpoint count or requested canvas is invalid.")
    }
    _ = try outputFrames(inputFrames:geometry.frames,rounds:rounds)
    guard rounds > 0 else { return [] }
    var seams=originalSlots, frames=geometry.frames, fps=geometry.fps
    var result:[AVBlockConfiguration]=[]
    for round in 1...rounds {
      frames=try outputFrames(inputFrames:frames,rounds:1)
      fps *= 2
      seams=seams.map { $0*2 }
      let tiles=try tiles(seams:seams,frames:frames,maximumTiles:1 << round)
      let conditioning=try conditioningFPS(fps)
      let frameTokens=geometry.latentHeight*geometry.latentWidth
      for tile in tiles {
        let tileGeometry=try AVGeometry(width:geometry.width,height:geometry.height,
          frames:tile.frames,fps:conditioning)
        let extraAnchors=tile.anchorFrames.count + endpointCount
        let videoTokens=tileGeometry.videoTokens+(extraAnchors+tile.slotFrames.count)*frameTokens
        let audioTokens=tileGeometry.audioFrames
        result.append(try AVBlockConfiguration(videoTokens:videoTokens,
          audioTokens:audioTokens,textTokens:1024))
      }
      let midpoints=tiles.flatMap(\.slotFrames)
      seams=Array(Set(seams+midpoints)).sorted()
    }
    return result
  }
}
