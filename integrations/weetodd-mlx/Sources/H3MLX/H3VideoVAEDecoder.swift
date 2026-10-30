import Foundation
import MLX

/// Temporal H3 video VAE decode with the released 17-frame/5-token overlap
/// contract. The chunk callback receives complete frames for live preview or
/// streaming publication, and can discard them before the next decode.
public enum H3VideoVAEDecoder {
  private struct TileAxis {
    let starts: [Int]
    let lengths: [Int]
    let overlaps: [Int]
  }

  private static func splitTiles(_ length: Int) -> TileAxis {
    let tileSize = 256
    let minimumOverlap = 64
    let ratio = 16
    if length <= tileSize {
      return TileAxis(starts: [0], lengths: [length], overlaps: [])
    }
    var count = (length + tileSize - 1) / tileSize
    while tileSize * count - minimumOverlap * (count - 1) < length {
      count += 1
    }
    var overlaps = Array(repeating: minimumOverlap, count: count - 1)
    let remaining = tileSize * count - overlaps.reduce(0, +) - length
    for index in 0..<(remaining / ratio) {
      overlaps[index % overlaps.count] += ratio
    }
    var starts = [0]
    for overlap in overlaps {
      starts.append(starts.last! + tileSize - overlap)
    }
    return TileAxis(starts: starts,
      lengths: Array(repeating: tileSize, count: count), overlaps: overlaps)
  }

  /// Header/shape-only expanded canvas check reuses the actual decoder's
  /// spatial partition. Each 17-frame temporal clip stays in bounded 256px
  /// spatial tiles; no full-2MP projection/convolution is admitted here.
  public static func preflightSpatial(geometry: H3Geometry) throws -> Int {
    try geometry.canvasAdmission.validate(width: geometry.width,height: geometry.height)
    try geometry.canvasAdmission.validatePackedRows(geometry.videoRows + geometry.audioRows + 1)
    let x = splitTiles(geometry.width), y = splitTiles(geometry.height)
    let count = x.starts.count * y.starts.count
    guard count > 0, count <= 64,
      x.lengths.allSatisfy({ $0 <= 256 }), y.lengths.allSatisfy({ $0 <= 256 }),
      UInt64(4 * 8 * 16 * 16 * 2048) < UInt64(Int32.max) else {
      throw H3CheckpointError.invalid("Expanded H3 spatial decoder exceeds the tile or 32-bit activation bound.")
    }
    return count
  }

  public static func preflightRegenerate(geometry:H3Geometry) throws -> Int {
    guard geometry.canvasAdmission == .regenerate,geometry.videoLatentFrames >= 7 else {
      throw H3CheckpointError.invalid("Released regeneration video decode needs at least22 padded frames and the explicit regenerate canvas.")
    }
    try geometry.canvasAdmission.validate(width:geometry.width,height:geometry.height)
    let count=splitTiles(geometry.width).starts.count*splitTiles(geometry.height).starts.count
    guard (1...512).contains(count) else { throw H3CheckpointError.invalid("Regenerate released decode exceeds512 bounded spatial tiles.") }
    return count
  }
  public static func decodeRegenerateChunks(checkpointURL:URL,latent:MLXArray,geometry:H3Geometry,
    onChunk:(MLXArray) throws -> Void) throws {
    _ = try preflightRegenerate(geometry:geometry)
    guard latent.shape == [1,geometry.videoLatentFrames,geometry.height/16,geometry.width/16,24] else {
      throw H3CheckpointError.invalid("Regenerate decoder latent geometry differs from its admitted grid.")
    }
    try decodeChunks(checkpointURL:checkpointURL,latent:latent,retainWeights:true,maximumSpatialTiles:512,onChunk:onChunk)
  }

  private static func blend(_ previous: MLXArray, _ current: MLXArray,
    extent: Int, axis: Int) -> MLXArray {
    let count = min(extent, previous.shape[axis], current.shape[axis])
    guard count > 0 else { return current }
    var weightShape = Array(repeating: 1, count: current.ndim)
    weightShape[axis] = count
    let position = MLXArray((0..<count).map(Float.init)).reshaped(weightShape)
    let firstWeight = 1 - position / Float(count)
    let secondWeight = position / Float(count)
    let tail: MLXArray
    let head: MLXArray
    let rest: MLXArray
    if axis == 2 {
      tail = previous[0..<1, 0..<previous.shape[1],
        (previous.shape[2] - count)..<previous.shape[2], 0..<previous.shape[3], 0..<3]
      head = current[0..<1, 0..<current.shape[1], 0..<count, 0..<current.shape[3], 0..<3]
      rest = current[0..<1, 0..<current.shape[1], count..<current.shape[2],
        0..<current.shape[3], 0..<3]
    } else {
      tail = previous[0..<1, 0..<previous.shape[1], 0..<previous.shape[2],
        (previous.shape[3] - count)..<previous.shape[3], 0..<3]
      head = current[0..<1, 0..<current.shape[1], 0..<current.shape[2], 0..<count, 0..<3]
      rest = current[0..<1, 0..<current.shape[1], 0..<current.shape[2],
        count..<current.shape[3], 0..<3]
    }
    let leading = tail * firstWeight + head * secondWeight
    return count == current.shape[axis] ? leading : concatenated([leading, rest], axis: axis)
  }

  private static func decodeClip(checkpointURL: URL, latent: MLXArray,
    session: H3VideoVAEDecodeSession?, spatialBatchSize: Int,
    precision: H3VideoDecodePrecision,maximumSpatialTiles:Int=64) throws -> MLXArray {
    let y = splitTiles(latent.shape[2] * 16)
    let x = splitTiles(latent.shape[3] * 16)
    let count = y.starts.count * x.starts.count
    guard count <= maximumSpatialTiles else {
      throw H3CheckpointError.invalid("H3 video VAE exceeds its selected bounded spatial tile count.")
    }
    if count == 1 {
      return try H3VideoVAETileDecoder.decode(checkpointURL: checkpointURL,
        latent: latent, session: session, precision:precision, observe: { _, _ in })
    }
    var inputs: [MLXArray] = []
    for row in y.starts.indices {
      for column in x.starts.indices {
        inputs.append(latent[0..<1, 0..<latent.shape[1],
          (y.starts[row] / 16)..<((y.starts[row] + y.lengths[row]) / 16),
          (x.starts[column] / 16)..<((x.starts[column] + x.lengths[column]) / 16),
          0..<24])
      }
    }
    var outputs: [MLXArray] = []
    for start in stride(from: 0, to: count, by: spatialBatchSize) {
      try Task.checkCancellation()
      let end = min(start + spatialBatchSize, count)
      let batch = concatenated(Array(inputs[start..<end]), axis: 0)
      let decoded = try H3VideoVAETileDecoder.decode(
        checkpointURL: checkpointURL, latent: batch, session: session, precision:precision,
        observe: { _, _ in })
      for index in 0..<(end - start) {
        let tile = decoded[index..<(index + 1), 0..<decoded.shape[1],
          0..<decoded.shape[2], 0..<decoded.shape[3], 0..<3]
        eval(tile)
        outputs.append(tile)
      }
    }
    var rows: [MLXArray] = []
    for row in y.starts.indices {
      var parts: [MLXArray] = []
      for column in x.starts.indices {
        var tile = outputs[row * x.starts.count + column]
        if row > 0 {
          tile = blend(outputs[(row - 1) * x.starts.count + column], tile,
            extent: y.overlaps[row - 1], axis: 2)
        }
        if column > 0 {
          tile = blend(outputs[row * x.starts.count + column - 1], tile,
            extent: x.overlaps[column - 1], axis: 3)
        }
        if row < y.overlaps.count {
          tile = tile[0..<1, 0..<tile.shape[1],
            0..<(tile.shape[2] - y.overlaps[row]), 0..<tile.shape[3], 0..<3]
        }
        if column < x.overlaps.count {
          tile = tile[0..<1, 0..<tile.shape[1], 0..<tile.shape[2],
            0..<(tile.shape[3] - x.overlaps[column]), 0..<3]
        }
        parts.append(tile)
      }
      rows.append(concatenated(parts, axis: 3))
    }
    let result = concatenated(rows, axis: 2)
    eval(result)
    return result
  }

  public static func decode(checkpointURL: URL, latent: MLXArray,
    onChunk: (MLXArray) throws -> Void = { _ in }) throws -> MLXArray {
    var chunks: [MLXArray] = []
    try decodeChunks(checkpointURL: checkpointURL, latent: latent) { chunk in
      chunks.append(chunk)
      try onChunk(chunk)
    }
    return concatenated(chunks, axis: 1)
  }

  /// A complete still window uses one or two temporal latents, with no video
  /// chunk padding or audio decoder. Spatial tiles use the same decoder core.
  public static func decodeStill(checkpointURL: URL, latent: MLXArray,
    mode: H3StillImageMode, selection: H3ImageFrameSelection = .middle,
    memoryMode: H3VideoDecodeMemoryMode? = nil,
    precision: H3VideoDecodePrecision = .float32) throws -> MLXArray {
    let layout = try H3VideoVAELayout(url: checkpointURL)
    try mode.validateDecoder(imageOnly: layout.imageOnly)
    guard latent.ndim == 5, latent.shape[0] == 1,
      latent.shape[1] == mode.latentFrames, latent.shape[4] == 24,
      (2...256).contains(latent.shape[2]), (2...256).contains(latent.shape[3]),
      latent.dtype == .float32 else {
      throw H3CheckpointError.invalid("Invalid H3 still-image latent geometry.")
    }
    try precision.validate(memoryMode: memoryMode)
    return try H3VideoVAEDecodeSession.withSession(checkpointURL: checkpointURL,
      memoryMode: memoryMode, precision: precision) { session in
      let decoded = try decodeClip(checkpointURL: checkpointURL, latent: latent,
        session: session, spatialBatchSize: H3VideoDecodeMemoryMode.spatialBatchSize(for: memoryMode),
        precision: precision)
      let selected = try mode.select(decoded: decoded, selection: selection)
      eval(selected)
      return selected
    }
  }

  public static func decodeChunks(checkpointURL: URL, latent: MLXArray,
    memoryMode: H3VideoDecodeMemoryMode? = nil,
    precision: H3VideoDecodePrecision = .float32,
    onChunk: (MLXArray) throws -> Void) throws {
    try precision.validate(memoryMode: memoryMode)
    try decodeChunks(checkpointURL: checkpointURL, latent: latent,
      retainWeights: true, memoryMode: memoryMode, precision: precision, onChunk: onChunk)
  }

  /// The legacy streamed-weight path remains available internally for exact
  /// decoder-only qualification against the same saved latent input.
  static func decodeChunks(checkpointURL: URL, latent: MLXArray,
    retainWeights: Bool,
    memoryMode: H3VideoDecodeMemoryMode? = nil,
    precision: H3VideoDecodePrecision = .defaultPrecision,
    spatialBatchSize: Int? = nil,
    maximumSpatialTiles:Int=64,
    allocationCacheLimitBytes: Int = H3VideoVAEDecodeSession.defaultAllocationCacheLimitBytes,
    groupedStageLoading: Bool? = nil,
    onSessionClosed: (H3VideoVAEDecodeSession.Statistics) -> Void = { _ in },
    onChunk: (MLXArray) throws -> Void) throws {
    try H3VideoVAEDecodeSession.validateAllocationCacheLimit(allocationCacheLimitBytes,
      memoryMode: memoryMode)
    let resolvedGroup = groupedStageLoading ?? retainWeights
    let spatialBatchSize = spatialBatchSize ?? H3VideoDecodeMemoryMode.spatialBatchSize(for:memoryMode)
    guard memoryMode == nil || retainWeights else {
      throw H3CheckpointError.invalid("Selected video memory mode requires a resident decoder session.")
    }
    guard !resolvedGroup || (retainWeights
      && allocationCacheLimitBytes == H3VideoVAEDecodeSession.defaultAllocationCacheLimitBytes) else {
      throw H3CheckpointError.invalid("Grouped H3 video loading requires a resident128MiB session.")
    }
    guard [64,512].contains(maximumSpatialTiles),(1...4).contains(spatialBatchSize), latent.ndim == 5, latent.shape[0] == 1,
      (7...128).contains(latent.shape[1]),
      (1...256).contains(latent.shape[2]),
      (1...256).contains(latent.shape[3]),
      latent.shape[4] == 24, latent.dtype == .float32 else {
      throw H3CheckpointError.invalid("Invalid H3 video VAE temporal decode geometry.")
    }
    try precision.validateHalfCastInput(latent)
    try H3VideoVAELayout(url: checkpointURL).validateVideoUse()
    if retainWeights {
      try H3VideoVAEDecodeSession.withSession(checkpointURL: checkpointURL,
        memoryMode: memoryMode, precision:precision, allocationCacheLimitBytes: allocationCacheLimitBytes,
        groupedStageLoading: resolvedGroup,
        onClose: onSessionClosed) { session in
        try decodeChunks(checkpointURL: checkpointURL, latent: latent,
          session: session, spatialBatchSize: spatialBatchSize, precision:precision, maximumSpatialTiles:maximumSpatialTiles,onChunk: onChunk)
      }
    } else {
      _ = try H3VideoVAELayout(url: checkpointURL)
      try decodeChunks(checkpointURL: checkpointURL, latent: latent,
        session: nil, spatialBatchSize: spatialBatchSize, precision:precision, maximumSpatialTiles:maximumSpatialTiles,onChunk: onChunk)
    }
  }

  private static func decodeChunks(checkpointURL: URL, latent: MLXArray,
    session: H3VideoVAEDecodeSession?, spatialBatchSize: Int,
    precision: H3VideoDecodePrecision,maximumSpatialTiles:Int,
    onChunk: (MLXArray) throws -> Void) throws {
    let originalTokens = latent.shape[1]
    let chunkTokens = 5
    let tokenDrop = 3
    let temporalRatio = 4
    let chunkFrames = chunkTokens * temporalRatio
    let tokenOverlap = 2
    let framePrePadding = 3
    let frameOverlap = 5
    let padding = (chunkTokens - (originalTokens + tokenDrop) % chunkTokens) % chunkTokens
    let totalTokens = originalTokens + padding
    let chunkCount = (originalTokens + tokenDrop + padding) / chunkTokens - 1
    guard chunkCount >= 1 else {
      throw H3CheckpointError.invalid("Too few H3 video latent frames to decode.")
    }
    let padded: MLXArray
    if padding > 0 {
      let tail = latent[0..<1, (originalTokens - 1)..<originalTokens,
        0..<latent.shape[2], 0..<latent.shape[3], 0..<24]
      padded = concatenated([latent] + Array(repeating: tail, count: padding), axis: 1)
    } else {
      padded = latent
    }
    func blend(_ previous: MLXArray, _ current: MLXArray) -> MLXArray {
      let extent = min(frameOverlap, previous.shape[1], current.shape[1])
      guard extent > 0 else { return current }
      let position = MLXArray((0..<extent).map(Float.init))
        .reshaped([1, extent, 1, 1, 1])
      let firstWeight = 1 - position / Float(extent)
      let secondWeight = position / Float(extent)
      let tail = previous[0..<1, (previous.shape[1] - extent)..<previous.shape[1],
        0..<previous.shape[2], 0..<previous.shape[3], 0..<3]
      let head = current[0..<1, 0..<extent,
        0..<current.shape[2], 0..<current.shape[3], 0..<3]
      let blended = tail * firstWeight + head * secondWeight
      return extent == current.shape[1] ? blended
        : concatenated([blended,
          current[0..<1, extent..<current.shape[1],
            0..<current.shape[2], 0..<current.shape[3], 0..<3]], axis: 1)
    }
    var pending: MLXArray?
    var overlap: MLXArray?
    for index in 0..<chunkCount {
      try Task.checkCancellation()
      try session?.checkUnchanged()
      let start = index * chunkTokens
      let end = start + chunkTokens + tokenOverlap
      guard end <= totalTokens else {
        throw H3CheckpointError.invalid("H3 video decode chunk extends beyond latent frames.")
      }
      let clip = try decodeClip(checkpointURL: checkpointURL,
        latent: padded[0..<1, start..<end,
          0..<latent.shape[2], 0..<latent.shape[3], 0..<24], session: session,
        spatialBatchSize: spatialBatchSize,precision:precision,maximumSpatialTiles:maximumSpatialTiles)
      for part in 0..<2 {
        let frameStart = part * chunkFrames
        let frameEnd = min(frameStart + chunkFrames, clip.shape[1])
        let chunk = clip[0..<1, (frameStart + framePrePadding)..<frameEnd,
          0..<clip.shape[2], 0..<clip.shape[3], 0..<3]
        if part == 0 {
          let ready = overlap.map { blend($0, chunk) } ?? chunk
          if let pending { try onChunk(pending) }
          pending = ready
        } else {
          overlap = chunk
        }
      }
    }
    if let overlap {
      if let pending { try onChunk(pending) }
      pending = overlap
    }
    guard var last = pending else {
      throw H3CheckpointError.invalid("H3 video VAE produced no temporal chunks.")
    }
    if padding > 0 {
      let padFrames = (0..<padding).reduce(0) { total, offset in
        total + ((originalTokens + offset) % chunkTokens == 0 ? 1 : temporalRatio)
      }
      guard padFrames < last.shape[1] else {
        throw H3CheckpointError.invalid("H3 video VAE trailing-frame trim exceeds its final chunk.")
      }
      last = last[0..<1, 0..<(last.shape[1] - padFrames),
        0..<last.shape[2], 0..<last.shape[3], 0..<3]
    }
    eval(last)
    try onChunk(last)
    try session?.checkUnchanged()
    try Task.checkCancellation()
  }
}
