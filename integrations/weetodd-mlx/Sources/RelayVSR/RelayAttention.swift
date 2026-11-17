import Foundation
import MLX

struct RelayKV {
  let key: MLXArray
  let value: MLXArray
}

struct RelayRotary {
  let cosine: MLXArray
  let sine: MLXArray

  init(height: Int, width: Int, time: Int, axes: [Int]) {
    let half = axes.reduce(0, +) / 2
    let angles: [Float] = (0..<(height * width)).flatMap { pixel in
      zip([time, pixel / width, pixel % width], axes).flatMap { position, channels in
        (0..<(channels / 2)).map { frequency in
          Float(position) * pow(10_000, -Float(frequency * 2) / Float(channels))
        }
      }
    }
    let angle = MLXArray(angles, [height * width, 1, half])
    cosine = cos(angle)
    sine = sin(angle)
  }

  func apply(_ x: MLXArray) -> MLXArray {
    let pairs = x.reshaped([x.dim(0), x.dim(1), x.dim(2) / 2, 2]).asType(.float32)
    let even = pairs[.ellipsis, 0]
    let odd = pairs[.ellipsis, 1]
    return stacked([even * cosine - odd * sine, even * sine + odd * cosine], axis: -1)
      .reshaped(x.shape).asType(x.dtype)
  }
}

struct RelayAttentionPlan {
  private struct Tile {
    let queries: MLXArray
    let keys: MLXArray
    let mask: MLXArray
    let blockTypes: MLXArray?
  }

  private let tiles: [Tile]
  private let restoreOrder: MLXArray?
  private let wholeGrid: Bool
  private let headDimension: Int

  init(height: Int, width: Int, windows: [RelayWindow], headDimension: Int = 128) {
    self.headDimension = headDimension
    let count = height * width
    wholeGrid = windows.allSatisfy { window in
      window.range(at: 0, extent: height, size: window.height).count == height &&
        window.range(at: height - 1, extent: height, size: window.height).count == height &&
        window.range(at: 0, extent: width, size: window.width).count == width &&
        window.range(at: width - 1, extent: width, size: window.width).count == width
    }
    if wholeGrid {
      tiles = []
      restoreOrder = nil
      return
    }
    var tiles: [Tile] = []
    var positions: [Int] = []
    for top in stride(from: 0, to: height, by: 8) {
      for left in stride(from: 0, to: width, by: 16) {
        let rows = top..<min(height, top + 8)
        let columns = left..<min(width, left + 16)
        let queries = rows.flatMap { row in columns.map { row * width + $0 } }
        var keys: [Int32] = []
        var masks: [MLXArray] = []
        var blockPlanes: [RelayBlockPlane] = []
        for (plane, window) in windows.enumerated() {
          let y = queries.map { window.range(at: $0 / width, extent: height, size: window.height) }
          let x = queries.map { window.range(at: $0 % width, extent: width, size: window.width) }
          let vertical = y[0].lowerBound..<y[y.count - 1].upperBound
          let horizontal = x[0].lowerBound..<x[x.count - 1].upperBound
          blockPlanes.append(RelayBlockPlane(rows: vertical, columns: horizontal, window: window))
          var keyRows: [Int32] = []
          var keyColumns: [Int32] = []
          for row in vertical {
            for column in horizontal {
              keys.append(Int32(plane * count + row * width + column))
              keyRows.append(Int32(row))
              keyColumns.append(Int32(column))
            }
          }
          masks.append(Self.axisMask(keyRows, within: y) .&& Self.axisMask(keyColumns, within: x))
        }
        let mask = concatenated(masks, axis: 1)
        let blockPlan = try? RelayBlockSkippingPlan(queries: queries, height: height, width: width,
          planes: blockPlanes, keyBlockSize: headDimension == 64 ? 32 : 16)
        let types = blockPlan.map { MLXArray($0.classes.flatMap { $0 }, [$0.classes.count, $0.classes[0].count]) }
        tiles.append(Tile(queries: MLXArray(queries.map(Int32.init)), keys: MLXArray(keys), mask: mask, blockTypes: types))
        positions += queries
      }
    }
    self.tiles = tiles
    var inverse = Array(repeating: Int32(0), count: count)
    for (index, pixel) in positions.enumerated() { inverse[pixel] = Int32(index) }
    restoreOrder = MLXArray(inverse)
  }

  private static func axisMask(_ coordinates: [Int32], within bounds: [Range<Int>]) -> MLXArray {
    let positions = MLXArray(coordinates).reshaped([1, -1])
    let lower = MLXArray(bounds.map { Int32($0.lowerBound) }).reshaped([-1, 1])
    let upper = MLXArray(bounds.map { Int32($0.upperBound) }).reshaped([-1, 1])
    return (positions .>= lower) .&& (positions .< upper)
  }

  func callAsFunction(_ query: MLXArray, _ planes: [RelayKV]) throws -> MLXArray {
    try evaluate(query, planes, useBlockSkipping: true,
      asynchronous: RelaySkippedAttention.runtimeQualified)
  }

  /// Conservative two-group reserve, without allocating tile arrays or models.
  static func queuedWorkspaceBytes(height h: Int, width w: Int, windows: [RelayWindow],
    queryChannels: Int, keyChannels: Int, scalarBytes: Int) -> UInt64 {
    let whole = windows.allSatisfy { window in
      [0,h-1].allSatisfy { window.range(at:$0,extent:h,size:window.height).count == h }
        && [0,w-1].allSatisfy { window.range(at:$0,extent:w,size:window.width).count == w }
    }
    if whole { return 0 }
    let batch = min(8, (h+7)/8 * ((w+15)/16))
    let queries = min(h,8) * min(w,16)
    let keys = windows.reduce(0) { $0 + min(h,$1.height+7) * min(w,$1.width+15) }
    let bytes = UInt64(batch) * UInt64(queries*queryChannels*scalarBytes*2
      + keys*keyChannels*scalarBytes*2 + queries*keys) * 2
    return min(bytes,4*1024*1024*1024)
  }

  func evaluate(_ query: MLXArray, _ planes: [RelayKV], useBlockSkipping: Bool,
    compiled: Bool = false, asynchronous: Bool = false,
    maximumWorkspaceBytes: UInt64? = nil,
    checkCancellation: () throws -> Void = { try Task.checkCancellation() }) throws -> MLXArray {
    try checkCancellation()
    let keys = concatenated(planes.map(\.key), axis: 0)
    let values = concatenated(planes.map(\.value), axis: 0)
    if wholeGrid { return Self.attend(query, keys, values) }
    let skip = useBlockSkipping && RelaySkippedAttention.runtimeQualified && query.dim(2) == headDimension
      && [DType.bfloat16, .float32].contains(query.dtype) && query.dtype == keys.dtype && query.dtype == values.dtype
    let safe = skip ? RelaySkippedAttention.safeInputs(query: query, key: keys, value: values) : MLXArray(false)
    RelaySkippedAttention.trace(custom: skip, safe: safe)
    // Bound both in-flight groups explicitly; large/unqualified cases retain the
    // original synchronous path. Full outputs are accounted by the movie plan.
    let scalarBytes = query.dtype == .float32 ? 4 : 2
    var largestGroup: UInt64 = 0
    for first in stride(from: 0, to: tiles.count, by: 8) {
      let bytes = tiles[first..<min(first + 8, tiles.count)].reduce(UInt64(0)) { total, t in
        total + UInt64(t.queries.size * query.dim(1) * query.dim(2) * scalarBytes * 2
          + t.keys.size * keys.dim(1) * keys.dim(2) * scalarBytes * 2 + t.mask.size)
      }
      largestGroup = max(largestGroup, bytes)
    }
    let budget = maximumWorkspaceBytes ?? min(4 * 1024 * 1024 * 1024, UInt64(max(0, Memory.memoryLimit)))
    let pipeline = asynchronous && skip && largestGroup * 2 <= budget
    let graph = compiled && skip ? RelayCompiledAttentionTile() : nil
    if RelaySkippedAttention.traceEnabled {
      let message = "Relay attention compiled=\(graph != nil) async-depth=\(pipeline ? 2:1) workspace=\(largestGroup*2) budget=\(budget)\n"
      FileHandle.standardError.write(Data(message.utf8))
    }
    var signatures = Set<String>()
    var pending: [[MLXArray]] = []
    // Groups are independent: every queued group must finish on all exit paths.
    defer { for group in pending { eval(group) } }
    var results: [MLXArray] = []
    for (index, tile) in tiles.enumerated() {
      if skip, let types = tile.blockTypes, tile.queries.size > 8 {
        let signature = "\(tile.queries.size),\(tile.keys.size)"
        if let graph, signatures.contains(signature) || signatures.count < 32 {
          signatures.insert(signature)
          results.append(graph.evaluate(query: query, key: keys, value: values,
            queries: tile.queries, keys: tile.keys, mask: tile.mask, classes: types, safe: safe))
        } else {
          results.append(try RelaySkippedAttention.apply(query: query.take(tile.queries, axis: 0),
            key: keys.take(tile.keys, axis: 0), value: values.take(tile.keys, axis: 0),
            mask: tile.mask, classes: types, allowSkipping: safe))
        }
      } else {
        results.append(Self.attend(query.take(tile.queries, axis: 0), keys.take(tile.keys, axis: 0),
          values.take(tile.keys, axis: 0), mask: tile.mask))
      }
      if index % 8 == 7 || index == tiles.count - 1 {
        let batch = Array(results.suffix(index % 8 + 1))
        if pipeline {
          asyncEval(batch); pending.append(batch)
          if pending.count == 2 { eval(pending.removeFirst()) }
        } else { eval(batch) }
        try checkCancellation()
      }
    }
    for group in pending { eval(group) }; pending.removeAll()
    try checkCancellation()
    return concatenated(results, axis: 0).take(restoreOrder!, axis: 0)
  }

  static func attend(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray, mask: MLXArray? = nil)
    -> MLXArray
  {
    let output = MLXFast.scaledDotProductAttention(
      queries: q.transposed(1, 0, 2).expandedDimensions(axis: 0),
      keys: k.transposed(1, 0, 2).expandedDimensions(axis: 0),
      values: v.transposed(1, 0, 2).expandedDimensions(axis: 0),
      scale: 1 / sqrt(Float(q.dim(-1))), mask: mask)
    return output.squeezed(axis: 0).transposed(1, 0, 2)
  }
}
