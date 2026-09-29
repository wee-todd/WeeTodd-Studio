import Foundation

/// Three-axis Qwen positions for H3's mixed text and image presentation.
/// Text rows advance one shared position; merged image patches advance time,
/// height and width separately before the following text resumes.
public enum H3QwenMRoPE {
  public static func positions(request: H3QwenRequest,
    grids: [H3QwenRequest.Grid]) throws -> [[Int32]] {
    guard request.visualRanges.count == grids.count,
      request.tokenIDs.count == request.tags.count,
      (1...1024).contains(request.tokenIDs.count) else {
      throw H3CheckpointError.invalid("H3 Qwen visual ranges and grids differ.")
    }
    let count = request.tokenIDs.count
    var axes = [[Int32]](repeating: [], count: 3)
    for index in 0..<3 { axes[index].reserveCapacity(count) }
    var cursor = 0
    var nextPosition = 0
    func appendText(until end: Int) throws {
      guard end >= cursor, end <= count else {
        throw H3CheckpointError.invalid("H3 Qwen visual ranges are out of order.")
      }
      for _ in cursor..<end {
        for axis in 0..<3 { axes[axis].append(Int32(nextPosition)) }
        nextPosition += 1
      }
      cursor = end
    }
    for (range, grid) in zip(request.visualRanges, grids) {
      let firstPad = range.lowerBound + 1
      let height = grid.height / 2, width = grid.width / 2
      let patchCount = grid.temporal * height * width
      guard grid.temporal > 0, height > 0, width > 0,
        patchCount == range.count - 2, range.upperBound <= count,
        request.tags[range.lowerBound..<range.upperBound].allSatisfy({ $0 == 0 }) else {
        throw H3CheckpointError.invalid("H3 Qwen image pads differ from their grid geometry.")
      }
      try appendText(until: firstPad)
      let base = nextPosition
      for time in 0..<grid.temporal {
        for row in 0..<height {
          for column in 0..<width {
            axes[0].append(Int32(base + time))
            axes[1].append(Int32(base + row))
            axes[2].append(Int32(base + column))
          }
        }
      }
      cursor += patchCount
      nextPosition = base + max(grid.temporal, height, width)
    }
    try appendText(until: count)
    guard axes.allSatisfy({ $0.count == count }) else {
      throw H3CheckpointError.invalid("H3 Qwen position rows were not fully assigned.")
    }
    return axes
  }
}
