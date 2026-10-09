import Foundation
import TensorIO

/// Own a bounded serial preparation window within one transformer stage.
/// Completed blocks release their arrays immediately after the caller drains
/// their output; the next window cannot load until the old window is closed.
final class H3PreparedBlockWindow {
  struct Plan {
    let blockCount: Int
    let windowSize: Int
    private(set) var nextIndex = 0
    private(set) var outstandingIndex: Int?
    private(set) var isClosed = false

    init(blockCount: Int, windowSize: Int, startIndex: Int = 0) throws {
      guard (1...50).contains(blockCount), [1, 2].contains(windowSize), (0..<blockCount).contains(startIndex),
        startIndex.isMultiple(of:windowSize) else {
        throw H3CheckpointError.invalid("Invalid prepared H3 block window plan.")
      }
      self.blockCount = blockCount
      self.windowSize = windowSize
      nextIndex = startIndex
    }

    mutating func begin(_ index: Int) throws -> Range<Int> {
      guard !isClosed, outstandingIndex == nil, index == nextIndex,
        (0..<blockCount).contains(index) else {
        throw H3CheckpointError.invalid("Prepared H3 blocks must be requested once in order.")
      }
      outstandingIndex = index
      let start = (index / windowSize) * windowSize
      return start..<min(start + windowSize, blockCount)
    }

    mutating func finish(_ index: Int) throws {
      guard !isClosed, outstandingIndex == index, index == nextIndex else {
        throw H3CheckpointError.invalid("Prepared H3 block completion differs from its active owner.")
      }
      outstandingIndex = nil
      nextIndex += 1
      if nextIndex == blockCount { isClosed = true }
    }

    mutating func close() { outstandingIndex = nil; isClosed = true }
  }

  let checkpointURL: URL
  private let projectionMode: H3ProjectionMode
  private let rowWindow: Int
  private let useMPP: Bool
  private let verificationScope: String
  private var plan: Plan
  private var owners: [Int: H3PreparedBlock] = [:]
  private var loadedRange: Range<Int>?

  var isClosed: Bool { plan.isClosed }
  var residentIndices: [Int] { owners.keys.sorted() }
  var storageBytes: Int { owners.values.reduce(0) { $0 + $1.storageBytes } }

  init(checkpointURL: URL, blockCount: Int, windowSize: Int,
    projectionMode: H3ProjectionMode = .weightDecoded,
    rowWindow: Int = 1024, useMPP: Bool = false, verificationScope: String? = nil, startIndex: Int = 0) throws {
    let admittedPlan = try Plan(blockCount: blockCount, windowSize: windowSize, startIndex:startIndex)
    guard projectionMode == .weightDecoded else {
      throw H3CheckpointError.invalid("Prepared H3 blocks require weight-decoded projections.")
    }
    guard [1024, 2048, 4096, 8192, 16384].contains(rowWindow) else {
      throw H3CheckpointError.invalid("Prepared H3 decode row window is unsupported.")
    }
    try Task.checkCancellation()
    self.checkpointURL = checkpointURL
    self.projectionMode = projectionMode
    self.rowWindow = rowWindow
    self.useMPP = useMPP
    self.verificationScope = verificationScope ?? checkpointURL.standardizedFileURL.path
    plan = admittedPlan
  }

  func weights(for index: Int) throws -> H3PreparedBlock {
    do {
      try Task.checkCancellation()
      let range = try plan.begin(index)
      if loadedRange != range {
        closeOwners()
        loadedRange = nil
        if plan.windowSize == 2 { try admitPackedFastWindow(range) }
        for block in range {
          try Task.checkCancellation()
          owners[block] = try H3PreparedBlock(checkpointURL: checkpointURL, index: block,
            projectionMode: projectionMode, rowWindow: rowWindow, useMPP: useMPP, verificationScope: verificationScope)
        }
        loadedRange = range
      }
      guard let owner = owners[index], !owner.isClosed else {
        throw H3CheckpointError.invalid("Prepared H3 window is missing its active owner.")
      }
      try owner.checkUnchanged()
      return owner
    } catch {
      close()
      throw error
    }
  }

  /// The caller must have evaluated the block output before completing it.
  /// H3PreparedBlock.close also drains any remaining GPU use before releasing
  /// base weights and the block-local adapter scope, including deferred pairs.
  func finish(index: Int, deferRetirement: Bool = false) throws {
    do {
      try Task.checkCancellation()
      guard let owner = owners[index], !owner.isClosed else {
        throw H3CheckpointError.invalid("Prepared H3 window is missing its completed owner.")
      }
      try owner.checkUnchanged()
      try plan.finish(index)
      if deferRetirement {
        // MLX retains submitted Metal inputs until completion. Keep the
        // bounded window until its final block, then drain before reporting
        // completed progress or preparing another window.
        if index + 1 == loadedRange?.upperBound { closeOwners(); loadedRange = nil }
      } else {
        owner.close()
        owners.removeValue(forKey: index)
      }
      if plan.isClosed { close() }
    } catch {
      close()
      throw error
    }
  }

  func close() {
    closeOwners()
    loadedRange = nil
    plan.close()
  }

  deinit { close() }

  private func closeOwners() {
    for index in owners.keys.sorted() { owners[index]?.close() }
    owners.removeAll()
  }

  /// Two-block preparation is restricted to packed FastH3 weights. Check the
  /// whole candidate window's headers before any of its GPU arrays load.
  private func admitPackedFastWindow(_ range: Range<Int>) throws {
    let layout = try H3CheckpointLayout(url: checkpointURL)
    guard layout.fastVariant != nil else {
      throw H3CheckpointError.invalid("Two-block H3 preparation requires packed FastH3 weights.")
    }
    for index in range {
      try Task.checkCancellation()
      let url = try H3CheckpointSource.fileURL(checkpointURL, block: index)
      let file = try SafeTensorFile(url: url)
      let prefix = layout.prefix + "blocks.\(index)."
      guard ["attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2"].allSatisfy({
        file.tensors[prefix + $0 + ".weight"]?.dtype == "U32"
      }) else {
        throw H3CheckpointError.invalid("Two-block H3 preparation requires packed FastH3 weights.")
      }
      try file.checkUnchanged(at: url)
    }
    try H3CheckpointSource.checkUnchanged(checkpointURL)
    try Task.checkCancellation()
  }
}
