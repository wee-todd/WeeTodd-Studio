import Foundation
import MLX
import LTX25Engine

/// One active packed block; evaluated hidden states stay in MLX between blocks.
/// The owning worker must serialize executions. Cache policy is process-global
/// in MLX, so overlapping jobs belong in separate workers, not this object.
public final class MLXAVStack {
  public struct Progress {
    public let completedBlocks:Int
    public let totalBlocks:Int
    public let weightBytes:Int
    public let activeBytes:Int
    public let cacheBytes:Int
    public let loadSeconds:Double
    public let computeSeconds:Double
  }
  public var residentWeightBytes:Int { block.loadedBytes }
  private let block:MLXAVBlock
  private let count:Int
  private let cacheBytes:Int
  private var running=false
  public init(configuration:AVBlockConfiguration,blockCount:Int=48,cacheBytes:Int=128*1024*1024,maximumActivationBytes:Int=2*1024*1024*1024,compileGraph:Bool=true) throws {
    guard (1...48).contains(blockCount), (0...1024*1024*1024).contains(cacheBytes) else {
      throw LTXError.invalid("Admit 1–48 blocks and at most 1 GiB cached allocations.")
    }
    block=try MLXAVBlock(configuration:configuration,maximumActivationBytes:maximumActivationBytes,compileGraph:compileGraph); count=blockCount; self.cacheBytes=cacheBytes
  }
  func admitPerTokenVideo() throws { try block.admitPerTokenVideo() }
  func admitPerTokenAudio() throws { try block.admitPerTokenAudio() }
  public func evaluate(_ inputs:[String:MLXArray],
    weights:(Int,String,[Int]) throws -> MLXWeight,
    adapters:(Int) throws -> [String:[MLXLoRA]] = { _ in [:] },
    progress:(Progress) throws -> Void = { _ in }) throws -> [String:MLXArray] {
    guard !running else { throw LTXError.invalid("MLX stack already executing.") }
    try block.validateInputs(inputs)
    running=true
    let previousLimit=Memory.cacheLimit
    Memory.cacheLimit=cacheBytes
    if Memory.cacheMemory > cacheBytes { Memory.clearCache() }
    defer {
      block.release(); Memory.clearCache(); Memory.cacheLimit=previousLimit; running=false
    }
    // Dictionary copies retain handles, not payload copies. Only the current
    // two stream arrays are replaced; prompt/modulation/rotary arrays are shared.
    var current=inputs
    var loading=0.0, computing=0.0
    for index in 0..<count {
      try Task.checkCancellation()
      let start=Date()
      try block.load { try weights(index,$0,$1) }
      try block.setAdapters(adapters(index))
      loading += Date().timeIntervalSince(start)
      let computeStart=Date()
      let output=try block.evaluate(current)
      computing += Date().timeIntervalSince(computeStart)
      current["video"]=output["video"]; current["audio"]=output["audio"]
      // MLX's cache limit is a target and can overshoot by an allocation.
      // Enforce our boundary policy after synchronized work, before reporting.
      if Memory.cacheMemory > cacheBytes { Memory.clearCache() }
      try progress(Progress(completedBlocks:index+1,totalBlocks:count,weightBytes:block.loadedBytes,
        activeBytes:Memory.activeMemory,cacheBytes:Memory.cacheMemory,loadSeconds:loading,computeSeconds:computing))
      try Task.checkCancellation()
      // Do not keep the previous block alive while loading its replacement.
      block.release()
      if Memory.cacheMemory > cacheBytes { Memory.clearCache() }
    }
    return ["video":current["video"]!,"audio":current["audio"]!]
  }
}
