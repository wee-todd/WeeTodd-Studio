import Foundation
import Metal

/// Sequential transformer blocks sharing one weighted GPU slot. This is a
/// transformer-stack component, not the complete LTX denoiser or sampling loop.
public final class AVStackRunner {
  public struct Progress: Sendable {
    public let completedBlocks: Int
    public let totalBlocks: Int
    public let residentBlocks: Int
    public let weightLoadSeconds: Double
    public let computeSeconds: Double
    public let runtimeBytes: UInt64
    public let metalAllocatedBytes: UInt64
    public let graphVariables: Int
    public let metrics: ExecutionMetrics
  }
  public struct TransferCounts: Sendable {
    public var activationUploads = 0
    public var activationDownloads = 0
    public init() {}
  }
  public let configuration: AVBlockConfiguration
  public let blockCount: Int
  public private(set) var lastTransferCounts = TransferCounts()
  private let sequenceAttention: Bool
  private let precision: LTXPrecisionPolicy
  private var slot: AVBlockRunner?
  private var active = false
  public private(set) var graphBuildCount = 0
  public var residentBlocks: Int { slot == nil ? 0 : 1 }
  public var metalAllocatedBytes: UInt64 { UInt64(MTLCreateSystemDefaultDevice()?.currentAllocatedSize ?? 0) }

  public init(configuration: AVBlockConfiguration, blockCount: Int, sequenceAttention: Bool = false, experimentalPrecision: LTXPrecisionPolicy = .float32) throws {
    guard (1...48).contains(blockCount) else { throw BlockError.invalid("Expected 1–48 ordered LTX blocks.") }
    try AVBlockRunner.validateAllocation(configuration: configuration)
    self.configuration = configuration; self.blockCount = blockCount
    self.sequenceAttention = sequenceAttention; self.precision = experimentalPrecision
  }

  public static func validatePreparationBudget(configuration: AVBlockConfiguration, maximumBytes: UInt64) throws {
    guard maximumBytes > 0 else { return }
    let shapes = try AVBlockRunner.expectedWeightShapes(configuration: configuration)
    _ = try WeightPreparationQueue(layout: shapes.sorted { $0.key < $1.key }.map { ($0.key, $0.value) },
      maximumBytes: maximumBytes)
  }

  /// The provider must preflight every source before calling this method. It is
  /// invoked on the inference thread by default. With an explicit CPU preparation
  /// budget it runs on one serial background queue and must perform CPU-only reads.
  /// Only the inference thread installs GPU weights after prior use completes.
  /// All failures release the slot, including failures in a progress observer.
  public func evaluate(_ inputs: [String: [Float]], retainWeights: Bool = false, maximumPreparationBytes: UInt64 = 0,
    weights: @escaping (Int, String, [Int]) throws -> [Float],
    progress: (Progress) throws -> Void = { _ in }) throws -> AVBlockRunner.Output {
    guard !active else { throw BlockError.invalid("The LTX stack is already executing.") }
    active = true
    var succeeded = false
    defer {
      if !succeeded || !retainWeights { slot = nil }
      active = false
    }
    lastTransferCounts = TransferCounts()
    let shapes = try AVBlockRunner.expectedInputShapes(configuration: configuration)
    guard Set(inputs.keys) == Set(shapes.keys), shapes.allSatisfy({ name, shape in
      inputs[name]!.count == shape.reduce(1, *) && inputs[name]!.allSatisfy(\.isFinite)
    }) else { throw BlockError.invalid("Stack inputs do not match the validated block layout.") }
    try Task.checkCancellation()
    try Self.validatePreparationBudget(configuration: configuration, maximumBytes: maximumPreparationBytes)
    if slot == nil {
      slot = try AVBlockRunner(configuration: configuration, sequenceAttention: sequenceAttention, precision: precision)
      graphBuildCount += 1
    }
    let result = try slot!.evaluateStack(inputs, blockCount: blockCount, maximumPreparationBytes: maximumPreparationBytes, weights: weights,
      transfers: { lastTransferCounts = $0 }, progress: progress)
    try Task.checkCancellation()
    succeeded = true
    return result
  }

  /// Releases owned graph/weight objects. Allocator/driver residency is measured
  /// separately; worker exit remains the hard release boundary for production.
  public func release() throws {
    guard !active else { throw BlockError.invalid("Cannot release a stack during its active evaluation.") }
    slot = nil
  }
}
