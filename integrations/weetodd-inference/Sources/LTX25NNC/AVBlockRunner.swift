import Foundation
import NNC
import TensorIO
import Metal

/// A single explicitly owned block, for numerical qualification. Float32 execution
/// is the baseline; no claim of BF16 production or Draw Things performance parity.
public final class AVBlockRunner {
  public struct Output {
    public let video: [Float]
    public let audio: [Float]
    public let intermediates: [String: [Float]]
  }
  private let graph = DynamicGraph()
  private let stream = StreamContext(.GPU(0))
  private let block: AVBlockGraph
  public let precision: LTXPrecisionPolicy
  private var loaded = false
  private var metrics = ExecutionMetrics()
  public private(set) var graphBuildSeconds = 0.0
  private var reportedGraphBuild = false
  public var inputShapes: [String: [Int]] { block.inputShapes }
  public var weightShapes: [String: [Int]] { Dictionary(uniqueKeysWithValues: block.bindings.map { ($0.name, $0.shape) }) }
  public var scratchBytes: UInt64 { block.model.runtimeMemorySize }

  /// Inspect the symbolic layout before compiling or allocating weighted GPU state.
  public static func expectedWeightShapes(configuration: AVBlockConfiguration) throws -> [String: [Int]] {
    try configuration.validate()
    return Dictionary(uniqueKeysWithValues: AVBlockGraph(configuration: configuration, diagnostics: false)
      .bindings.map { ($0.name, $0.shape) })
  }

  public static func expectedInputShapes(configuration: AVBlockConfiguration) throws -> [String: [Int]] {
    try configuration.validate()
    return AVBlockGraph(configuration: configuration, diagnostics: false).inputShapes
  }

  /// Qualification safety limits for known input/weight storage, not an OS memory
  /// cap or an estimate of every NNC intermediate. Production uses host admission.
  public static func validateAllocation(configuration: AVBlockConfiguration,
    maximumDecodedWeightBytes: UInt64 = 2 * 1024 * 1024 * 1024,
    maximumInputBytes: UInt64 = 128 * 1024 * 1024) throws {
    try configuration.validate()
    let block = AVBlockGraph(configuration: configuration, diagnostics: false)
    try validateAllocation(block: block, maximumDecodedWeightBytes: maximumDecodedWeightBytes,
      maximumInputBytes: maximumInputBytes)
  }

  private static func validateAllocation(block: AVBlockGraph, maximumDecodedWeightBytes: UInt64,
    maximumInputBytes: UInt64) throws {
    // Dimensions have already been bounded; these products fit UInt64.
    let weightBytes = block.bindings.reduce(UInt64(0)) { $0 + $1.shape.reduce(UInt64(4)) { $0 * UInt64($1) } }
    let inputBytes = block.inputShapes.values.reduce(UInt64(0)) { $0 + $1.reduce(UInt64(4)) { $0 * UInt64($1) } }
    guard weightBytes <= maximumDecodedWeightBytes, inputBytes <= maximumInputBytes else {
      throw BlockError.invalid("Block exceeds the qualification weight/input allocation budget.")
    }
  }

  public init(configuration: AVBlockConfiguration, diagnostics: Bool = false, sequenceAttention: Bool = false,
    precision: LTXPrecisionPolicy = .float32) throws {
    try configuration.validate()
    self.precision = precision
    block = AVBlockGraph(configuration: configuration, diagnostics: diagnostics, sequenceAttention: sequenceAttention, precision: precision)
    try Self.validateAllocation(block: block, maximumDecodedWeightBytes: 2 * 1024 * 1024 * 1024,
      maximumInputBytes: 128 * 1024 * 1024)
    let began = Date()
    graph.withNoGrad {
      let inputs = block.inputNames.map { name -> DynamicGraph.AnyTensor in
        let shape = block.inputShapes[name]!
        return graph.variable(Tensor<Float>([Float](repeating: 0, count: shape.reduce(1, *)))
          .reshaped(format: .NHWC, shape: TensorShape(shape)).toGPU(0))
      }
      block.model.compile(inputs: inputs)
    }
    graphBuildSeconds = Date().timeIntervalSince(began)
  }

  /// Load one matrix at a time. The provider validates source shape before returning
  /// its bounded Float32 reconstruction; neither this API nor the reader caches a block.
  public func load(_ provider: (String, [Int]) throws -> [Float]) throws {
    loaded = false
    try graph.withNoGrad {
      for binding in block.bindings {
        try Task.checkCancellation()
        let preparation = Date()
        let values = try provider(binding.name, binding.shape)
        metrics.preparationSeconds += Date().timeIntervalSince(preparation)
        let validation = Date()
        guard values.count == binding.shape.reduce(1, *), FloatValidation.allFinite(values) else {
          throw BlockError.invalid("LTX block weight has invalid values or shape: \(binding.name)")
        }
        metrics.validationSeconds += Date().timeIntervalSince(validation)
        metrics.decodedWeightBytes += UInt64(values.count) * 4
        let installation = Date()
        let source = Tensor<Float>(values).reshaped(format: .NHWC, shape: TensorShape(binding.storageShape))
        let tensor: AnyTensor
        if binding.projection && precision == .float16Projections {
          tensor = Tensor<Float16>(try Float16Conversion.nearest(values))
            .reshaped(format: .NHWC,shape: TensorShape(binding.storageShape))
        }
        else if binding.projection && precision == .bfloat16Projections { tensor = Tensor<BFloat16>(from: source) }
        else { tensor = source }
        if binding.bias { binding.layer.bias.copy(from: tensor) }
        else { binding.layer.weight.copy(from: tensor) }
        metrics.installationSeconds += Date().timeIntervalSince(installation)
      }
    }
    let waiting = Date()
    stream.joined()
    metrics.installationSeconds += Date().timeIntervalSince(waiting)
    try Task.checkCancellation()
    loaded = true
  }

  public func evaluate(_ inputs: [String: [Float]]) throws -> Output {
    guard loaded, Set(inputs.keys) == Set(block.inputNames) else {
      throw BlockError.invalid("Load the block and supply exactly its supported inputs before evaluation.")
    }
    for (name, shape) in block.inputShapes {
      guard inputs[name]!.count == shape.reduce(1, *), inputs[name]!.allSatisfy(\.isFinite) else {
        throw BlockError.invalid("LTX block input has invalid values or shape: \(name)")
      }
    }
    try Task.checkCancellation()
    let values: [String: [Float]] = graph.withNoGrad {
      let tensors = block.inputNames.map { name -> DynamicGraph.AnyTensor in
        graph.variable(Tensor<Float>(inputs[name]!).reshaped(format: .NHWC,
          shape: TensorShape(block.inputShapes[name]!)).toGPU(0))
      }
      let results = block.model(inputs: tensors[0], Array(tensors.dropFirst()), streamContext: stream)
      stream.joined()
      return Dictionary(uniqueKeysWithValues: zip(block.outputNames, results).map { name, result in
        let cpu = DynamicGraph.Tensor<Float>(result).toCPU().rawValue
        let array = cpu.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        return (name, array)
      })
    }
    try Task.checkCancellation()
    guard values.values.allSatisfy({ $0.allSatisfy(\.isFinite) }) else {
      throw BlockError.invalid("LTX block produced nonfinite output; execution cannot continue.")
    }
    return Output(video: values["video"]!, audio: values["audio"]!, intermediates: values)
  }

  /// Internal GPU-resident path. No full activation downloads occur between blocks.
  /// Small GPU reductions reject nonfinite values before proceeding to another block.
  func evaluateStack(_ inputs: [String: [Float]], blockCount: Int, maximumPreparationBytes: UInt64 = 0,
    weights: @escaping (Int, String, [Int]) throws -> [Float],
    transfers: (AVStackRunner.TransferCounts) -> Void,
    progress: (AVStackRunner.Progress) throws -> Void) throws -> Output {
    guard block.outputNames == ["video", "audio"] else {
      throw BlockError.invalid("Stack execution cannot retain intermediate diagnostics.")
    }
    let videoIndex = block.inputNames.firstIndex(of: "video")!
    let audioIndex = block.inputNames.firstIndex(of: "audio")!
    metrics = ExecutionMetrics()
    metrics.graphBuildSeconds = reportedGraphBuild ? 0 : graphBuildSeconds
    reportedGraphBuild = true
    var counts = AVStackRunner.TransferCounts()
    defer { transfers(counts) }
    let provider = weights
    let prefetch = maximumPreparationBytes == 0 ? nil : try WeightPreparationQueue(
      layout: block.bindings.map { ($0.name, $0.shape) }, maximumBytes: maximumPreparationBytes)
    metrics.preparedWeightBytes = prefetch?.admittedBytes ?? 0
    defer { prefetch?.drain() }
    if let prefetch { try prefetch.submit(index: 0, provider: provider) }
    return try graph.withNoGrad {
      var tensors = block.inputNames.map { name -> DynamicGraph.AnyTensor in
        counts.activationUploads += 1
        return graph.variable(Tensor<Float>(inputs[name]!).reshaped(format: .NHWC,
          shape: TensorShape(block.inputShapes[name]!)).toGPU(0))
      }
      var loadSeconds = 0.0, computeSeconds = 0.0
      for index in 0..<blockCount {
        try Task.checkCancellation()
        let loading = Date()
        try autoreleasepool {
          if let prefetch {
            let waiting = Date()
            let prepared = try prefetch.take()
            metrics.preparationWaitSeconds += Date().timeIntervalSince(waiting)
            metrics.backgroundPreparationSeconds += prepared.seconds
            guard prepared.index == index else { throw BlockError.invalid("Prepared block order changed.") }
            try load { name, shape in
              guard let values = prepared.tensors[name], values.count == shape.reduce(1, *) else {
                throw BlockError.invalid("Prepared weight layout changed.")
              }
              return values
            }
          } else { try load { name, shape in try weights(index, name, shape) } }
        }
        if let prefetch, index + 1 < blockCount { try prefetch.submit(index: index + 1, provider: provider) }
        loadSeconds += Date().timeIntervalSince(loading)
        let computing = Date()
        try autoreleasepool {
          let results = block.model(inputs: tensors[0], Array(tensors.dropFirst()), streamContext: stream)
          stream.joined()
          metrics.computeSeconds += Date().timeIntervalSince(computing)
          try Task.checkCancellation()
          let health = Date()
          var bounds: [DynamicGraph.Tensor<Float>] = []
          for result in results {
            // NNC's isNaN performs its own synchronized reduction. Retain it:
            // min/max alone can hide NaNs depending on the reduction kernel.
            guard !result.isNaN else { throw BlockError.invalid("LTX block \(index) produced NaN.") }
            let typed = DynamicGraph.Tensor<Float>(result)
            bounds.append(typed.reduced(.min, axis: [0, 1], streamContext: stream))
            bounds.append(typed.reduced(.max, axis: [0, 1], streamContext: stream))
          }
          let status = Functional.concat(axis: 1, bounds[0], bounds[1], bounds[2], bounds[3], streamContext: stream)
          stream.joined()
          let finite = status.toCPU().rawValue.withUnsafeBytes {
            $0.bindMemory(to: Float.self).allSatisfy(\.isFinite)
          }
          guard finite else { throw BlockError.invalid("LTX block \(index) produced infinite values.") }
          metrics.healthCheckSeconds += Date().timeIntervalSince(health)
          tensors[videoIndex] = results[0]; tensors[audioIndex] = results[1]
        }
        computeSeconds += Date().timeIntervalSince(computing)
        try progress(AVStackRunner.Progress(completedBlocks: index + 1, totalBlocks: blockCount,
          residentBlocks: 1, weightLoadSeconds: loadSeconds, computeSeconds: computeSeconds,
          runtimeBytes: scratchBytes, metalAllocatedBytes: UInt64(MTLCreateSystemDefaultDevice()?.currentAllocatedSize ?? 0),
          graphVariables: graph.statistics.variables, metrics: metrics))
        try Task.checkCancellation()
      }
      func download(_ index: Int) -> [Float] {
        counts.activationDownloads += 1
        return DynamicGraph.Tensor<Float>(tensors[index]).toCPU().rawValue.withUnsafeBytes {
          Array($0.bindMemory(to: Float.self))
        }
      }
      let video = download(videoIndex), audio = download(audioIndex)
      guard video.allSatisfy(\.isFinite), audio.allSatisfy(\.isFinite) else {
        throw BlockError.invalid("LTX stack produced nonfinite output.")
      }
      try Task.checkCancellation()
      return Output(video: video, audio: audio, intermediates: ["video": video, "audio": audio])
    }
  }
}
