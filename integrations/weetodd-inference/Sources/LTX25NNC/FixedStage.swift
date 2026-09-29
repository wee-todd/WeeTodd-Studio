import Foundation
import NNC
import TensorIO

/// One staged top-level component. Never instantiate all eight adaptive heads together.
struct FixedStage {
  struct Binding { let layer: Model; let name: String; let shape: [Int]; let bias: Bool }
  let model: Model
  let inputs: [[Int]]
  let outputShapes: [[Int]]
  let bindings: [Binding]
  var weightShapes: [String: [Int]] { Dictionary(uniqueKeysWithValues: bindings.map { ($0.name, $0.shape) }) }

  static func adaptive(_ name: String, dimension: Int, parameters: Int) -> FixedStage {
    var bindings: [Binding] = []
    let input = Input()
    func dense(_ suffix: String, _ value: ModelIOConvertible, _ width: Int, _ count: Int) -> Model.IO {
      let key = name + suffix, layer = Dense(count: count, name: key)
      bindings += [Binding(layer: layer, name: key + ".weight", shape: [count, width], bias: false),
        Binding(layer: layer, name: key + ".bias", shape: [count], bias: true)]
      return layer(value)
    }
    let hidden = dense(".emb.timestep_embedder.linear1", input, 256, dimension).swish()
    let embedded = dense(".emb.timestep_embedder.linear2", hidden, dimension, dimension)
    let params = dense(".linear", embedded.swish(), dimension, dimension * parameters)
    return FixedStage(model: Model([input], [params, embedded]), inputs: [[1, 256]],
      outputShapes: [[1, dimension * parameters], [1, dimension]], bindings: bindings)
  }

  static func projection(_ name: String, rows: Int, width: Int, output: Int,
    table: String? = nil) -> FixedStage {
    let input = Input(), time = Input()
    var value: Model.IO = input.io, bindings: [Binding] = []
    if let table {
      let shiftScale = Parameter<Float>(.GPU(0), format: .NHWC, shape: [2, width], name: table)
      bindings.append(Binding(layer: shiftScale, name: table, shape: [2, width], bias: false))
      let shift = shiftScale.io.reshaped([1, width], offset: [0, 0], strides: [width, 1]) + time
      let scale = shiftScale.io.reshaped([1, width], offset: [1, 0], strides: [width, 1]) + time
      value = LayerNorm(epsilon: 1e-6, axis: [1], elementwiseAffine: false)(value) .* (1 + scale) + shift
    }
    let layer = Dense(count: output, name: name)
    bindings += [Binding(layer: layer, name: name + ".weight", shape: [output, width], bias: false),
      Binding(layer: layer, name: name + ".bias", shape: [output], bias: true)]
    return FixedStage(model: Model(table == nil ? [input] : [input, time], [layer(value)]),
      inputs: table == nil ? [[rows, width]] : [[rows, width], [1, width]],
      outputShapes: [[rows, output]], bindings: bindings)
  }

  func evaluate(_ values: [[Float]], weights: (String, [Int]) throws -> [Float]) throws -> [[Float]] {
    try evaluateMany([values], weights: weights)[0]
  }

  /// Compile/load once, retaining row-one arithmetic for every timestep. The
  /// caller releases this entire head before advancing to another weighted head.
  func evaluateMany(_ cases: [[[Float]]], weights: (String, [Int]) throws -> [Float]) throws -> [[[Float]]] {
    guard !cases.isEmpty, cases.count <= 256, cases.allSatisfy({ values in
      values.count == inputs.count && zip(values, inputs).allSatisfy {
        $0.0.count == $0.1.reduce(1, *) && FloatValidation.allFinite($0.0)
      }
    }), bindings.reduce(0, { $0 + $1.shape.reduce(4, *) }) <= 1024 * 1024 * 1024 else {
      throw BlockError.invalid("Invalid fixed-stage inputs or weight budget.")
    }
    try Task.checkCancellation()
    let graph = DynamicGraph(), stream = StreamContext(.GPU(0))
    defer { stream.joined() }
    model.maxConcurrency = .limit(1)
    return try graph.withNoGrad {
      func upload(_ values: [[Float]]) -> [DynamicGraph.AnyTensor] {
        zip(values, inputs).map { value, shape in
          graph.variable(Tensor<Float>(value).reshaped(format: .NHWC, shape: TensorShape(shape)).toGPU(0))
        }
      }
      model.compile(inputs: upload(cases[0]))
      for binding in bindings {
        try Task.checkCancellation()
        let array = try weights(binding.name, binding.shape)
        guard array.count == binding.shape.reduce(1, *), FloatValidation.allFinite(array) else {
          throw BlockError.invalid("Invalid fixed-stage weight: \(binding.name)")
        }
        let tensor = Tensor<Float>(array).reshaped(format: .NHWC, shape: TensorShape(binding.shape))
        if binding.bias { binding.layer.bias.copy(from: tensor) } else { binding.layer.weight.copy(from: tensor) }
      }
      return try cases.map { values in
        try autoreleasepool {
          try Task.checkCancellation()
          let tensors = upload(values)
          let outputs = model(inputs: tensors[0], Array(tensors.dropFirst()), streamContext: stream)
          stream.joined()
          let result = outputs.map { DynamicGraph.Tensor<Float>($0).toCPU().rawValue.withUnsafeBytes {
            Array($0.bindMemory(to: Float.self))
          } }
          guard result.allSatisfy({ FloatValidation.allFinite($0) }) else {
            throw BlockError.invalid("Nonfinite fixed-stage output.")
          }
          try Task.checkCancellation()
          return result
        }
      }
    }
  }
}
