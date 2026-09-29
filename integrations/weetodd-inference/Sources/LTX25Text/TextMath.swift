import Foundation
import Metal
import MetalPerformanceShaders
import TensorIO

struct TextMatrixInput {
  fileprivate let owner: UUID
  let buffer: MTLBuffer
  fileprivate let rows: Int
  fileprivate let inner: Int
}

/// A bounded, synchronous GPU matrix product. Shared buffers live only through the
/// command's completion; every failure/cancellation unwinds their ownership.
final class TextMatrixGPU {
  let device: MTLDevice
  let queue: MTLCommandQueue
  var attentionPipelines: [String: MTLComputePipelineState]?
  var lastAttentionReadbacks = 0
  var lastAttentionCommands = 0
  private let identity = UUID()
  private(set) var leftUploadCount = 0
  init() throws {
    guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
      throw TextEncodingError.invalid("Metal is required for native Gemma inference.")
    }
    self.device = device; self.queue = queue
  }
  func multiply(_ a: [Float], _ b: [Float], rows: Int, inner: Int, columns: Int,
    transposeB: Bool = true, scale: Float = 1) throws -> [Float] {
    guard (1...1048576).contains(inner), (1...1048576).contains(columns),
      b.count == inner*columns, b.count <= 256*1024*1024, scale.isFinite else {
      throw TextEncodingError.invalid("GPU text right-matrix shape or scale invalid.")
    }
    return try multiply(prepareLeft(a,rows: rows,inner: inner),b,columns: columns,transposeB: transposeB,scale: scale)
  }
  func prepareLeft(_ values: [Float], rows: Int, inner: Int) throws -> TextMatrixInput {
    try Task.checkCancellation()
    guard (1...1048576).contains(rows), (1...1048576).contains(inner),
      values.count == rows*inner, values.count <= 256*1024*1024, FloatValidation.allFinite(values),
      values.count <= device.maxBufferLength/4,
      let buffer = device.makeBuffer(bytes: values,length: values.count*4,options: .storageModeShared) else {
      throw TextEncodingError.invalid("GPU text left-matrix shape, values or allocation invalid.")
    }
    leftUploadCount += 1
    return TextMatrixInput(owner: identity,buffer: buffer,rows: rows,inner: inner)
  }
  func multiply(_ left: TextMatrixInput, _ b: [Float], columns: Int,
    transposeB: Bool = true, scale: Float = 1,
    checkCancelled: () throws -> Void = { try Task.checkCancellation() }) throws -> [Float] {
    return try autoreleasepool {
    try checkCancelled()
    let rows = left.rows, inner = left.inner
    guard left.owner == identity, (1...1048576).contains(columns), b.count == inner*columns,
      b.count <= 256*1024*1024, rows*columns <= 256*1024*1024,
      scale.isFinite, FloatValidation.allFinite(b) else {
      throw TextEncodingError.invalid("GPU text matrix shape or allocation budget invalid.")
    }
    let outputCount = rows * columns
    let ab = left.buffer
    guard let bb = device.makeBuffer(bytes: b, length: b.count * 4, options: .storageModeShared),
      let cb = device.makeBuffer(length: outputCount * 4, options: .storageModeShared),
      let command = queue.makeCommandBuffer() else { throw TextEncodingError.invalid("Cannot allocate text matrix buffers.") }
    let ad = MPSMatrixDescriptor(rows: rows, columns: inner, rowBytes: inner * 4, dataType: .float32)
    let bd = MPSMatrixDescriptor(rows: transposeB ? columns : inner, columns: transposeB ? inner : columns,
      rowBytes: (transposeB ? inner : columns) * 4, dataType: .float32)
    let cd = MPSMatrixDescriptor(rows: rows, columns: columns, rowBytes: columns * 4, dataType: .float32)
    let kernel = MPSMatrixMultiplication(device: device, transposeLeft: false, transposeRight: transposeB,
      resultRows: rows, resultColumns: columns, interiorColumns: inner, alpha: Double(scale), beta: 0)
    kernel.encode(commandBuffer: command, leftMatrix: MPSMatrix(buffer: ab, descriptor: ad),
      rightMatrix: MPSMatrix(buffer: bb, descriptor: bd), resultMatrix: MPSMatrix(buffer: cb, descriptor: cd))
    command.commit(); command.waitUntilCompleted()
    guard command.status == .completed else {
      throw TextEncodingError.invalid("Text matrix GPU command failed: \(command.error?.localizedDescription ?? "unknown")")
    }
    try checkCancelled()
    let result = Array(UnsafeBufferPointer(start: cb.contents().assumingMemoryBound(to: Float.self), count: outputCount))
    guard result.allSatisfy(\.isFinite) else { throw TextEncodingError.invalid("Nonfinite GPU text matrix result.") }
    return result
    }
  }
}

enum TextMath {
  static func rms(_ x: [Float], width: Int, weight: [Float]? = nil, epsilon: Float = 1e-6) -> [Float] {
    precondition(width > 0 && x.count % width == 0 && (weight == nil || weight!.count == width))
    var result = x
    for row in 0..<(x.count / width) {
      let start = row * width
      var sum: Double = 0
      for d in 0..<width { sum += Double(x[start+d]) * Double(x[start+d]) }
      let scale = 1 / (Float(sum / Double(width)) + epsilon).squareRoot()
      for d in 0..<width { result[start+d] = x[start+d] * scale * (weight?[d] ?? 1) }
    }
    return result
  }
  static func interleavedStates(_ states: [[Float]], tokens: Int, width: Int) throws -> [Float] {
    guard tokens > 0, tokens <= 1024, width > 0, width <= 8192, states.count > 0, states.count <= 65,
      states.allSatisfy({ $0.count == tokens * width && $0.allSatisfy(\.isFinite) }) else {
      throw TextEncodingError.invalid("Invalid Gemma hidden states.")
    }
    var result = [Float](repeating: 0, count: tokens * width * states.count)
    for (layer, state) in states.enumerated() {
      let normed = rms(state, width: width)
      for i in normed.indices { result[i * states.count + layer] = normed[i] }
    }
    return result
  }
  static func gemmaRotary(_ x: [Float], tokens: Int, heads: Int, width: Int,
    theta: Double, fraction: Double) -> [Float] {
    var result = x
    let rotatingPairs = Int(Double(width) * fraction) / 2
    for t in 0..<tokens { for h in 0..<heads { for d in 0..<rotatingPairs {
      let angle = Double(t) / pow(theta, Double(2*d)/Double(width))
      let c = Float(cos(angle)), s = Float(sin(angle)), start = (t * heads + h) * width
      let a = x[start+d], b = x[start+d+width/2]
      result[start+d] = a*c-b*s; result[start+d+width/2] = a*s+b*c
    } } }
    return result
  }
  static func connectorRotary(_ x: [Float], tokens: Int, heads: Int, width: Int) -> [Float] {
    let frequencies = heads * width / 2
    var result = x
    for t in 0..<tokens { for h in 0..<heads { for d in 0..<(width/2) {
      // LTX: (position/maxPos)*2 - 1, multiplied by the float64-built,
      // float32-cast log grid. Frequency bands span ALL heads.
      let index = h * width / 2 + d
      let frequency = Float(pow(10000, Double(index)/Double(frequencies-1)) * (.pi/2))
      let angle = (Float(t)/4096 * 2 - 1) * frequency
      let c = cos(angle), s = sin(angle), start = (t * heads + h) * width
      let a = x[start+d], b = x[start+d+width/2]
      result[start+d] = a*c-b*s; result[start+d+width/2] = a*s+b*c
    } } }
    return result
  }
  static func gelu(_ x: Float) -> Float { 0.5*x*(1+tanh(sqrt(2/Float.pi)*(x+0.044715*x*x*x))) }
  static func attention(q: [Float], k: [Float], v: [Float], tokens: Int, heads: Int,
    kvHeads: Int, width: Int, scale: Float, window: Int? = nil, causal: Bool = true,
    gpu provided: TextMatrixGPU? = nil) throws -> [Float] {
    let gpu = try provided ?? TextMatrixGPU()
    return try gpu.attention(q: q, k: k, v: v, tokens: tokens, heads: heads, kvHeads: kvHeads,
      width: width, scale: scale, window: window, causal: causal)
  }

  /// Retained CPU-softmax path for numerical comparisons; never selected implicitly.
  static func attentionReference(q: [Float], k: [Float], v: [Float], tokens: Int, heads: Int,
    kvHeads: Int, width: Int, scale: Float, window: Int? = nil, causal: Bool = true,
    gpu provided: TextMatrixGPU? = nil) throws -> [Float] {
    guard tokens > 0, tokens <= 1024, heads > 0, kvHeads > 0, heads % kvHeads == 0,
      q.count == tokens*heads*width, k.count == tokens*kvHeads*width, v.count == k.count,
      window == nil || window! > 0 else { throw TextEncodingError.invalid("Invalid grouped text attention shape.") }
    let gpu = try provided ?? TextMatrixGPU()
    var result = [Float](repeating: 0, count: q.count)
    for h in 0..<heads {
      try Task.checkCancellation()
      let kh = h/(heads/kvHeads)
      var query = [Float](repeating: 0, count: tokens*width), key = query, value = query
      for t in 0..<tokens { for d in 0..<width {
        query[t*width+d] = q[(t*heads+h)*width+d]
        key[t*width+d] = k[(t*kvHeads+kh)*width+d]
        value[t*width+d] = v[(t*kvHeads+kh)*width+d]
      } }
      var scores = try gpu.multiply(query, key, rows: tokens, inner: width, columns: tokens, scale: scale)
      for row in 0..<tokens {
        let lower = causal ? max(0, row-(window ?? tokens)+1) : 0
        let upper = causal ? row+1 : tokens
        var maximum = -Float.infinity
        for col in lower..<upper { maximum = max(maximum, scores[row*tokens+col]) }
        var sum: Double = 0
        for col in 0..<tokens {
          let score: Float = (lower..<upper).contains(col) ? exp(scores[row*tokens+col]-maximum) : 0
          scores[row*tokens+col] = score; sum += Double(score)
        }
        for col in 0..<tokens { scores[row*tokens+col] /= Float(sum) }
      }
      let output = try gpu.multiply(scores, value, rows: tokens, inner: tokens, columns: width, transposeB: false)
      for t in 0..<tokens { for d in 0..<width { result[(t*heads+h)*width+d] = output[t*width+d] } }
    }
    return result
  }
}
