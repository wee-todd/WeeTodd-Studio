import Foundation
import MLX

enum MOSSMath {
  static func regionalRope(height: Int, width: Int, textLengths: [Int], headDimension: Int) -> MLXArray {
    let total = textLengths.reduce(0,+), rows = height*width
    let all = rope(height: height, width: width, textLength: total, headDimension: headDimension)
    return concatenated([all[0...,0..<rows,0...]] + textLengths.map { all[0...,(rows+total-$0)..<(rows+total),0...] }, axis: 1)
  }
  static func situ(_ hidden: MLXArray, _ gate: MLXArray) -> MLXArray {
    (4 * tanh(gate / 4) * sigmoid(gate)) * (25 * tanh(hidden / 25))
  }
  static func timeEmbedding(_ timestep: Double, dimension: Int = 256) -> [Float] {
    let phases = (0..<(dimension / 2)).map { timestep * Foundation.pow(10000, -Double($0) / Double(dimension / 2)) }
    return phases.map { Float(cos($0)) } + phases.map { Float(sin($0)) }
  }
  static func rope(height: Int, width: Int, textLength: Int, headDimension: Int = 128) -> MLXArray {
    let bands = headDimension / 8
    let frequencies = (0..<bands).map { 1 / Foundation.pow(Float(10000), Float($0) / Float(bands)) }
    var values = [Float](); values.reserveCapacity((height * width + textLength) * headDimension)
    for token in 0..<(height * width + textLength) {
      let coordinates: [Float]
      if token < height * width {
        coordinates = [0, Float(textLength), Float(token / width) - Float(height - 1) / 2, Float(token % width) - Float(width - 1) / 2]
      } else { coordinates = [0, Float(token - height * width), 0, 0] }
      let phase = coordinates.flatMap { coordinate in frequencies.map { coordinate * $0 } }
      values += phase.map { sin($0) } + phase.map { cos($0) }
    }
    return MLXArray(values).reshaped(1, height * width + textLength, headDimension)
  }
  static func rotate(_ x: MLXArray, rope: MLXArray) -> MLXArray {
    let half = x.dim(-1) / 2
    let s = rope[.ellipsis, 0..<half].expandedDimensions(axis: 2)
    let c = rope[.ellipsis, half...].expandedDimensions(axis: 2)
    let sin = concatenated([s, s], axis: -1), cos = concatenated([c, c], axis: -1)
    let rotated = concatenated([-x[.ellipsis, half...], x[.ellipsis, 0..<half]], axis: -1)
    return x * cos + rotated * sin
  }
  static func rms(_ x: MLXArray, weight: MLXArray, epsilon: Float = 1e-6) -> MLXArray {
    let f = x.asType(.float32)
    return (f * rsqrt(mean(f * f, axis: -1, keepDims: true) + epsilon)).asType(x.dtype) * weight
  }
}
