import Foundation
import Metal
import MetalPerformanceShaders

/// Metal Performance Shaders SGEMM for the bounded convolution tiles. All buffers
/// have a lexical lifetime; completed commands retain no tensors or weight cache.
final class AudioMatrixEngine {
  private let device: MTLDevice
  private let queue: MTLCommandQueue
  init() throws {
    guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
      throw AudioError.invalid("Audio decoding requires an available Metal device")
    }
    self.device = device; self.queue = queue
  }
  func withWeight<T>(_ weights: [Float], rows: Int, columns: Int,
    body: (MPSMatrix) throws -> T) throws -> T {
    return try autoreleasepool {
    let buffer = weights.withUnsafeBytes { bytes in device.makeBuffer(bytes: bytes.baseAddress!, length: bytes.count, options: .storageModeShared) }
    guard let buffer else { throw AudioError.invalid("Cannot allocate audio weight buffer") }
    let matrix = MPSMatrix(buffer: buffer, descriptor: MPSMatrixDescriptor(rows: rows, columns: columns, rowBytes: columns * 4, dataType: .float32))
    return try body(matrix)
    }
  }
  func multiply(_ a: [Float], weight: MPSMatrix, rows: Int, columns: Int, inner: Int) throws -> [Float] {
    return try autoreleasepool {
    try Task.checkCancellation()
    guard let input = a.withUnsafeBytes({ device.makeBuffer(bytes: $0.baseAddress!, length: rows * inner * 4, options: .storageModeShared) }),
          let output = device.makeBuffer(length: rows * columns * 4, options: .storageModeShared),
          let command = queue.makeCommandBuffer() else { throw AudioError.invalid("Cannot allocate bounded audio convolution tile") }
    let left = MPSMatrix(buffer: input, descriptor: MPSMatrixDescriptor(rows: rows, columns: inner, rowBytes: inner * 4, dataType: .float32))
    let result = MPSMatrix(buffer: output, descriptor: MPSMatrixDescriptor(rows: rows, columns: columns, rowBytes: columns * 4, dataType: .float32))
    let multiplication = MPSMatrixMultiplication(device: device, transposeLeft: false, transposeRight: true,
      resultRows: rows, resultColumns: columns, interiorColumns: inner, alpha: 1, beta: 0)
    multiplication.encode(commandBuffer: command, leftMatrix: left, rightMatrix: weight, resultMatrix: result)
    command.commit(); command.waitUntilCompleted()
    guard command.status == .completed else { throw AudioError.invalid("Audio Metal multiplication failed: \(command.error?.localizedDescription ?? "unknown error")") }
    try Task.checkCancellation()
    return Array(UnsafeBufferPointer(start: output.contents().assumingMemoryBound(to: Float.self), count: rows * columns))
    }
  }
}
