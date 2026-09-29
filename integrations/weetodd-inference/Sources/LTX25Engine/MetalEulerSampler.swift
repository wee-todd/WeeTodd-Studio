import Foundation
import Metal

/// Encodes a Float32 update into an existing command buffer. Inputs and output remain
/// on the GPU; the stage owner controls synchronization, cancellation and buffer lifetime.
public final class MetalEulerSampler {
  private let pipeline: MTLComputePipelineState

  public init(device: MTLDevice) throws {
    guard let url = Bundle.module.url(forResource: "Euler", withExtension: "metal", subdirectory: "Shaders") else {
      throw LTXError.invalid("The LTX Euler shader resource is missing.")
    }
    let options = MTLCompileOptions()
    options.fastMathEnabled = false
    let library = try device.makeLibrary(source: String(contentsOf: url, encoding: .utf8), options: options)
    guard let kernel = library.makeFunction(name: "ltx25_euler") else {
      throw LTXError.invalid("The LTX Euler shader entry point is missing.")
    }
    pipeline = try device.makeComputePipelineState(function: kernel)
  }

  private struct Parameters {
    var sampleScale: Float
    var predictionScale: Float
    var noiseScale: Float
    var count: UInt32
    var channels: UInt32
    var flags: UInt32
    var velocitySigma: Float
  }

  public func encode(commandBuffer: MTLCommandBuffer, sample: MTLBuffer, denoised: MTLBuffer,
    noise: MTLBuffer? = nil, clean: MTLBuffer? = nil, mask: MTLBuffer? = nil,
    output: MTLBuffer, elementCount: Int, channels: Int, step: EulerStep,
    velocitySigma: Float? = nil) throws {
    if let sigma = velocitySigma {
      guard sigma.isFinite, sigma > 0, sigma <= 1 else { throw LTXError.invalid("Invalid velocity sigma.") }
    }
    guard elementCount > 0, elementCount <= Int(UInt32.max), channels > 0,
          channels <= elementCount, elementCount % channels == 0 else {
      throw LTXError.invalid("LTX latent dimensions must contain complete nonempty channel rows.")
    }
    guard (clean == nil) == (mask == nil), !step.ancestral || noise != nil else {
      throw LTXError.invalid("LTX ancestral sampling requires noise; conditioning requires both clean latents and a mask.")
    }
    let bytes = elementCount * MemoryLayout<Float>.stride
    let registryID = pipeline.device.registryID
    guard commandBuffer.commandQueue.device.registryID == registryID,
          commandBuffer.status == .notEnqueued else {
      throw LTXError.invalid("LTX sampling requires an uncommitted command buffer on the sampler device.")
    }
    for buffer in [sample, denoised, output] + [noise, clean].compactMap({ $0 }) {
      guard buffer.device.registryID == registryID, buffer.length >= bytes else {
        throw LTXError.invalid("LTX tensor buffer has the wrong device or insufficient storage.")
      }
    }
    if let mask {
      guard mask.device.registryID == registryID, mask.length >= elementCount / channels * 4,
            mask !== output else {
        throw LTXError.invalid("LTX mask buffer has the wrong device, insufficient rows or aliases the output.")
      }
    }
    var parameters = Parameters(sampleScale: step.sampleScale, predictionScale: step.predictionScale,
      noiseScale: step.noiseScale, count: UInt32(elementCount), channels: UInt32(channels),
      flags: (clean != nil ? 1 : 0) | (step.terminal ? 2 : 0) | (step.ancestral ? 4 : 0) | (velocitySigma != nil ? 8 : 0),
      velocitySigma: velocitySigma ?? 0)
    guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
      throw LTXError.invalid("Cannot create the LTX sampler command encoder.")
    }
    encoder.label = "LTX 2.5 Euler update"
    encoder.setComputePipelineState(pipeline)
    for (index, buffer) in [sample, denoised, noise ?? sample, clean ?? sample, mask ?? sample, output].enumerated() {
      encoder.setBuffer(buffer, offset: 0, index: index)
    }
    encoder.setBytes(&parameters, length: MemoryLayout<Parameters>.stride, index: 6)
    let width = min(256, pipeline.maxTotalThreadsPerThreadgroup)
    encoder.dispatchThreads(MTLSize(width: elementCount, height: 1, depth: 1),
      threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
    encoder.endEncoding()
  }
}
