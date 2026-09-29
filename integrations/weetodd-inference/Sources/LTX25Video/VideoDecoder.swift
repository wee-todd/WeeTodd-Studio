import Foundation
import Metal
import TensorIO

/// An exact, bounded decode window. Larger scene stitching is deliberately not
/// implicit: noncausal convolutions require context on both sides of a boundary.
public struct VideoDecodeConfiguration: Sendable {
  public var frameRate: Double = 24
  public var maximumLatentFrames = 17
  public var maximumLatentHeight = 64
  public var maximumLatentWidth = 64
  public var maximumActivationBytes = 512 * 1024 * 1024
  public var maximumWeightBytes = 512 * 1024 * 1024
  public var convolutionWindowSites = 128
  public var windowsPerCommand = 16
  public var fuseNormalization = true
  public init() {}
}

public struct VideoDecodePlan: Sendable {
  /// THWC; batch one and RGB are explicit decoder contracts.
  public let outputShape: [Int]
  public let duration: Double
  public let admittedActivationBytes: Int
  public init(shape: [Int], configuration c: VideoDecodeConfiguration) throws {
    guard shape.count == 5, shape[0] == 1, shape[1] == 128,
      shape[2...].allSatisfy({ $0 > 0 && $0 <= 4096 }),
      c.frameRate.isFinite, c.frameRate > 0, c.frameRate <= 240,
      c.maximumLatentFrames > 0, c.maximumLatentHeight > 0, c.maximumLatentWidth > 0,
      shape[2] <= c.maximumLatentFrames, shape[3] <= c.maximumLatentHeight,
      shape[4] <= c.maximumLatentWidth, c.convolutionWindowSites > 0,
      c.convolutionWindowSites <= 4096, (1...64).contains(c.windowsPerCommand),
      c.maximumActivationBytes > 0, c.maximumWeightBytes > 0 else {
      throw VideoDecodeError.invalid("Video decode requires finite timing and a batch-one 128-channel latent inside the configured temporal/spatial window.")
    }
    var t = UInt64(shape[2]), h = UInt64(shape[3]), w = UInt64(shape[4])
    var peak = t*h*w*1024
    let stages = [(4096,2,2),(4096,2,2),(512,1,2),(512,2,1)]
    for (channels,spatial,temporal) in stages {
      peak = max(peak,t*h*w*UInt64(channels))
      t = t*UInt64(temporal) - (temporal > 1 ? 1 : 0)
      h *= UInt64(spatial); w *= UInt64(spatial)
      peak = max(peak,t*h*w*UInt64(channels/(spatial*spatial*temporal)))
    }
    // Fusion retains residual, convolution input and output, plus one scale per
    // input site (normalized inputs have at least128 channels). It never builds
    // a full normalized activation. Unfused qualification keeps the old bound.
    let bytes = peak*4*(c.fuseNormalization ? 3 : 5)
      + (c.fuseNormalization ? peak/128*4 : 0)
      + UInt64(c.convolutionWindowSites)*1024*27*4
    guard bytes <= UInt64(c.maximumActivationBytes), bytes <= UInt64(Int.max), peak <= UInt64(UInt32.max),
      t*h*w*48 <= UInt64(UInt32.max) else {
      throw VideoDecodeError.invalid("Video activation window exceeds its configured memory/index budget; reduce latent dimensions or explicitly raise the budget.")
    }
    admittedActivationBytes = Int(bytes)
    outputShape = [Int(t),Int(h)*4,Int(w)*4,3]
    let seconds = Double(t)/c.frameRate
    guard seconds.isFinite else { throw VideoDecodeError.invalid("Video duration must be finite.") }
    duration = seconds
  }
}

public struct VideoFrameChunk: Sendable {
  public let startFrame: Int
  public let frameCount: Int
  public let width: Int
  public let height: Int
  public let frameRate: Double
  /// Unclipped floating point RGB in THWC order, conventionally [-1,1].
  public let rgb: [Float]
  public init(startFrame:Int,frameCount:Int,width:Int,height:Int,frameRate:Double,rgb:[Float]) {
    self.startFrame=startFrame;self.frameCount=frameCount;self.width=width;self.height=height;self.frameRate=frameRate;self.rgb=rgb
  }
}

/// Convolutional LTX 2.5 VAE. No Python process, conversion file, cached model or
/// complete checkpoint payload is required. One convolution's weights are loaded
/// directly into shared Metal memory and released after its command completes.
public final class VideoDecoder {
  private let checkpoint: URL
  private var running = false
  public private(set) var residentWeightBytes = 0

  public init(checkpoint: URL) throws {
    let file = try SafeTensorFile(url: checkpoint)
    _ = try Self.validate(file)
    self.checkpoint = checkpoint
  }

  public func decode(latent: [Float], shape: [Int], configuration c: VideoDecodeConfiguration = .init(),
    checkCancelled: () throws -> Void = { try Task.checkCancellation() },
    receive: (VideoFrameChunk) throws -> Void) throws {
    let plan = try VideoDecodePlan(shape: shape, configuration: c)
    guard latent.count == shape.reduce(1,*), latent.allSatisfy(\.isFinite), !running else {
      throw VideoDecodeError.invalid("Video latent payload must be finite and match its shape; decoder calls cannot overlap.")
    }
    try checkCancelled()
    running = true
    defer { running = false; residentWeightBytes = 0 }
    try autoreleasepool {
      let file = try SafeTensorFile(url: checkpoint)
      let causal = try Self.validate(file)
      let largest = Self.layers.map { $0.input * $0.output * 27 * 4 }.max()!
      guard largest <= c.maximumWeightBytes else {
        throw VideoDecodeError.invalid("Video convolution weights exceed the configured single-layer memory budget.")
      }
      let gpu = try VideoGPU()
      let mean = try file.readFloat32(named: "per_channel_statistics.mean-of-means")
      let std = try file.readFloat32(named: "per_channel_statistics.std-of-means")
      guard mean.allSatisfy(\.isFinite), std.allSatisfy({ $0.isFinite && $0 > 0 }) else {
        throw VideoDecodeError.invalid("Video latent channel statistics must be finite with positive standard deviations.")
      }
      let sites = shape[2]*shape[3]*shape[4]
      var arranged = [Float](repeating: 0, count: latent.count)
      for site in 0..<sites {
        if site % 4096 == 0 { try checkCancelled() }
        for channel in 0..<128 { arranged[site*128+channel] = latent[channel*sites+site]*std[channel]+mean[channel] }
      }
      var x = try gpu.tensor(arranged, shape: [shape[2],shape[3],shape[4],128])
      arranged.removeAll(keepingCapacity: false)
      func conv(_ input: VideoTensor, _ name: String, normalized: Bool = false) throws -> VideoTensor {
        try checkCancelled()
        return try autoreleasepool {
          let input = normalized && !c.fuseNormalization ? try gpu.normalize(input) : input
          let descriptor = file.tensors[name+".weight"]!
          let outputChannels = Int(descriptor.shape[0])
          let count = descriptor.shape.reduce(UInt64(1),*)
          let weights = try gpu.buffer(count: Int(count))
          residentWeightBytes = Int(count)*4
          defer { residentWeightBytes = 0 }
          // Never create a full Float32 staging copy alongside the GPU layer.
          for start in stride(from: UInt64(0), to: count, by: 1024*1024) {
            try checkCancelled()
            let values = try file.readFloat32(named: name+".weight", elements: start..<min(count,start+1024*1024),
              maximumBytes: 4*1024*1024)
            guard values.allSatisfy(\.isFinite) else { throw VideoDecodeError.invalid("Video convolution contains nonfinite weights.") }
            values.withUnsafeBufferPointer {
              weights.contents().advanced(by: Int(start)*4).copyMemory(from: $0.baseAddress!, byteCount: values.count*4)
            }
          }
          let biasValues = try file.readFloat32(named: name+".bias")
          guard biasValues.allSatisfy(\.isFinite) else { throw VideoDecodeError.invalid("Video convolution contains nonfinite bias.") }
          let bias = try gpu.tensor(biasValues, shape: [1,1,1,outputChannels])
          return try gpu.convolution(input, weights: weights, bias: bias.buffer, outputChannels: outputChannels,
            causal: causal, windowSites: c.convolutionWindowSites,
            normalizeInput: normalized && c.fuseNormalization, windowsPerCommand: c.windowsPerCommand,
            checkCancelled: checkCancelled)
        }
      }
      x = try conv(x,"decoder.conv_in.conv")
      let blocks = [2,2,4,6,4]
      let scales = [(2,2),(2,2),(1,2),(2,1)]
      for stage in 0..<5 {
        for block in 0..<blocks[stage] {
          let prefix = "decoder.up_blocks.\(stage*2).res_blocks.\(block)"
          let residual = x
          x = try conv(x,prefix+".conv1.conv",normalized: true)
          x = try conv(x,prefix+".conv2.conv",normalized: true)
          x = try gpu.add(x,residual: residual)
        }
        if stage < 4 {
          x = try conv(x,"decoder.up_blocks.\(stage*2+1).conv.conv")
          x = try gpu.shuffle(x,spatial: scales[stage].0,temporal: scales[stage].1)
        }
      }
      x = try conv(x,"decoder.conv_out.conv",normalized: true)
      x = try gpu.shuffle(x,spatial: 4,temporal: 1,unpatch: true)
      guard x.shape == plan.outputShape else { throw VideoDecodeError.invalid("Video decoder output shape disagrees with its admitted plan.") }
      let frameSize = x.shape[1]*x.shape[2]*3
      let pointer = x.buffer.contents().assumingMemoryBound(to: Float.self)
      for frame in 0..<x.shape[0] {
        try checkCancelled()
        let pixels = Array(UnsafeBufferPointer(start: pointer.advanced(by: frame*frameSize),count: frameSize))
        guard pixels.allSatisfy(\.isFinite) else { throw VideoDecodeError.invalid("Video decoder produced nonfinite pixels.") }
        try receive(VideoFrameChunk(startFrame: frame,frameCount: 1,width: x.shape[2],height: x.shape[1],
          frameRate: c.frameRate,rgb: pixels))
      }
    }
  }

  private struct Layer { let name: String; let input: Int; let output: Int }
  private static var layers: [Layer] {
    var layers = [Layer(name: "decoder.conv_in.conv",input: 128,output: 1024)]
    let channels = [1024,512,512,256,128], counts = [2,2,4,6,4], outputs = [4096,4096,512,512]
    for stage in 0..<5 {
      for block in 0..<counts[stage] {
        for conv in 1...2 {
          layers.append(Layer(name: "decoder.up_blocks.\(stage*2).res_blocks.\(block).conv\(conv).conv",
            input: channels[stage],output: channels[stage]))
        }
      }
      if stage < 4 { layers.append(Layer(name: "decoder.up_blocks.\(stage*2+1).conv.conv",input: channels[stage],output: outputs[stage])) }
    }
    layers.append(Layer(name: "decoder.conv_out.conv",input: 128,output: 48))
    return layers
  }

  private static func validate(_ file: SafeTensorFile) throws -> Bool {
    guard file.metadata["model_version"] == "2.5.0",
      let config = file.metadata["config"]?.data(using: .utf8),
      let object = try JSONSerialization.jsonObject(with: config) as? [String: Any],
      let vae = object["vae"] as? [String: Any], vae["_class_name"] as? String == "CausalVideoAutoencoder",
      vae["spatial_padding_mode"] as? String == "zeros", vae["timestep_conditioning"] as? Bool == false,
      let causal = vae["causal_decoder"] as? Bool else {
      throw VideoDecodeError.invalid("Expected the LTX 2.5 convolutional VAE with zero spatial padding; diffusion VAEs are not supported.")
    }
    for layer in layers {
      try require(file, name: layer.name+".weight", shape: [layer.output,layer.input,3,3,3])
      try require(file, name: layer.name+".bias", shape: [layer.output])
    }
    for name in ["mean-of-means","std-of-means"] {
      try require(file,name: "per_channel_statistics."+name,shape: [128])
    }
    return causal
  }

  private static func require(_ file: SafeTensorFile, name: String, shape: [Int]) throws {
    guard let tensor = file.tensors[name], tensor.shape == shape.map(UInt64.init),
      ["F32","BF16"].contains(tensor.dtype) else {
      throw VideoDecodeError.invalid("Missing or incompatible video tensor: \(name)")
    }
  }
}
