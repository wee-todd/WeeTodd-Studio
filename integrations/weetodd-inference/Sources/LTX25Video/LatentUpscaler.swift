import Foundation
import TensorIO

public struct LatentUpscaleConfiguration: Sendable {
  public var maximumActivationBytes = 512*1024*1024
  public var maximumWeightBytes = 256*1024*1024
  public var convolutionWindowSites = 128
  public var windowsPerCommand = 16
  public init() {}
}

/// Packed THWC, batch one. Normalization spans the full volume; independent tiles
/// would change the network. The plan bounds full activations and one weight layer.
public struct LatentUpscalePlan: Sendable {
  public let outputShape: [Int]
  public let activationBytes: Int
  public static let largestWeightBytes = 4096*1024*9*4
  public init(shape: [Int],configuration c: LatentUpscaleConfiguration = .init()) throws {
    guard shape.count == 4, (1...129).contains(shape[0]), (1...128).contains(shape[1]),
      (1...128).contains(shape[2]), shape[3] == 128,
      (1...4096).contains(c.convolutionWindowSites), (1...64).contains(c.windowsPerCommand),
      c.maximumActivationBytes > 0, c.maximumWeightBytes >= Self.largestWeightBytes else {
      throw VideoDecodeError.invalid("Unsupported upscaler geometry or insufficient single-layer budget.")
    }
    let sites = UInt64(shape[0]*shape[1]*shape[2]), high = sites*4
    // Six high-resolution hidden tensors conservatively cover residual, norm and
    // convolution overlaps. Include caller/input/output copies and bounded im2col.
    let bytes = high*1024*4*6 + (sites*128*2+high*128*2)*4
      + UInt64(c.convolutionWindowSites)*1024*27*4 + 4*1024*1024
    guard bytes <= UInt64(c.maximumActivationBytes), high*1024 <= UInt64(UInt32.max) else {
      throw VideoDecodeError.invalid("Full-context latent upscaling exceeds its activation budget.")
    }
    activationBytes = Int(bytes); outputShape = [shape[0],shape[1]*2,shape[2]*2,128]
  }
}

/// Independently implemented released spatial x2 network. Original checkpoint
/// weights are decoded into one Metal layer at a time; no complete-weight cache.
public final class LatentUpscaler {
  private let checkpoint: URL
  private let statisticsCheckpoint: URL
  private let lock = NSLock()
  public private(set) var residentWeightBytes = 0
  public init(checkpoint: URL,statisticsCheckpoint: URL) throws {
    try Self.validate(SafeTensorFile(url: checkpoint))
    try Self.validateStatistics(SafeTensorFile(url: statisticsCheckpoint))
    self.checkpoint = checkpoint; self.statisticsCheckpoint = statisticsCheckpoint
  }

  public func upscale(packed: [Float],shape: [Int],configuration c: LatentUpscaleConfiguration = .init(),
    checkCancelled: () throws -> Void = { try Task.checkCancellation() },
    progress: (String) throws -> Void = { _ in }) throws -> [Float] {
    let plan = try LatentUpscalePlan(shape: shape,configuration: c)
    guard packed.count == shape.reduce(1,*), FloatValidation.allFinite(packed), lock.try() else {
      throw VideoDecodeError.invalid("Invalid upscaler input or overlapping evaluation.")
    }
    defer { residentWeightBytes = 0; lock.unlock() }
    try checkCancelled()
    return try autoreleasepool {
      let file = try SafeTensorFile(url: checkpoint), stats = try SafeTensorFile(url: statisticsCheckpoint)
      try Self.validate(file); try Self.validateStatistics(stats)
      let mean = try stats.readFloat32(named: "per_channel_statistics.mean-of-means")
      let std = try stats.readFloat32(named: "per_channel_statistics.std-of-means")
      guard FloatValidation.allFinite(mean),std.allSatisfy({ $0.isFinite && $0 > 0 }) else {
        throw VideoDecodeError.invalid("Invalid latent normalization statistics.")
      }
      let gpu = try VideoGPU()
      var x = try autoreleasepool {
        var values = packed
        for i in values.indices {
          if i % 4096 == 0 { try checkCancelled() }
          values[i] = values[i]*std[i%128]+mean[i%128]
        }
        return try gpu.tensor(values,shape: shape)
      }
      func conv(_ input: VideoTensor,_ name: String) throws -> VideoTensor {
        try checkCancelled()
        return try autoreleasepool {
          let descriptor = file.tensors[name+".weight"]!
          let count = Int(descriptor.shape.reduce(1,*)), channels = Int(descriptor.shape[0])
          let buffer = try gpu.buffer(count: count)
          residentWeightBytes = count*4
          defer { residentWeightBytes = 0 }
          for start in stride(from: 0,to: count,by: 1024*1024) {
            try checkCancelled()
            let values = try file.readFloat32(named: name+".weight",elements: UInt64(start)..<UInt64(min(count,start+1024*1024)),maximumBytes: 4*1024*1024)
            guard FloatValidation.allFinite(values) else { throw VideoDecodeError.invalid("Nonfinite upscaler convolution weights.") }
            values.withUnsafeBytes { buffer.contents().advanced(by: start*4).copyMemory(from: $0.baseAddress!,byteCount: $0.count) }
          }
          let bias = try file.readFloat32(named: name+".bias")
          guard FloatValidation.allFinite(bias) else { throw VideoDecodeError.invalid("Nonfinite upscaler bias.") }
          let b = try gpu.tensor(bias,shape: [1,1,1,channels])
          return try gpu.convolution(input,weights: buffer,bias: b.buffer,outputChannels: channels,
            causal: false,windowSites: c.convolutionWindowSites,windowsPerCommand: c.windowsPerCommand,
            kernelDepth: descriptor.shape.count == 4 ? 1 : 3,zeroTemporalPadding: true,checkCancelled: checkCancelled)
        }
      }
      func norm(_ input: VideoTensor,_ name: String,_ residual: VideoTensor? = nil) throws -> VideoTensor {
        try gpu.groupNormSilu(input,weight: file.readFloat32(named: name+".weight"),
          bias: file.readFloat32(named: name+".bias"),residual: residual,checkCancelled: checkCancelled)
      }
      x = try norm(conv(x,"initial_conv"),"initial_norm")
      try progress("initial")
      for stage in ["res_blocks","post_upsample_res_blocks"] {
        if stage == "post_upsample_res_blocks" {
          x = try autoreleasepool { try gpu.shuffle(conv(x,"upsampler.0"),spatial: 2,temporal: 1) }
          try progress("spatial_x2")
        }
        for block in 0..<4 {
          x = try autoreleasepool {
            let residual = x, stem = "\(stage).\(block)"
            let a = try norm(conv(x,stem+".conv1"),stem+".norm1")
            return try norm(conv(a,stem+".conv2"),stem+".norm2",residual)
          }
          try progress("\(stage).\(block)")
        }
      }
      x = try conv(x,"final_conv")
      guard x.shape == plan.outputShape else { throw VideoDecodeError.invalid("Upscaler output geometry mismatch.") }
      var output = gpu.values(x)
      for i in output.indices {
        if i % 4096 == 0 { try checkCancelled() }
        output[i] = (output[i]-mean[i%128])/std[i%128]
      }
      guard FloatValidation.allFinite(output) else { throw VideoDecodeError.invalid("Nonfinite upscaler output.") }
      try progress("complete")
      return output
    }
  }

  private static func validateStatistics(_ file: SafeTensorFile) throws {
    for name in ["mean-of-means","std-of-means"] {
      guard let d = file.tensors["per_channel_statistics."+name],d.shape == [128],
        ["F32","F16","BF16"].contains(d.dtype) else { throw VideoDecodeError.invalid("Missing latent channel statistics.") }
    }
  }
  private static func validate(_ file: SafeTensorFile) throws {
    guard let json = file.metadata["config"],let data = json.data(using: .utf8),
      let config = try JSONSerialization.jsonObject(with: data) as? [String:Any],
      config["_class_name"] as? String == "LatentUpsampler",config["in_channels"] as? Int == 128,
      config["mid_channels"] as? Int == 1024,config["num_blocks_per_stage"] as? Int == 4,
      config["dims"] as? Int == 3,config["spatial_upsample"] as? Bool == true,
      config["temporal_upsample"] as? Bool == false,config["spatial_scale"] as? Double == 2,
      config["rational_resampler"] as? Bool == false else {
      throw VideoDecodeError.invalid("Checkpoint is not the supported LTX 2.5 spatial x2 upscaler.")
    }
    var expected: [String:[UInt64]] = [:]
    func add(_ stem: String,_ shape: [UInt64]) { expected[stem+".weight"] = shape; expected[stem+".bias"] = [shape[0]] }
    add("initial_conv",[1024,128,3,3,3]); add("initial_norm",[1024])
    add("upsampler.0",[4096,1024,3,3]); add("final_conv",[128,1024,3,3,3])
    for stage in ["res_blocks","post_upsample_res_blocks"] { for block in 0..<4 { for layer in 1...2 {
      add("\(stage).\(block).conv\(layer)",[1024,1024,3,3,3])
      add("\(stage).\(block).norm\(layer)",[1024])
    } } }
    guard Set(file.tensors.keys) == Set(expected.keys),expected.allSatisfy({ name,shape in
      file.tensors[name]?.shape == shape && ["F32","F16","BF16"].contains(file.tensors[name]!.dtype)
    }) else { throw VideoDecodeError.invalid("Spatial upscaler tensor layout differs from its released contract.") }
  }
}
