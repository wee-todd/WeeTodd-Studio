import CryptoKit
import Darwin
import Foundation
import MLX
import MLXNN
import TensorIO

/// The released 322-tensor, 24-channel H3 learned spatial upscaler. Header
/// admission and digest binding precede tensor reads; audio is never consumed.
public struct H3LearnedUpscalerLayout: Sendable {
  public let headerSHA256: String
  public let residentWeightBytes: UInt64
  let prefix: String

  static var expectedShapes: [String: [UInt64]] {
    var shapes: [String: [UInt64]] = [
      "conv_in.weight": [512,24,3,3,3], "conv_in.bias": [512],
      "conv_out.weight": [24,512,3,3,3], "conv_out.bias": [24],
      "embed.0.weight": [64,1], "embed.0.bias": [64],
      "embed.2.weight": [64,64], "embed.2.bias": [64],
      "norm_out.weight": [512], "norm_out.bias": [512],
    ]
    for family in ["in_blocks", "out_blocks"] {
      for index in 0..<18 {
        let p = "\(family).\(index)."
        if index % 3 == 1 {
          shapes[p + "norm.weight"] = [512]; shapes[p + "norm.bias"] = [512]
          shapes[p + "dwconv.weight"] = [512,1,5,1,1]; shapes[p + "dwconv.bias"] = [512]
          shapes[p + "pwconv.weight"] = [512,512,1,1,1]; shapes[p + "pwconv.bias"] = [512]
        } else {
          for norm in ["in_layers.0", "out_norm"] {
            shapes[p + norm + ".weight"] = [512]; shapes[p + norm + ".bias"] = [512]
          }
          for conv in ["in_layers.2", "out_layers.2"] {
            shapes[p + conv + ".weight"] = [512,512,3,3,3]
            shapes[p + conv + ".bias"] = [512]
          }
          shapes[p + "emb_layers.1.weight"] = [1024,64]
          shapes[p + "emb_layers.1.bias"] = [1024]
        }
      }
    }
    return shapes
  }

  static func validate(descriptors: [String: TensorDescriptor]) throws -> (String, UInt64) {
    let expected = expectedShapes
    let prefixed = descriptors["upscaler.conv_in.weight"] != nil
    let prefix = prefixed ? "upscaler." : ""
    guard descriptors.count == 322, expected.count == 322,
      Set(descriptors.keys) == Set(expected.keys.map { prefix + $0 }) else {
      throw H3CheckpointError.invalid("H3 learned upscaler requires exactly the released 322-tensor architecture.")
    }
    var bytes: UInt64 = 0
    for (name, shape) in expected {
      try Task.checkCancellation()
      guard let value = descriptors[prefix + name], value.dtype == "BF16", value.shape == shape,
        value.byteCount == shape.reduce(1, *) * 2 else {
        throw H3CheckpointError.invalid("Malformed learned H3 tensor: \(name)")
      }
      bytes += value.byteCount
    }
    guard bytes == 690_560_432, bytes <= 768 * 1024 * 1024 else {
      throw H3CheckpointError.invalid("H3 learned-upscaler weights exceed the staged resident budget.")
    }
    return (prefix, bytes)
  }

  /// Digest includes the 8-byte little-endian length prefix and original JSON.
  static func headerDigest(url: URL, file: SafeTensorFile) throws -> String {
    let fd = Darwin.open(url.path, O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK)
    guard fd >= 0 else { throw H3CheckpointError.invalid("Cannot open learned H3 header.") }
    let handle = FileHandle(fileDescriptor: fd, closeOnDealloc: true)
    defer { try? handle.close() }
    var before = stat(), after = stat(), named = stat()
    guard fstat(fd, &before) == 0, lstat(url.path, &named) == 0,
      before.st_mode & S_IFMT == S_IFREG, before.st_dev == named.st_dev,
      before.st_ino == named.st_ino, before.st_size == named.st_size else {
      throw H3CheckpointError.invalid("Learned H3 source changed or is not a regular file.")
    }
    try file.checkUnchanged(at: url); try Task.checkCancellation()
    let prefix = try handle.read(upToCount: 8) ?? Data()
    guard prefix.count == 8 else { throw H3CheckpointError.invalid("Truncated learned H3 header.") }
    let length = prefix.withUnsafeBytes { UInt64(littleEndian: $0.loadUnaligned(as: UInt64.self)) }
    guard length > 0, length <= 1024 * 1024, UInt64(before.st_size) >= length + 8 else {
      throw H3CheckpointError.invalid("Unbounded learned H3 header.")
    }
    let json = try handle.read(upToCount: Int(length)) ?? Data()
    guard json.count == Int(length), fstat(fd, &after) == 0,
      before.st_dev == after.st_dev, before.st_ino == after.st_ino,
      before.st_size == after.st_size,
      before.st_mtimespec.tv_sec == after.st_mtimespec.tv_sec,
      before.st_mtimespec.tv_nsec == after.st_mtimespec.tv_nsec,
      before.st_ctimespec.tv_sec == after.st_ctimespec.tv_sec,
      before.st_ctimespec.tv_nsec == after.st_ctimespec.tv_nsec else {
      throw H3CheckpointError.invalid("Learned H3 header changed during admission.")
    }
    try file.checkUnchanged(at: url); try Task.checkCancellation()
    return SHA256.hash(data: prefix + json).map { String(format: "%02x", $0) }.joined()
  }

  public init(url: URL, expectedHeaderSHA256: String) throws {
    guard expectedHeaderSHA256.utf8.count == 64,
      expectedHeaderSHA256.utf8.allSatisfy({ (48...57).contains($0) || (97...102).contains($0) }) else {
      throw H3CheckpointError.invalid("Learned H3 upscaling requires an explicit header SHA256 binding.")
    }
    let file = try SafeTensorFile(url: url)
    let (prefix, bytes) = try Self.validate(descriptors: file.tensors)
    let digest = try Self.headerDigest(url: url, file: file)
    guard digest == expectedHeaderSHA256 else {
      throw H3CheckpointError.invalid("Learned H3 upscaler header differs from its saved binding.")
    }
    self.prefix = prefix; self.residentWeightBytes = bytes; self.headerSHA256 = digest
  }
}

/// Independent operation helpers permit tiny full-versus-halo-convolution and
/// global-norm tests without loading the released network.
enum H3LearnedUpscalerOps {
  /// The learned model consumes sampler-normalized24-channel latents. The
  /// VAE decoder's mean/std are applied only when decoding final video pixels.
  static func unpackSamplerRows(_ rows: MLXArray, geometry: H3Geometry) throws -> MLXArray {
    try H3LatentCodec.videoDecoderInput(rows: rows, latentFrames: geometry.videoLatentFrames,
      latentHeight: geometry.height/16, latentWidth: geometry.width/16,
      mean: [Float](repeating: 0, count: 24),
      standardDeviation: [Float](repeating: 1, count: 24))
  }

  static func packSamplerLatents(_ latents: MLXArray) throws -> MLXArray {
    try H3LatentCodec.videoEncoderRows(latents: latents,
      mean: [Float](repeating: 0, count: 24),
      standardDeviation: [Float](repeating: 1, count: 24))
  }

  static func convolve(_ input: MLXArray, weight: MLXArray, bias: MLXArray,
    plan: H3UpscalerConvolutionPlan) throws -> MLXArray {
    guard input.ndim == 5, input.shape[0] == 1, weight.ndim == 5,
      [1,3].contains(weight.shape[1]), weight.shape[1] == weight.shape[2],
      weight.shape[1] == weight.shape[3], weight.shape[4] == input.shape[4],
      bias.shape == [weight.shape[0]], input.dtype.isFloatingPoint,
      weight.dtype.isFloatingPoint else {
      throw H3CheckpointError.invalid("Invalid learned H3 convolution inputs.")
    }
    let radius = weight.shape[1] / 2
    _ = try plan.validate(channels: input.shape[4], kernel: weight.shape[1], bytesPerElement: 4)
    _ = try plan.tileCount(frames: input.shape[1], height: input.shape[2], width: input.shape[3])
    var temporal: [MLXArray] = []
    for t in stride(from: 0, to: input.shape[1], by: plan.frames) {
      let te = min(t + plan.frames, input.shape[1])
      var vertical: [MLXArray] = []
      for y in stride(from: 0, to: input.shape[2], by: plan.height) {
        let ye = min(y + plan.height, input.shape[2])
        var horizontal: [MLXArray] = []
        for x in stride(from: 0, to: input.shape[3], by: plan.width) {
          try Task.checkCancellation()
          let xe = min(x + plan.width, input.shape[3])
          // Crop true neighbors first, then pad only missing global-boundary
          // values. A tile boundary is never treated as a zero boundary.
          let ts = max(0,t-radius), ys = max(0,y-radius), xs = max(0,x-radius)
          let tz = min(input.shape[1],te+radius), yz = min(input.shape[2],ye+radius)
          let xz = min(input.shape[3],xe+radius)
          let output = try autoreleasepool { () throws -> MLXArray in
            let local = input[0..<1, ts..<tz, ys..<yz, xs..<xz, 0..<input.shape[4]]
            let halo = padded(local, widths: [IntOrPair(0),
              IntOrPair((max(0,radius-t),max(0,te+radius-input.shape[1]))),
              IntOrPair((max(0,radius-y),max(0,ye+radius-input.shape[2]))),
              IntOrPair((max(0,radius-x),max(0,xe+radius-input.shape[3]))), IntOrPair(0)])
            let value = MLX.conv3d(halo, weight) + bias
            eval(value); try Task.checkCancellation()
            guard value.shape == [1,te-t,ye-y,xe-x,weight.shape[0]],
              UInt64(Memory.activeMemory) <= H3CanvasAdmission.maximumStageBytes else {
              throw H3CheckpointError.invalid("Learned H3 convolution exceeded shape or 32 GiB active-memory admission.")
            }
            return value
          }
          horizontal.append(output)
        }
        let row = concatenated(horizontal, axis: 3); eval(row)
        vertical.append(row)
      }
      let clip = concatenated(vertical, axis: 2); eval(clip)
      temporal.append(clip)
    }
    let result = concatenated(temporal, axis: 1); eval(result)
    try Task.checkCancellation(); return result
  }

  static func normalized(_ input: MLXArray, weight: MLXArray, bias: MLXArray,
    groups: Int = 32) throws -> MLXArray {
    guard input.ndim == 5, groups > 0, input.shape[4] % groups == 0,
      weight.shape == [input.shape[4]], bias.shape == weight.shape else {
      throw H3CheckpointError.invalid("Invalid learned H3 global group normalization.")
    }
    let normalization = GroupNorm(groupCount: groups, dimensions: input.shape[4],
      eps: 1e-5, affine: false, pytorchCompatible: true)
    // OWN Python GroupNorm affine order is weight * normalized + bias.
    return weight * normalization(input) + bias
  }

  static func resizeBilinear(_ input: MLXArray, height: Int, width: Int) -> MLXArray {
    func resize(_ value: MLXArray, axis: Int, size: Int) -> MLXArray {
      let source = value.shape[axis]
      let destination = MLXArray(0..<size).asType(.float32)
      let position = (destination + Float(0.5)) * Float(Double(source) / Double(size)) - Float(0.5)
      let lower = floor(position).asType(.int32)
      var shape = [Int](repeating: 1, count: 5); shape[axis] = size
      let low = take(value, clip(lower, min: Int32(0), max: Int32(source-1)), axis: axis)
      let high = take(value, clip(lower + Int32(1), min: Int32(0), max: Int32(source-1)), axis: axis)
      return low + (high-low) * (position-lower.asType(.float32)).reshaped(shape)
    }
    return resize(resize(input, axis: 2, size: height), axis: 3, size: width)
  }
}

public enum H3LearnedLatentUpscaler {
  public struct Result {
    public let videoRows: MLXArray
    public let headerSHA256: String
    public let residentWeightBytes: UInt64
    public let maximumConvolutionWorkspaceBytes: UInt64
    public let targetTilesPerConvolution: Int
    public let loadSeconds: Double
    public let upscaleSeconds: Double
  }

  static let latentMean: [Float] = [0.858090341091156, -0.9606591463088989, 1.0661640167236328, -0.5090325474739075, -0.2727581858634949, -1.3675414323806763, -0.2553254961967468, -0.26907554268836975, -0.5376840829849243, -0.0464097298681736, 0.6657370328903198, 0.19690127670764923, -0.5460608005523682, -0.4035342037677765, -0.23683024942874908, 0.25928452610969543, -0.30133944749832153, 0.211341992020607, -1.1206848621368408, 0.3581933379173279, -0.04225143790245056, 0.2604829967021942, 0.22864092886447906, 0.7056031823158264]
  static let latentStandardDeviation: [Float] = [1.2223774194717407, 1.2767263650894165, 1.6831774711608887, 1.7549455165863037, 1.5636216402053833, 2.194143533706665, 0.9653137922286987, 1.0569885969161987, 0.841948926448822, 0.7729952931404114, 1.8955937623977661, 0.946841835975647, 0.7996809482574463, 0.44988900423049927, 0.7197399735450745, 0.6936293244361877, 2.961095094680786, 2.7694199085235596, 3.0496184825897217, 2.1088054180145264, 3.276226282119751, 3.1627357006073, 2.2816812992095947, 2.6127843856811523]

  /// Staged before the transformer. The returned materialized normalized rows
  /// own no upscaler weights; success, failure and cancellation drop the stage.
  public static func upscale(rows: MLXArray, source: H3Geometry, target: H3Geometry,
    checkpointURL: URL, expectedHeaderSHA256: String, videoVAE: URL,
    progress: (Int,Int) -> Void = { _,_ in }) throws -> Result {
    guard source.frames == target.frames, target.canvasAdmission == .spatialRefinement,
      target.width > source.width, target.height > source.height,
      target.width <= 2*source.width, target.height <= 2*source.height,
      rows.shape == [1,source.videoRows,96], rows.dtype == .float32 else {
      throw H3CheckpointError.invalid("Learned H3 spatial refinement needs matching complete source video and bounded enlarged target.")
    }
    try source.canvasAdmission.validate(width: source.width, height: source.height)
    try target.canvasAdmission.validate(width: target.width, height: target.height)
    try target.canvasAdmission.validatePackedRows(target.videoRows + target.audioRows + 1)
    let layout = try H3LearnedUpscalerLayout(url: checkpointURL, expectedHeaderSHA256: expectedHeaderSHA256)
    let file = try SafeTensorFile(url: checkpointURL)
    _ = try H3VideoVAELayout(url: videoVAE)
    let plan = try H3UpscalerConvolutionPlan()
    let workspace = try plan.validate(channels: 512,kernel:3,bytesPerElement:4)
    let tileCount = try plan.tileCount(frames: target.videoLatentFrames,
      height: target.height/16,width:target.width/16)
    let previousLimit = Memory.cacheLimit
    Memory.cacheLimit = 128 * 1024 * 1024
    var weights: [String:MLXArray] = [:]
    defer {
      weights.removeAll(); Stream.gpu.synchronize(); Memory.clearCache()
      Memory.cacheLimit = previousLimit
    }
    progress(0,39)
    let loadStarted = Date()
    for (name,shape) in H3LearnedUpscalerLayout.expectedShapes.sorted(by: { $0.key < $1.key }) {
      try Task.checkCancellation()
      let value = try H3TensorPayload.withTensorBytes(file:file,name:layout.prefix+name,
        maximumBufferedBytes:16*1024*1024) {
        MLXArray($0,shape.map(Int.init),type:UInt16.self).view(dtype:.bfloat16)
      }
      eval(value); weights[name] = value
    }
    try file.checkUnchanged(at:checkpointURL)
    let loadSeconds = Date().timeIntervalSince(loadStarted)
    let upscaleStarted = Date()
    func read(_ name: String) throws -> MLXArray {
      guard let value = weights[name] else { throw H3CheckpointError.invalid("Missing staged learned H3 tensor: \(name)") }
      return value
    }
    func convolution(_ input:MLXArray,_ name:String) throws -> MLXArray {
      try H3LearnedUpscalerOps.convolve(input,
        weight:read(name+".weight").transposed(0,2,3,4,1),bias:read(name+".bias"),plan:plan)
    }
    func norm(_ input:MLXArray,_ name:String) throws -> MLXArray {
      try H3LearnedUpscalerOps.normalized(input,weight:read(name+".weight"),bias:read(name+".bias"))
    }
    func linear(_ input:MLXArray,_ name:String) throws -> MLXArray {
      try addMM(read(name+".bias"),input,read(name+".weight").T)
    }
    func block(_ input:MLXArray,_ name:String,_ embedding:MLXArray,_ temporal:Bool) throws -> MLXArray {
      if temporal {
        let normalized = try silu(norm(input,name+".norm"))
        let depthwise = try read(name+".dwconv.weight")[0..<512,0,0..<5,0,0].transposed(1,0)
        let paddedTime = padded(normalized,widths:[IntOrPair(0),IntOrPair((2,2)),IntOrPair(0),IntOrPair(0),IntOrPair(0)])
        var sum = try read(name+".dwconv.bias")
        for tap in 0..<5 {
          try Task.checkCancellation()
          sum = sum + paddedTime[0..<1,tap..<(tap+input.shape[1]),0..<input.shape[2],0..<input.shape[3],0..<512] * depthwise[tap]
        }
        return try input + convolution(sum,name+".pwconv")
      }
      let hidden = try convolution(silu(norm(input,name+".in_layers.0")),name+".in_layers.2")
      let modulation = try linear(silu(embedding),name+".emb_layers.1").asType(hidden.dtype)
      let scale = modulation[0..<1,0..<512].reshaped([1,1,1,1,512])
      let shift = modulation[0..<1,512..<1024].reshaped([1,1,1,1,512])
      let modulated = try norm(hidden,name+".out_norm") * (Float(1)+scale) + shift
      return try input + convolution(silu(modulated),name+".out_layers.2")
    }
    let raw = try H3LearnedUpscalerOps.unpackSamplerRows(rows, geometry: source)
    let mean = MLXArray(latentMean).reshaped([1,1,1,1,24])
    let std = MLXArray(latentStandardDeviation).reshaped([1,1,1,1,24])
    let scale = sqrt(Double(target.width*target.height)/Double(source.width*source.height))
    let embedding = try linear(silu(linear(MLXArray([Float(scale-1)],[1,1]),"embed.0")),"embed.2")
    var completed = 0
    func materialize(_ value:MLXArray) throws -> MLXArray {
      eval(value); try Task.checkCancellation()
      guard UInt64(Memory.activeMemory) <= H3CanvasAdmission.maximumStageBytes,
        all(isFinite(value)).item(Bool.self) else {
        throw H3CheckpointError.invalid("Learned H3 upscaler exceeded its active-memory or finite-output contract.")
      }
      completed += 1; progress(completed,39); return value
    }
    var value = try materialize(convolution((raw-mean)/std,"conv_in"))
    for index in 0..<18 {
      value = try autoreleasepool { try materialize(block(value,"in_blocks.\(index)",embedding,index%3 == 1)) }
    }
    value = try materialize(H3LearnedUpscalerOps.resizeBilinear(value,height:target.height/16,width:target.width/16))
    for index in 0..<18 {
      value = try autoreleasepool { try materialize(block(value,"out_blocks.\(index)",embedding,index%3 == 1)) }
    }
    let output = try materialize(convolution(silu(norm(value,"norm_out")),"conv_out"))
    let packed = try H3LearnedUpscalerOps.packSamplerLatents(output*std+mean)
    try file.checkUnchanged(at:checkpointURL)
    guard try H3LearnedUpscalerLayout.headerDigest(url:checkpointURL,file:file) == layout.headerSHA256 else {
      throw H3CheckpointError.invalid("Learned H3 source changed during execution.")
    }
    return Result(videoRows:packed,headerSHA256:layout.headerSHA256,
      residentWeightBytes:layout.residentWeightBytes,
      maximumConvolutionWorkspaceBytes:workspace,targetTilesPerConvolution:tileCount,
      loadSeconds:loadSeconds,upscaleSeconds:Date().timeIntervalSince(upscaleStarted))
  }
}
