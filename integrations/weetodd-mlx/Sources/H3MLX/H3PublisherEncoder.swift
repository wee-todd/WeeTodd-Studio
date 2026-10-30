import CryptoKit
import Foundation
import MLX
import MLXNN
import TensorIO

public struct H3PublisherEncoderLayout:Sendable {
  public static let version="h3-encoder-f32-odhwi-v1"
  public let mean:[Float],standardDeviation:[Float]
  public init(url:URL) throws {
    let file=try SafeTensorFile(url:url)
    guard file.metadata["format"] == Self.version,file.metadata["tensor_layout"] == "ODHWI",
      file.metadata["source_shard_2_sha256"] == H3PublisherEncoderConverter.sourceSHA256[0],
      file.metadata["source_shard_3_sha256"] == H3PublisherEncoderConverter.sourceSHA256[1],
      file.tensors.count == H3VideoVAEEncoder.encoderShapes.count else {
      throw H3CheckpointError.invalid("Publisher encoder requires the dedicated F32 source-derived ODHWI component.")
    }
    for (name,shape) in H3VideoVAEEncoder.encoderShapes {
      guard file.tensors[name]?.dtype == "F32",file.tensors[name]?.shape == shape.map(UInt64.init) else {
        throw H3CheckpointError.invalid("Missing publisher F32 encoder tensor: \(name)")
      }
    }
    mean=H3LearnedLatentUpscaler.latentMean;standardDeviation=H3LearnedLatentUpscaler.latentStandardDeviation
    try file.checkUnchanged(at:url)
  }
}

/// Swift-only, lossless F32 conversion of the official two source shards into
/// an encoder-only component. At most one output-channel convolution plane is
/// transposed at a time; no decoder payload, GPU model or Python is required.
public enum H3PublisherEncoderConverter {
  public static let sourceSHA256=["2e05e8bc23fa4071043e17fd242be8acd0685e781a43987432b2eae925be4198",
    "c05d6ac4b1a33de372799d708531da6320f6a3ce6d1ce6d895e770988e004a39"]
  public static let sourceBytes:[UInt64]=[4_955_986_528,398_539_336]
  public struct Result:Sendable { public let url:URL,sha256:String,payloadBytes:UInt64 }

  static func nativeName(_ source:String) -> String {
    var value=source.replacingOccurrences(of:"encoder.conv_norm_out.",with:"encoder.norm_out.")
      .replacingOccurrences(of:".conv_shortcut.",with:".nin_shortcut.")
    for block in 0..<6 {
      value=value.replacingOccurrences(of:"encoder.down_blocks.\(block).downsamplers.0.conv.",with:"encoder.down.\(block).downsample.conv.")
      for layer in 0..<2 {
        value=value.replacingOccurrences(of:"encoder.down_blocks.\(block).resnets.\(layer).",with:"encoder.down.\(block).block.\(layer).")
      }
    }
    return value
  }
  static func transposePlane(_ words:[UInt32],inputs:Int,depth:Int,height:Int,width:Int) throws -> [UInt32] {
    let spatial=depth*height*width
    guard inputs > 0,spatial > 0,words.count == inputs*spatial,words.count <= 1024*1024 else {
      throw H3CheckpointError.invalid("Invalid bounded F32 encoder convolution plane.")
    }
    var result=[UInt32](repeating:0,count:words.count)
    for position in 0..<spatial {
      for channel in 0..<inputs { result[position*inputs+channel]=words[channel*spatial+position] }
    }
    return result
  }
  private static func hash(_ url:URL) throws -> String {
    let handle=try FileHandle(forReadingFrom:url);defer { try? handle.close() }
    var digest=SHA256()
    while try autoreleasepool(invoking: { () throws -> Bool in
      try Task.checkCancellation()
      guard let data=try handle.read(upToCount:4*1024*1024),!data.isEmpty else { return false }
      digest.update(data:data);return true
    }) {}
    return digest.finalize().map { String(format:"%02x",$0) }.joined()
  }
  public static func convert(shards:[URL],destination:URL,modelRoot:URL,
    progress:(Int,Int)->Void={_,_ in}) throws -> Result {
    let root=modelRoot.resolvingSymlinksInPath().standardizedFileURL.path
    let output=destination.deletingLastPathComponent().resolvingSymlinksInPath().appendingPathComponent(destination.lastPathComponent).standardizedFileURL
    guard shards.count == 2,modelRoot.isFileURL,destination.isFileURL,output.path.hasPrefix(root+"/"),
      !FileManager.default.fileExists(atPath:output.path) else {
      throw H3CheckpointError.invalid("F32 encoder conversion requires two source shards and a new file under the selected model library.")
    }
    let files=try shards.map { try SafeTensorFile(url:$0) }
    for index in 0..<2 {
      guard files[index].fileByteCount == sourceBytes[index],try hash(shards[index]) == sourceSHA256[index] else {
        throw H3CheckpointError.invalid("F32 encoder source shard differs from the pinned released checkpoint.")
      }
      try files[index].checkUnchanged(at:shards[index])
    }
    var sources:[String:(Int,String)]=[:]
    var tensors:[String:SafeTensorStreamWriter.Tensor]=[:]
    for (index,file) in files.enumerated() {
      for (name,tensor) in file.tensors where name.hasPrefix("encoder.") || name.hasPrefix("quant_conv.") {
        let native=nativeName(name),shape=tensor.shape.map(Int.init)
        let converted=shape.count == 5 ? [shape[0],shape[2],shape[3],shape[4],shape[1]]:shape
        guard sources[native] == nil,tensor.dtype == "F32",H3VideoVAEEncoder.encoderShapes[native] == converted else {
          throw H3CheckpointError.invalid("Unsupported or duplicate released encoder tensor: \(name)")
        }
        sources[native]=(index,name);tensors[native]=try .init(dtype:"F32",shape:converted.map(UInt64.init))
      }
    }
    guard tensors.count == 118 else { throw H3CheckpointError.invalid("The complete 118-tensor F32 encoder is required.") }
    let writer=try SafeTensorStreamWriter(url:output,tensors:tensors,metadata:["format":H3PublisherEncoderLayout.version,
      "tensor_layout":"ODHWI","source_shard_2_sha256":sourceSHA256[0],"source_shard_3_sha256":sourceSHA256[1]])
    for (finished,native) in tensors.keys.sorted().enumerated() {
      let (index,name)=sources[native]!,file=files[index],tensor=file.tensors[name]!
      try file.checkUnchanged(at:shards[index]);try Task.checkCancellation()
      if tensor.shape.count == 5 {
        let shape=tensor.shape.map(Int.init),plane=shape[1]*shape[2]*shape[3]*shape[4]
        for out in 0..<shape[0] {
          try autoreleasepool {
            try file.withTensorBytes(named:name,range:UInt64(out*plane*4)..<UInt64((out+1)*plane*4),access:.buffered) { bytes in
              let words=(0..<plane).map { bytes.loadUnaligned(fromByteOffset:$0*4,as:UInt32.self) }
              let converted=try transposePlane(words,inputs:shape[1],depth:shape[2],height:shape[3],width:shape[4])
              try converted.withUnsafeBytes { try writer.append(tensor:native,bytes:$0) }
            }
          }
        }
      } else {
        var offset:UInt64=0
        while offset < tensor.byteCount {
          let end=min(offset+4*1024*1024,tensor.byteCount)
          try file.withTensorBytes(named:name,range:offset..<end,access:.buffered) { try writer.append(tensor:native,bytes:$0) }
          offset=end
        }
      }
      try file.checkUnchanged(at:shards[index]);progress(finished+1,118)
    }
    for index in 0..<2 { try files[index].checkUnchanged(at:shards[index]) }
    let digest=try writer.finish();_ = try H3PublisherEncoderLayout(url:output)
    return Result(url:output,sha256:digest,payloadBytes:writer.payloadBytes)
  }
}

/// Per-frame storage avoids a single 17×H×W×128 tensor crossing signed-32
/// indexing at the 768p draft grid. Causal convolution sees its complete true
/// temporal neighborhood; frame-isolated normalization is never spatially tiled.
enum H3PublisherEncoderOps {
  static func clip(_ pixels:MLXArray,read:(String,[Int]) throws -> MLXArray) throws -> MLXArray {
    guard pixels.ndim == 5,pixels.shape[0] == 1,pixels.dtype == .float32 else {
      throw H3CheckpointError.invalid("Publisher encoder clip requires Float32 pixels.")
    }
    func conv(_ frames:[MLXArray],_ name:String,_ out:Int,_ kernel:Int,_ spatialPadding:Int=0,
      _ temporalPadding:Int=0,_ temporalStride:Int=1,_ spatialStride:Int=1) throws -> [MLXArray] {
      let channels=frames[0].shape[4]
      let weight=try read(name+".weight",[out,kernel,kernel,kernel,channels]),bias=try read(name+".bias",[out])
      return try convolution(frames:frames,weight:weight,bias:bias,spatialPadding:spatialPadding,
        temporalPadding:temporalPadding,temporalStride:temporalStride,spatialStride:spatialStride)
    }
    func norm(_ frames:[MLXArray],_ name:String) throws -> [MLXArray] {
      let count=frames[0].shape[4]
      let weight=try read(name+".weight",[count]),bias=try read(name+".bias",[count])
      return try frames.map { try H3VideoVAEEncoderOps.temporallyIsolatedGroupNorm($0,weight:weight,bias:bias) }
    }
    func activated(_ frames:[MLXArray]) -> [MLXArray] { frames.map { let value=silu($0);eval(value);return value } }
    func block(_ frames:[MLXArray],_ name:String,_ out:Int) throws -> [MLXArray] {
      func projected() throws -> [MLXArray] {
        let first=try conv(activated(norm(frames,name+".norm1")),name+".conv1",out,3,1,2)
        return try conv(activated(norm(first,name+".norm2")),name+".conv2",out,3,1,2)
      }
      let projected=try projected()
      let residual=try frames[0].shape[4] == out ? frames:conv(frames,name+".nin_shortcut",out,1)
      return zip(residual,projected).map { let value=$0+$1;eval(value);return value }
    }
    var frames=(0..<pixels.shape[1]).map { pixels[0..<1,$0..<($0+1),0..<pixels.shape[2],0..<pixels.shape[3],0..<3] }
    frames=try conv(frames,"encoder.conv_in",128,3,1,2)
    let channels=[128,256,256,512,512,1024],temporal=[1,2,2,1,1,1],spatial=[2,2,2,2,1,1]
    for index in 0..<6 {
      for layer in 0..<2 { frames=try block(frames,"encoder.down.\(index).block.\(layer)",channels[index]) }
      if temporal[index]*spatial[index] > 1 {
        if spatial[index] == 2 {
          frames=try frames.map { try H3VideoVAEEncoderOps.reflectSpatial(H3VideoVAEEncoderOps.reflectSpatial($0,axis:2,before:0,after:1),axis:3,before:0,after:1) }
        }
        frames=try conv(frames,"encoder.down.\(index).downsample.conv",channels[index],3,0,2,temporal[index],spatial[index])
      }
    }
    frames=try conv(activated(norm(frames,"encoder.norm_out")),"encoder.conv_out",48,3,1,2)
    frames=try conv(frames,"quant_conv",48,1)
    let result=concatenated(frames,axis:1);eval(result);return result
  }
  static func convolution(frames:[MLXArray],weight:MLXArray,bias:MLXArray,spatialPadding:Int,
    temporalPadding:Int,temporalStride:Int,spatialStride:Int) throws -> [MLXArray] {
    guard let first=frames.first,first.ndim == 5,first.shape[1] == 1,
      frames.allSatisfy({ $0.shape == first.shape && $0.dtype == .float32 }),weight.ndim == 5,
      weight.dtype == .float32,temporalStride > 0,spatialStride > 0,
      temporalPadding >= 0,temporalPadding < weight.shape[1] else {
      throw H3CheckpointError.invalid("Invalid bounded publisher causal convolution.")
    }
    let outputCount=(frames.count+temporalPadding-weight.shape[1])/temporalStride+1
    guard outputCount > 0 else { throw H3CheckpointError.invalid("Publisher temporal kernel exceeds its source.") }
    let zero=MLXArray.zeros(first.shape,dtype:.float32)
    var result:[MLXArray]=[]
    for output in 0..<outputCount {
      try Task.checkCancellation()
      let start=output*temporalStride-temporalPadding
      let input=concatenated((0..<weight.shape[1]).map { start+$0 < 0 ? zero:frames[start+$0] },axis:1)
      let value=try H3VideoVAEEncoderOps.causalConv(input,weight:weight,bias:bias,
        spatialPadding:spatialPadding,temporalPadding:0,stride:IntOrTriple((1,spatialStride,spatialStride)))
      eval(value);result.append(value)
    }
    return result
  }
  static func posterior(mean:MLXArray,logVariance:MLXArray,noise:MLXArray) throws -> MLXArray {
    guard mean.shape == logVariance.shape,mean.shape == noise.shape,
      mean.dtype == .float32,logVariance.dtype == .float32,noise.dtype == .float32,
      all(isFinite(mean)).item(Bool.self),all(isFinite(logVariance)).item(Bool.self),all(isFinite(noise)).item(Bool.self) else {
      throw H3CheckpointError.invalid("Publisher posterior requires finite Float32 moments and exact supplied noise.")
    }
    return mean+exp(Float(0.5)*MLX.clip(logVariance,min:-30,max:20))*noise
  }
}
