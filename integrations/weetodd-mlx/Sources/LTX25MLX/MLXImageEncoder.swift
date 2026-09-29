import Foundation
import MLX
import TensorIO
import LTX25Engine

/// Whole-image admission, including residual/normalization overlap and one
/// convolution's packed/Float32/folded weights. Driver memory is additional.
public struct MLXImageEncodePlan:Sendable {
  public let latentShape:[Int]
  public let ownedBufferBytes:Int
  public init(width:Int,height:Int,maximumOwnedBufferBytes:Int=1024*1024*1024) throws {
    guard (32...4096).contains(width), (32...4096).contains(height),
      width%32 == 0, height%32 == 0, maximumOwnedBufferBytes > 0 else {
      throw LTXError.invalid("Reference encoding needs bounded dimensions divisible by32.")
    }
    ownedBufferBytes=width*height/16*128*4*8+384*1024*1024
    guard ownedBufferBytes <= maximumOwnedBufferBytes else { throw LTXError.invalid("Reference encoder exceeds its owned-buffer budget.") }
    latentShape=[height/32*width/32,128]
  }
}

/// Released convolutional VAE image encoder, batch one, RGB [-1,1]. All causal
/// temporal samples are identical for a single image: summing the three depth
/// slices gives an equivalent 2D kernel. Temporal packing still duplicates each
/// channel in its trained order. This specialization does not admit video inputs.
public final class MLXImageEncoder {
  private let file:SafeTensorFile
  private let gate=NSLock()
  public private(set) var residentWeightBytes=0
  private struct Layer { let name:String; let input:Int; let output:Int }
  private static var layers:[Layer] {
    var result=[Layer(name:"encoder.conv_in.conv",input:48,output:128)]
    let channels=[128,256,512,1024,1024],blocks=[4,6,4,2,2],downOutputs=[64,256,128,128]
    for stage in 0..<5 {
      for block in 0..<blocks[stage] { for conv in 1...2 {
        result.append(Layer(name:"encoder.down_blocks.\(stage*2).res_blocks.\(block).conv\(conv).conv",input:channels[stage],output:channels[stage]))
      } }
      if stage<4 { result.append(Layer(name:"encoder.down_blocks.\(stage*2+1).conv.conv",input:channels[stage],output:downOutputs[stage])) }
    }
    result.append(Layer(name:"encoder.conv_out.conv",input:1024,output:129))
    return result
  }
  public init(checkpoint:URL) throws {
    file=try SafeTensorFile(url:checkpoint)
    guard file.metadata["model_version"] == "2.5.0",
      let data=file.metadata["config"]?.data(using:.utf8),
      let config=try JSONSerialization.jsonObject(with:data) as? [String:Any],
      let vae=config["vae"] as? [String:Any],
      vae["_class_name"] as? String == "CausalVideoAutoencoder",
      vae["spatial_padding_mode"] as? String == "zeros",
      vae["norm_layer"] as? String == "pixel_norm",vae["patch_size"] as? Int == 4,
      vae["encoder_base_channels"] as? Int == 128,vae["latent_channels"] as? Int == 128,
      vae["latent_log_var"] as? String == "uniform",vae["use_quant_conv"] as? Bool == false else {
      throw LTXError.invalid("Expected the released LTX2.5 convolutional image encoder architecture.")
    }
    var expected:[String:[Int]]=[:]
    for layer in Self.layers {
      expected[layer.name+".weight"]=[layer.output,layer.input,3,3,3]
      expected[layer.name+".bias"]=[layer.output]
    }
    for name in ["mean-of-means","std-of-means"] { expected["per_channel_statistics."+name]=[128] }
    let encoderKeys=Set(file.tensors.keys.filter { $0.hasPrefix("encoder.") })
    guard encoderKeys == Set(expected.keys.filter { $0.hasPrefix("encoder.") }) else {
      throw LTXError.invalid("Missing or unsupported image encoder tensors.")
    }
    for (name,shape) in expected {
      guard let descriptor=file.tensors[name],descriptor.shape == shape.map(UInt64.init),
        ["F32","BF16"].contains(descriptor.dtype) else { throw LTXError.invalid("Image encoder tensor mismatch: \(name)") }
    }
  }
  public func encode(_ pixels:MLXArray,maximumOwnedBufferBytes:Int=1024*1024*1024,
    progress:(Int) throws -> Void = { _ in }) throws -> MLXArray {
    guard gate.try() else { throw LTXError.invalid("Image encoder is already running.") }
    defer { gate.unlock() }
    guard pixels.ndim == 4,pixels.shape[0] == 1,pixels.shape[3] == 3,pixels.dtype == .float32 else {
      throw LTXError.invalid("Image encoder requires one Float32 NHWC RGB image.")
    }
    let plan=try MLXImageEncodePlan(width:pixels.shape[2],height:pixels.shape[1],maximumOwnedBufferBytes:maximumOwnedBufferBytes)
    guard MLX.isFinite(pixels).all().item(Bool.self),pixels.min().item(Float.self) >= -1,
      pixels.max().item(Float.self) <= 1 else { throw LTXError.invalid("Reference pixels must be finite RGB in[-1,1].") }
    try Task.checkCancellation()
    let oldLimit=Memory.cacheLimit;Memory.cacheLimit=128*1024*1024
    defer { Stream.gpu.synchronize();residentWeightBytes=0;Memory.clearCache();Memory.cacheLimit=oldLimit }
    var completed=0
    func finish(_ value:MLXArray) throws -> MLXArray {
      eval(value);try Task.checkCancellation()
      guard MLX.isFinite(value).all().item(Bool.self) else { throw LTXError.invalid("Nonfinite image encoder activation.") }
      return value
    }
    func convolution(_ input:MLXArray,_ name:String,normalized:Bool=false) throws -> MLXArray {
      try Task.checkCancellation()
      let result=try autoreleasepool {
        let weight=try MLXWeight.read(file,name+".weight").asType(.float32)
        let bias=try MLXWeight.read(file,name+".bias").asType(.float32)
        residentWeightBytes=weight.nbytes+bias.nbytes
        defer { residentWeightBytes=0 }
        guard MLX.isFinite(weight).all().item(Bool.self),MLX.isFinite(bias).all().item(Bool.self) else {
          throw LTXError.invalid("Nonfinite image encoder weights.")
        }
        let x:MLXArray
        if normalized {
          let norm=MLXFast.rmsNorm(input,weight:.ones([input.shape[3]]),eps:1e-8)
          x=norm*sigmoid(norm)
        } else { x=input }
        return try finish(Self.convolve(x,weight:weight,bias:bias))
      }
      if Memory.cacheMemory>128*1024*1024 { Memory.clearCache() }
      completed += 1;try progress(completed);try Task.checkCancellation()
      return result
    }
    var x=try finish(Self.patch(pixels.reshaped(pixels.shape)))
    x=try convolution(x,"encoder.conv_in.conv")
    let blocks=[4,6,4,2,2],channels=[256,512,1024,1024],spatial=[2,1,2,2],temporal=[1,2,2,2]
    for stage in 0..<5 {
      for block in 0..<blocks[stage] {
        let residual=x.reshaped(x.shape),prefix="encoder.down_blocks.\(stage*2).res_blocks.\(block)"
        x=try convolution(x,prefix+".conv1.conv",normalized:true)
        x=try convolution(x,prefix+".conv2.conv",normalized:true)
        x=try finish(x+residual)
      }
      if stage<4 {
        let skip=Self.downpack(x,spatial:spatial[stage],temporal:temporal[stage])
        let group=skip.shape[3]/channels[stage]
        let residual=try finish(skip.reshaped([1,skip.shape[1],skip.shape[2],channels[stage],group]).mean(axis:-1))
        let projected=try convolution(x,"encoder.down_blocks.\(stage*2+1).conv.conv")
        x=try finish(Self.downpack(projected,spatial:spatial[stage],temporal:temporal[stage])+residual)
      }
    }
    x=try convolution(x,"encoder.conv_out.conv",normalized:true)
    let mean=try MLXWeight.read(file,"per_channel_statistics.mean-of-means").asType(.float32)
    let std=try MLXWeight.read(file,"per_channel_statistics.std-of-means").asType(.float32)
    guard MLX.isFinite(mean).all().item(Bool.self),MLX.isFinite(std).all().item(Bool.self),std.min().item(Float.self)>0 else {
      throw LTXError.invalid("Invalid image encoder channel statistics.")
    }
    return try finish((x[0...,0...,0...,0..<128].reshaped(plan.latentShape)-mean)/std)
  }
  static func patch(_ x:MLXArray) -> MLXArray {
    let h=x.shape[1],w=x.shape[2]
    return x.reshaped([1,h/4,4,w/4,4,3]).transposed(0,1,3,5,4,2).reshaped([1,h/4,w/4,48])
  }
  static func downpack(_ x:MLXArray,spatial s:Int,temporal t:Int) -> MLXArray {
    let h=x.shape[1],w=x.shape[2],c=x.shape[3]
    let split=x.reshaped([1,h/s,s,w/s,s,c]).transposed(0,1,3,5,2,4).expandedDimensions(axis:4)
    return repeated(split,count:t,axis:4).reshaped([1,h/s,w/s,c*t*s*s])
  }
  static func convolve(_ x:MLXArray,weight:MLXArray,bias:MLXArray) -> MLXArray {
    let folded=weight.sum(axis:2).transposed(0,2,3,1)
    return conv2d(x,folded,padding:1)+bias
  }
}
