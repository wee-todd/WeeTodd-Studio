import Foundation
import MLX
import TensorIO
import LTX25Audio
import LTX25Engine

public enum MLXAudioBackend:String,Sendable { case mps,mlx }

/// Float32 stereo VAE, BigVGAN and bandwidth extension. Activations stay on the
/// GPU between evaluated layers; weights are loaded and released one layer at a
/// time. No CPU im2col tiles or retained decoded model are used.
public final class MLXAudioDecoder {
  private let file:SafeTensorFile
  private let gate=NSLock()
  public let maximumLatentFrames:Int
  public let maximumResidentBytes:UInt64
  public private(set) var residentWeightBytes=0
  public init(checkpoint:URL,maximumLatentFrames:Int=501,maximumResidentBytes:UInt64=2*1024*1024*1024) throws {
    guard (1...1501).contains(maximumLatentFrames),(256*1024*1024...8*1024*1024*1024).contains(maximumResidentBytes) else {
      throw LTXError.invalid("Invalid MLX audio frame or workspace allowance.")
    }
    try AudioDecoder.validateCheckpoint(checkpoint)
    file=try SafeTensorFile(url:checkpoint)
    self.maximumLatentFrames=maximumLatentFrames;self.maximumResidentBytes=maximumResidentBytes
  }
  /// Eight largest activation copies plus 512 MiB for weight promotion,
  /// convolution scratch and filters. Admission estimate, not a process cap.
  public static func estimatedPeakBytes(latentFrames:Int) throws -> UInt64 {
    _ = try AudioDecoder.sampleCount(latentFrames:latentFrames)
    return UInt64(4*latentFrames)*64*256*4*8+512*1024*1024
  }
  private func read(_ name:String) throws -> MLXArray {
    let x=try MLXWeight.read(file,name).asType(.float32)
    eval(x);try MLXVideoDecoder.checkFinite(x)
    residentWeightBytes += x.nbytes
    return x
  }
  private func finish(_ x:MLXArray) throws -> MLXArray {
    eval(x);try Task.checkCancellation();try MLXVideoDecoder.checkFinite(x);return x
  }
  private func convolution(_ x:MLXArray,_ name:String,causal:Bool=false,dilation:Int=1) throws -> MLXArray {
    try Task.checkCancellation()
    return try autoreleasepool {
      defer { residentWeightBytes=0 }
      let w=try read(name+".weight")
      var y=MLXAudioMath.convolution(x,weight:w,causal:causal,dilation:dilation)
      if file.tensors[name+".bias"] != nil { y=y+(try read(name+".bias")) }
      return try finish(y)
    }
  }
  private func residual(_ x:MLXArray,_ name:String) throws -> MLXArray {
    let a=try convolution(MLXAudioMath.normalizedSiLU(x),name+".conv1.conv",causal:true)
    let b=try convolution(MLXAudioMath.normalizedSiLU(a),name+".conv2.conv",causal:true)
    let skip=try x.shape.last==b.shape.last ? x : convolution(x,name+".nin_shortcut.conv")
    return try finish(skip+b)
  }
  private func activation(_ x:MLXArray,_ name:String) throws -> MLXArray {
    try autoreleasepool {
      defer { residentWeightBytes=0 }
      let up=try read(name+".upsample.filter"),down=try read(name+".downsample.lowpass.filter")
      let alpha=exp(try read(name+".act.alpha")),beta=exp(try read(name+".act.beta"))
      var y=try finish(MLXAudioMath.upsampleFilter(x,filter:up,ratio:2,inputPad:5,cropLeft:15))
      let s=sin(y*alpha);y=try finish(y+s*s/(beta+1e-9))
      return try finish(MLXAudioMath.downsampleFilter(y,filter:down))
    }
  }
  private func decodeMel(_ latent:MLXArray) throws -> MLXArray {
    let p="audio_vae.decoder."
    var x=try autoreleasepool {
      defer { residentWeightBytes=0 }
      let mean=try read("audio_vae.per_channel_statistics.mean-of-means").reshaped([8,16]).T
      let std=try read("audio_vae.per_channel_statistics.std-of-means").reshaped([8,16]).T
      guard std.min().item(Float.self)>0 else { throw LTXError.invalid("Audio normalization standard deviation must be positive.") }
      return try finish(latent.transposed(1,2,0)*std+mean)
    }
    x=try convolution(x,p+"conv_in.conv",causal:true)
    x=try residual(x,p+"mid.block_1");x=try residual(x,p+"mid.block_2")
    for stage in [2,1,0] {
      for b in 0..<3 { x=try residual(x,p+"up.\(stage).block.\(b)") }
      if stage>0 {
        x=try convolution(MLXAudioMath.nearest2x(x),p+"up.\(stage).upsample.conv.conv",causal:true)
        x=try finish(x[1...])
      }
    }
    x=try convolution(MLXAudioMath.normalizedSiLU(x),p+"conv_out.conv",causal:true)
    return try finish(x.transposed(0,2,1).reshaped([x.shape[0],128]))
  }
  private func vocode(_ input:MLXArray,prefix:String,rates:[Int],clamp:Bool,progress:(String) throws -> Void) throws -> MLXArray {
    var x=try convolution(input,prefix+".conv_pre")
    for (stage,rate) in rates.enumerated() {
      x=try autoreleasepool {
        defer { residentWeightBytes=0 }
        let p=prefix+".ups.\(stage)",w=try read(p+".weight"),b=try read(p+".bias")
        return try finish(MLXAudioMath.transpose(x,weight:w,stride:rate,padding:(w.shape[2]-rate)/2)+b)
      }
      var sum:MLXArray?
      for j in 0..<3 {
        let p=prefix+".resblocks.\(stage*3+j)"
        var branch=x.reshaped(x.shape)
        for (layer,dilation) in [1,3,5].enumerated() {
          let a=try convolution(activation(branch,p+".acts1.\(layer)"),p+".convs1.\(layer)",dilation:dilation)
          let b=try convolution(activation(a,p+".acts2.\(layer)"),p+".convs2.\(layer)")
          branch=try finish(branch+b)
        }
        sum=try finish(sum.map { $0+branch } ?? branch)
      }
      x=try finish(sum!/Float(3));try progress(prefix+":stage\(stage+1)")
    }
    x=try convolution(activation(x,prefix+".act_post"),prefix+".conv_post")
    return try finish(clamp ? clip(x,min:-1,max:1) : x)
  }
  private func computeMel(_ wave:MLXArray) throws -> MLXArray {
    try autoreleasepool {
      defer { residentWeightBytes=0 }
      let basis=try read("vocoder.mel_stft.stft_fn.forward_basis").transposed(0,2,1)
      let mel=try read("vocoder.mel_stft.mel_basis")
      // Checkpoint's convolutional STFT uses causal zero padding, not a generic
      // centered/reflect STFT. Channels are independent batch items here.
      let input=padded(wave.T.expandedDimensions(axis:2),widths:[.init(0),.init((432,0)),.init(0)])
      let spectrum=conv1d(input,basis,stride:80)
      let re=spectrum[0...,0...,0..<257],im=spectrum[0...,0...,257...]
      let magnitude=sqrt(re*re+im*im)
      let result=log(maximum(matmul(magnitude,mel.T),1e-5))
      return try finish(result.transposed(1,0,2).reshaped([wave.shape[0]/80,128]))
    }
  }
  public func decode(latent:[Float],latentFrames:Int,maximumSamples:Int?=nil,
    progress:(String) throws -> Void = { _ in }) throws -> AudioWaveform {
    let full=try AudioDecoder.sampleCount(latentFrames:latentFrames)
    guard latentFrames<=maximumLatentFrames,latent.count==latentFrames*128,latent.allSatisfy(\.isFinite),
      maximumSamples==nil || (maximumSamples!>0 && maximumSamples!<=full),
      try Self.estimatedPeakBytes(latentFrames:latentFrames)<=maximumResidentBytes else {
      throw LTXError.invalid("Audio latent, duration or workspace is outside admission.")
    }
    guard gate.try() else { throw LTXError.invalid("Audio decoder is already evaluating.") }
    let oldCache=Memory.cacheLimit;Memory.cacheLimit=64*1024*1024
    defer { Stream.gpu.synchronize();residentWeightBytes=0;Memory.clearCache();Memory.cacheLimit=oldCache;gate.unlock() }
    func report(_ stage:String) throws { try Task.checkCancellation();try progress(stage);try Task.checkCancellation() }
    return try autoreleasepool {
      try report("start")
      let mel=try decodeMel(MLXArray(latent,[8,latentFrames,16]));try report("audio_vae")
      var wave=try vocode(mel,prefix:"vocoder.vocoder",rates:[5,2,2,2,2,2],clamp:true,progress:report)
      try report("vocoder")
      guard wave.shape[0]*3==full else { throw LTXError.invalid("Vocoder sample count disagrees with audio timing.") }
      if wave.shape[0]%80 != 0 { wave=try finish(padded(wave,widths:[.init((0,80-wave.shape[0]%80)),.init(0)])) }
      let residual=try vocode(computeMel(wave),prefix:"vocoder.bwe_generator",rates:[6,5,2,2,2],clamp:false,progress:report)
      let skip=try finish(MLXAudioMath.resample48k(wave))
      guard residual.shape==skip.shape else { throw LTXError.invalid("Audio bandwidth extension sample alignment mismatch.") }
      let count=maximumSamples ?? full
      let samples=try finish(clip((residual+skip)[0..<count].T,min:-1,max:1)).asArray(Float.self)
      try report("bandwidth_extension")
      return AudioWaveform(samples:samples,sampleRate:48000,channels:2)
    }
  }
}
