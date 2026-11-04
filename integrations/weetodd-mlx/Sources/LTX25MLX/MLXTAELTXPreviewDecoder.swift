import Foundation
import MLX
import TensorIO
import LTX25Engine

public enum MLXTAELTXPreviewVariant:String,Sendable {
  case standard = "taeltx2_3"
  case wide = "taeltx2_3_wide"
}

public struct MLXTAELTXPreviewFrame:Sendable {
  public let rgb:Data
  public let width:Int
  public let height:Int
  /// RGB timeline after the seven-frame causal warmup is removed.
  public let frameIndex:Int
}

/// Display-only approximation of normalized LTX latents. This is deliberately
/// separate from the final video VAE and never denormalizes its input.
///
/// Independently implemented from the published TAELTX2.3 operator/checkpoint
/// contracts, Ollin Boer Bohan, MIT, revision
/// 32ac0146b11007cda5a57b60a3b35653361fb8a4 (madebyollin/taehv).
/// The model's architecture is identified by exact tensor headers, never its
/// filename. Only one convolution's copied parameters are held at a time.
public final class MLXTAELTXPreviewDecoder {
  public let variant:MLXTAELTXPreviewVariant
  public private(set) var residentWeightBytes=0
  public private(set) var lastDecodeSeconds:Double=0
  private let file:SafeTensorFile
  private let checkpoint:URL
  private let gate=NSLock()
  private static let executionGate=NSLock()

  /// Header-only admission: no MLX arrays or model parameters are allocated.
  public init(checkpoint:URL) throws {
    let file=try SafeTensorFile(url:checkpoint,maximumHeaderBytes:256*1024)
    guard file.fileByteCount<=128*1024*1024 else {
      throw LTXError.invalid("TAELTX preview checkpoint exceeds its 128 MiB file allowance.")
    }
    let variant:MLXTAELTXPreviewVariant
    switch file.tensors["decoder.1.weight"]?.shape {
    case [256,128,3,3]: variant = .standard
    case [1024,128,3,3]: variant = .wide
    default: throw LTXError.invalid("Unrecognized TAELTX2.3 preview architecture.")
    }
    let expected=Self.parameterShapes(variant)
    guard Set(expected.keys)==Set(file.tensors.keys) else {
      throw LTXError.invalid("TAELTX preview checkpoint has missing or unexpected parameters.")
    }
    for (name,shape) in expected {
      guard let record=file.tensors[name],record.dtype=="F16",record.shape==shape.map(UInt64.init) else {
        throw LTXError.invalid("TAELTX preview parameter has an unsupported shape or dtype: \(name)")
      }
    }
    try file.checkUnchanged(at:checkpoint)
    self.file=file;self.checkpoint=checkpoint;self.variant=variant
  }

  public func decode(latent:MLXArray,maximumEdge:Int=384,maximumLatentFrames:Int=3) throws -> MLXTAELTXPreviewFrame {
    let plan=try MLXTAELTXPreviewPlan(shape:latent.shape,maximumEdge:maximumEdge,maximumLatentFrames:maximumLatentFrames)
    guard latent.dtype == .float32 else {
      throw LTXError.invalid("TAELTX preview requires normalized Float32 BCFHW latents with 128 channels and batch one.")
    }
    try Task.checkCancellation()
    try file.checkUnchanged(at:checkpoint)
    guard gate.try() else { throw LTXError.invalid("TAELTX preview decoder is already running.") }
    defer { gate.unlock() }
    guard Self.executionGate.try() else { throw LTXError.invalid("Another TAELTX preview decoder is running.") }
    defer { Self.executionGate.unlock() }
    let started=Date()
    // Bound Winograd scratch as well as the explicitly admitted activations.
    // This setting is process-global; native jobs execute weighted stages
    // serially and preview instances serialize through executionGate.
    let key="MLX_CONV_WINOGRAD_WORKING_SET",oldWorkspace=getenv("MLX_CONV_WINOGRAD_WORKING_SET").map { String(cString:$0) }
    setenv(key,"268435456",1)
    defer {
      Stream.gpu.synchronize()
      residentWeightBytes=0
      Memory.clearCache()
      if let oldWorkspace { setenv(key,oldWorkspace,1) } else { unsetenv(key) }
      lastDecodeSeconds=Date().timeIntervalSince(started)
    }
    func finish(_ value:MLXArray) throws -> MLXArray {
      eval(value)
      try Task.checkCancellation()
      try MLXVideoDecoder.checkFinite(value)
      return value
    }
    func conv(_ value:MLXArray,_ name:String,groups:Int=1) throws -> MLXArray {
      try Task.checkCancellation()
      try file.checkUnchanged(at:checkpoint)
      return try autoreleasepool {
        let weight=try MLXWeight.read(file,name+".weight").asType(.float32)
        let bias=try file.tensors[name+".bias"].map { _ in try MLXWeight.read(file,name+".bias").asType(.float32) }
        eval([weight]+(bias.map { [$0] } ?? []))
        residentWeightBytes=weight.nbytes+(bias?.nbytes ?? 0)
        defer { residentWeightBytes=0 }
        try file.checkUnchanged(at:checkpoint)
        try MLXVideoDecoder.checkFinite(weight)
        if let bias { try MLXVideoDecoder.checkFinite(bias) }
        let result=try finish(Self.convolve(value,weight:weight,bias:bias,groups:groups))
        try file.checkUnchanged(at:checkpoint)
        return result
      }
    }
    // Slice wrappers and functional operators preserve the caller's arrays.
    // Scan the original tensor with bounded scratch, including frames outside
    // the displayed prefix, before performing any weighted operation.
    try MLXVideoDecoder.checkFinite(latent)
    var x=latent[0,0...,0..<plan.latentFrames,0...,0...].transposed(1,2,3,0)
    x=try finish(Self.resizeLatent(x,height:plan.latentHeight,width:plan.latentWidth))
    x=try finish(tanh(x/Float(3))*Float(3))
    x=try finish(maximum(try conv(x,"decoder.1"),Float(0)))
    for stage in 0..<3 {
      let start=3+stage*6
      for block in start..<(start+3) {
        let residual=x.reshaped(x.shape)
        let memory=Self.causalHistory(x)
        var y=concatenated([x,memory],axis:3)
        let prefix="decoder.\(block).conv."
        if variant == .wide {
          let groups=x.shape[3]/64
          y=try finish(maximum(try conv(y,prefix+"0"),Float(0)))
          y=try finish(maximum(try conv(y,prefix+"2",groups:groups),Float(0)))
          y=try finish(maximum(try conv(y,prefix+"4"),Float(0)))
          y=try conv(y,prefix+"6",groups:groups)
        } else {
          y=try finish(maximum(try conv(y,prefix+"0"),Float(0)))
          y=try finish(maximum(try conv(y,prefix+"2"),Float(0)))
          y=try conv(y,prefix+"4")
        }
        x=try finish(maximum(y+residual,Float(0)))
      }
      // No memory block follows the last temporal expansion. All remaining
      // operators are framewise, so the last input and its second child are
      // sufficient. Prune only here: earlier pruning would lose causal state.
      if stage==2 { x=x[(x.shape[0]-1)..<x.shape[0]] }
      x=try finish(repeated(repeated(x,count:2,axis:1),count:2,axis:2))
      x=try finish(Self.temporalExpand(try conv(x,"decoder.\(start+4).conv")))
      if stage==2 { x=x[(x.shape[0]-1)..<x.shape[0]] }
      x=try conv(x,"decoder.\(start+5)")
    }
    x=try conv(try finish(maximum(x,Float(0))),"decoder.22")
    // Only the last causal frame is needed. Pixel shuffle after selecting it
    // avoids allocating a full RGB preview volume.
    let image=try finish(Self.pixelShuffle(x[(x.shape[0]-1)..<x.shape[0]]))[0]
    guard image.shape == [plan.latentHeight*32,plan.latentWidth*32,3] else {
      throw LTXError.invalid("TAELTX preview output disagrees with admitted geometry.")
    }
    let rgb=Self.rgb8(image)
    try Task.checkCancellation()
    try file.checkUnchanged(at:checkpoint)
    return MLXTAELTXPreviewFrame(rgb:rgb,width:image.shape[1],height:image.shape[0],frameIndex:plan.frameIndex)
  }

  /// Checkpoint kernels are OIHW; MLX convolution consumes OHWI/NHWC.
  static func convolve(_ x:MLXArray,weight:MLXArray,bias:MLXArray?,groups:Int=1) throws -> MLXArray {
    guard x.ndim==4,weight.ndim==4,groups>0,
      x.shape[3]==weight.shape[1]*groups,weight.shape[0]%groups==0,
      weight.shape[2]==weight.shape[3],[1,3].contains(weight.shape[2]),
      bias == nil || bias!.shape == [weight.shape[0]] else {
      throw LTXError.invalid("Invalid TAELTX grouped convolution layout.")
    }
    let result=conv2d(x,weight.transposed(0,2,3,1),padding:.init(weight.shape[2]/2),groups:groups)
    return bias.map { result+$0 } ?? result
  }

  static func causalHistory(_ x:MLXArray) -> MLXArray {
    let empty=MLXArray.zeros([1,x.shape[1],x.shape[2],x.shape[3]],dtype:x.dtype)
    return x.shape[0]==1 ? empty : concatenated([empty,x[0..<(x.shape[0]-1)]],axis:0)
  }
  static func temporalExpand(_ x:MLXArray) -> MLXArray {
    let t=x.shape[0],h=x.shape[1],w=x.shape[2],c=x.shape[3]/2
    return x.reshaped([t,h,w,2,c]).transposed(0,3,1,2,4).reshaped([t*2,h,w,c])
  }
  static func pixelShuffle(_ x:MLXArray) -> MLXArray {
    let t=x.shape[0],h=x.shape[1],w=x.shape[2]
    return x.reshaped([t,h,w,3,4,4]).transposed(0,1,4,2,5,3).reshaped([t,h*4,w*4,3])
  }
  static func rgb8(_ image:MLXArray) -> Data {
    Data(round(clip(image,min:0,max:1)*Float(255)).asType(.uint8).asArray(UInt8.self))
  }
  private static func resizeLatent(_ x:MLXArray,height:Int,width:Int) -> MLXArray {
    guard height != x.shape[1] || width != x.shape[2] else { return x.reshaped(x.shape) }
    let rows=MLXArray((0..<height).map { Int32($0*x.shape[1]/height) })
    let columns=MLXArray((0..<width).map { Int32($0*x.shape[2]/width) })
    return take(take(x,rows,axis:1),columns,axis:2)
  }

  /// Exact published F16 checkpoint contract, including the unused encoder.
  private static func parameterShapes(_ variant:MLXTAELTXPreviewVariant) -> [String:[Int]] {
    var shapes:[String:[Int]]=[:]
    func conv(_ name:String,_ output:Int,_ input:Int,_ kernel:Int=3,_ bias:Bool=true) {
      shapes[name+".weight"]=[output,input,kernel,kernel]
      if bias { shapes[name+".bias"]=[output] }
    }
    conv("encoder.0",64,48)
    for start in [2,7,12] {
      conv("encoder.\(start).conv",64,128,1,false)
      conv("encoder.\(start+1)",64,64,3,false)
      for block in (start+2)...(start+4) {
        for layer in [0,2,4] { conv("encoder.\(block).conv.\(layer)",64,layer==0 ? 128 : 64) }
      }
    }
    conv("encoder.17",128,64)
    let wide=variant == .wide,channels=wide ? [1024,512,256,64] : [256,128,64,64]
    conv("decoder.1",channels[0],128)
    for stage in 0..<3 {
      let c=channels[stage],start=3+stage*6
      for block in start..<(start+3) {
        conv("decoder.\(block).conv.0",c,2*c,wide ? 1 : 3)
        conv("decoder.\(block).conv.2",c,wide ? 64 : c)
        conv("decoder.\(block).conv.4",c,c,wide ? 1 : 3)
        if wide { conv("decoder.\(block).conv.6",c,64) }
      }
      conv("decoder.\(start+4).conv",2*c,c,1,false)
      conv("decoder.\(start+5)",channels[stage+1],c,3,false)
    }
    conv("decoder.22",48,64)
    return shapes
  }
}

/// Pure admission precedes finite scans, copying, weight reads and convolution.
struct MLXTAELTXPreviewPlan {
  let latentFrames:Int
  let latentHeight:Int
  let latentWidth:Int
  let frameIndex:Int
  init(shape:[Int],maximumEdge:Int,maximumLatentFrames:Int) throws {
    guard shape.count==5,shape[0]==1,shape[1]==128,
      shape.allSatisfy({ (1...65536).contains($0) }),
      (32...384).contains(maximumEdge),(1...3).contains(maximumLatentFrames) else {
      throw LTXError.invalid("TAELTX preview requires bounded batch-one BCFHW/128 latents, a 32–384 pixel edge and one to three latent frames.")
    }
    var elements=1
    for dimension in shape {
      let product=elements.multipliedReportingOverflow(by:dimension)
      guard !product.overflow,product.partialValue<=Int(Int32.max) else {
        throw LTXError.invalid("TAELTX source latent geometry exceeds its indexing allowance.")
      }
      elements=product.partialValue
    }
    latentFrames=min(shape[2],maximumLatentFrames)
    let grid=maximumEdge/32,sourceEdge=max(shape[3],shape[4])
    latentHeight=sourceEdge<=grid ? shape[3] : max(1,(shape[3]*grid+sourceEdge/2)/sourceEdge)
    latentWidth=sourceEdge<=grid ? shape[4] : max(1,(shape[4]*grid+sourceEdge/2)/sourceEdge)
    frameIndex=(latentFrames-1)*8
    // Wide's second temporal projection is the largest feature map:
    // 6×48×48×1024 at the default bound. The final framewise stage is
    // pruned above, preserving its exact last frame with a smaller allocation.
    let largest=latentFrames*2*latentHeight*4*latentWidth*4*1024*4
    guard largest<=64*1024*1024 else {
      throw LTXError.invalid("TAELTX preview activation exceeds its 64 MiB tensor allowance.")
    }
  }
}
