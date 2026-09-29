import LTX25Engine
import LTX25Video

/// Conservative allocation estimate for the chosen depth-window algorithm.
/// Includes visible temporaries and the bounded MLX 0.32.2 convolution workspace;
/// driver allocations and allocator caches are outside this admission estimate.
public struct MLXVideoDecodePlan:Sendable {
  public let outputShape:[Int]
  public let admittedActivationBytes:Int
  public let largestWindowBytes:Int
  public init(shape:[Int],configuration c:VideoDecodeConfiguration,precision:MLXVideoPrecision = .bfloat16) throws {
    // Reuse geometry/index validation, but not the MPS-specific workspace bound.
    var geometryConfig=c;geometryConfig.maximumActivationBytes=Int.max
    let geometry=try VideoDecodePlan(shape:shape,configuration:geometryConfig)
    guard c.maximumActivationBytes>0 else { throw LTXError.invalid("Video workspace must be positive.") }
    var t=shape[2],h=shape[3],w=shape[4],peak=0,windowPeak=0,scratchPeak=0
    func layer(_ ic:Int,_ oc:Int) {
      let plane=h*w*max(ic,oc),n=min(t,max(1,precision.windowElements/plane))
      let window=n*plane*precision.bytes
      peak=max(peak,t*plane);windowPeak=max(windowPeak,window)
      if precision == .bfloat16 {
        // Bounded normalized halo, evaluated slices/accumulators, input layout
        // copies and the core's scoped Winograd budget. The full-output chunks
        // plus concatenation are accounted separately below.
        let halo=2*plane*precision.bytes
        scratchPeak=max(scratchPeak,8*window+3*halo+1024*1024*1024+oc*ic*27*precision.bytes)
        return
      }
      // Gather, RMS/SiLU, contribution, accumulator, bias and contiguous-window
      // copies. One spatial frame may exceed the nominal 8M-element window.
      var scratch=8*window+oc*ic*9*4
      // Pinned Metal conv2d selects F(6x6,3x3) Winograd at this boundary.
      if n*h*w>=4096,ic%32==0,oc%32==0,ic+oc>=256 {
        let th=(h+5)/6,tw=(w+5)/6
        scratch += (n*(th*6+2)*(tw*6+2)*ic+64*n*th*tw*(ic+oc)+64*ic*oc)*4
      }
      // Source dtype promotion can overlap its evaluated Float32 layer. Reserve
      // a full layer even for BF16 files; base layer weights have a separate cap.
      scratch += oc*ic*27*4
      scratchPeak=max(scratchPeak,scratch)
    }
    layer(128,1024)
    let channels=[1024,512,512,256,128],outputs=[4096,4096,512,512],scales=[(2,2),(2,2),(1,2),(2,1)]
    for stage in 0..<5 {
      layer(channels[stage],channels[stage])
      if stage<4 {
        layer(channels[stage],outputs[stage])
        let (s,d)=scales[stage];t=t*d-(d>1 ? 1 : 0);h *= s;w *= s
      }
    }
    layer(128,48)
    // Residual, input, output and a conservative one-byte-per-element reserve.
    // Finite checks now use bounded slices. Other operations (shuffle/add) fit
    // inside the same three-activation envelope.
    // BF16 GPU assembly may overlap residual + input + chunks + concatenation.
    let bytes=peak*(precision == .bfloat16 ? 8 : 13)+scratchPeak
    guard bytes<=c.maximumActivationBytes else {
      throw LTXError.invalid("MLX video decode needs an estimated \(bytes) activation/workspace bytes; raise its allowance or reduce geometry.")
    }
    outputShape=geometry.outputShape;admittedActivationBytes=bytes;largestWindowBytes=windowPeak
  }
}
