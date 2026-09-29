import Foundation
import MLX

/// Channels-last audio operators. Batch is introduced only at convolution
/// boundaries; independent channels become batches for shared FIR filtering.
enum MLXAudioMath {
  static func convolution(_ x:MLXArray,weight:MLXArray,causal:Bool=false,dilation:Int=1) -> MLXArray {
    if x.ndim==2 {
      return conv1d(x.expandedDimensions(axis:0),weight.transposed(0,2,1),
        padding:(weight.shape[2]-1)*dilation/2,dilation:dilation)[0]
    }
    let kh=weight.shape[2],kw=weight.shape[3]
    let top=causal ? kh-1 : kh/2,bottom=causal ? 0 : kh/2
    let input=padded(x.expandedDimensions(axis:0),widths:[.init(0),.init((top,bottom)),.init(kw/2),.init(0)])
    return conv2d(input,weight.transposed(0,2,3,1))[0]
  }
  static func transpose(_ x:MLXArray,weight:MLXArray,stride:Int,padding:Int) -> MLXArray {
    convTransposed1d(x.expandedDimensions(axis:0),weight.transposed(1,2,0),stride:stride,padding:padding)[0]
  }
  static func normalizedSiLU(_ x:MLXArray) -> MLXArray {
    let y=MLXFast.rmsNorm(x,weight:.ones([x.shape.last!]),eps:1e-6)
    return y*sigmoid(y)
  }
  static func nearest2x(_ x:MLXArray) -> MLXArray {
    let a=take(x,MLXArray((0..<x.shape[0]*2).map { Int32($0/2) }),axis:0)
    return take(a,MLXArray((0..<x.shape[1]*2).map { Int32($0/2) }),axis:1)
  }
  static func upsampleFilter(_ x:MLXArray,filter:MLXArray,ratio:Int,inputPad:Int,cropLeft:Int) -> MLXArray {
    let input=padded(x.T.expandedDimensions(axis:2),widths:[.init(0),.init(inputPad),.init(0)],mode:.edge)
    let output=convTransposed1d(input,filter.reshaped([1,filter.size,1]),stride:ratio)*Float(ratio)
    return output[0...,cropLeft..<cropLeft+x.shape[0]*ratio,0].T
  }
  static func downsampleFilter(_ x:MLXArray,filter:MLXArray) -> MLXArray {
    let input=padded(x.T.expandedDimensions(axis:2),widths:[.init(0),.init(((filter.size-1)/2,filter.size/2)),.init(0)],mode:.edge)
    return conv1d(input,filter.reshaped([1,filter.size,1]),stride:2)[0...,0..<x.shape[0]/2,0].T
  }
  static func resample48k(_ x:MLXArray) -> MLXArray {
    let filter:[Float]=(0..<43).map { i in
      let t=(Double(i)/3-7)*0.99
      let window=pow(cos(min(6,max(-6,t))*Double.pi/12),2)
      return Float((t==0 ? 1 : sin(Double.pi*t)/(Double.pi*t))*window*0.99/3)
    }
    return upsampleFilter(x,filter:MLXArray(filter),ratio:3,inputPad:7,cropLeft:42)
  }
}
