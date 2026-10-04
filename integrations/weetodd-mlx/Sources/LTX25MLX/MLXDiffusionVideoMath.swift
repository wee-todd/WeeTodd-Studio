import Foundation
import MLX
import LTX25Engine

/// Independently implemented DiffVAE operators. These are separate from LTX AV
/// attention: Q/K normalization is per head and RoPE rotates adjacent pairs.
enum MLXDiffusionVideoMath {
  static func rms(_ x:MLXArray,weight:MLXArray) -> MLXArray {
    MLXFast.rmsNorm(x,weight:weight,eps:0.000001)
  }
  static func silu(_ x:MLXArray) -> MLXArray { x*sigmoid(x) }
  static func rotarySplit(_ dimension:Int) -> [Int] {
    var time=(dimension/4)/2*2
    if ((dimension-time)/2)%2 != 0 { time-=2 }
    return [time,(dimension-time)/2,(dimension-time)/2]
  }
  static func rotary(_ x:MLXArray) throws -> MLXArray {
    guard x.ndim == 6,x.shape.last!>=8,x.shape.last!%2 == 0 else { throw LTXError.invalid("DiffVAE rotary requires BFHWHD layout with even head width.") }
    let split=rotarySplit(x.shape.last!),dtype=x.dtype
    var offset=0,pieces:[MLXArray]=[]
    for index in 0..<3 {
      let dimension=split[index],axis=index+1
      let part=x[.ellipsis,offset..<offset+dimension]
      let frequency=exp(-Float(log(10000.0))*MLXArray(stride(from:0,to:dimension,by:2)).asType(.float32)/Float(dimension))
      let angle=MLXArray(0..<x.shape[axis]).asType(.float32).expandedDimensions(axis:1)*frequency.expandedDimensions(axis:0)
      var shape=[Int](repeating:1,count:6);shape[axis]=x.shape[axis];shape[5]=dimension/2
      let cosine=cos(angle).reshaped(shape),sine=sin(angle).reshaped(shape)
      let paired=part.reshaped(Array(part.shape.dropLast())+[dimension/2,2]).asType(.float32)
      let even=paired[.ellipsis,0],odd=paired[.ellipsis,1]
      pieces.append(stacked([even*cosine-odd*sine,even*sine+odd*cosine],axis:-1).reshaped(part.shape).asType(dtype))
      offset+=dimension
    }
    return concatenated(pieces,axis:-1)
  }
  static func patch(_ pixels:MLXArray,patch:Int=4) throws -> MLXArray {
    guard pixels.ndim == 5,patch>0,pixels.shape[1] == 3,pixels.shape[3]%patch == 0,pixels.shape[4]%patch == 0 else {
      throw LTXError.invalid("DiffVAE noise requires B3FHW with divisible patch dimensions.")
    }
    let s=pixels.shape,h=s[3]/patch,w=s[4]/patch
    return pixels.reshaped([s[0],3,s[2],h,patch,w,patch]).transposed(0,2,3,5,1,6,4).reshaped([s[0],s[2],h,w,3*patch*patch])
  }
  static func unpatch(_ x:MLXArray,patch:Int=4) throws -> MLXArray {
    guard x.ndim == 5,patch>0,x.shape[4] == 3*patch*patch else { throw LTXError.invalid("DiffVAE output patch layout is invalid.") }
    let s=x.shape
    return x.reshaped([s[0],s[1],s[2],s[3],3,patch,patch]).transposed(0,4,1,2,6,3,5).reshaped([s[0],3,s[1],s[2]*patch,s[3]*patch])
  }
  static func upsample(_ projected:MLXArray,stride s:[Int],outputChannels:Int) throws -> MLXArray {
    guard projected.ndim == 5,s.count == 3,s.allSatisfy({ $0>0 }),outputChannels>0,
      projected.shape[4] == outputChannels*s.reduce(1,*) else { throw LTXError.invalid("DiffVAE shuffle dimensions do not match their projection.") }
    let a=projected.shape
    var x=projected.reshaped([a[0],a[1],a[2],a[3],outputChannels,s[0],s[1],s[2]])
      .transposed(0,1,5,2,6,3,7,4).reshaped([a[0],a[1]*s[0],a[2]*s[1],a[3]*s[2],outputChannels])
    if s[0] == 2 { x=x[0...,1...,0...,0...,0...] }
    return x
  }
  static func oneStep(noise:MLXArray,prediction:MLXArray) -> MLXArray {
    // Preserve low-precision subtraction boundaries, even at terminal t=1→0.
    noise-(noise-prediction)
  }
  static func neighborhoodIndices(shape:[Int],kernel:[Int],queries:Range<Int>) throws -> [Int32] {
    guard shape.count == 6,kernel.count == 3,kernel.allSatisfy({ $0>0 && $0%2 == 1 }),
      shape.allSatisfy({ $0>0 }),zip(Array(shape[1...3]),kernel).allSatisfy({ $0 >= $1 }) else {
      throw LTXError.invalid("Shifted DiffVAE neighborhoods require complete odd kernels.")
    }
    let frames=shape[1],height=shape[2],width=shape[3],rows=try MLXDiffusionVideoPlan.product([frames,height,width],limit:Int(Int32.max))
    guard queries.lowerBound>=0,queries.upperBound<=rows else { throw LTXError.invalid("DiffVAE query interval is outside its volume.") }
    let count=try MLXDiffusionVideoPlan.product([max(1,queries.count),kernel.reduce(1,*)],limit:16*1024*1024)
    if queries.isEmpty { return [] }
    var output:[Int32]=[];output.reserveCapacity(count)
    for query in queries {
      try Task.checkCancellation()
      let t=query/(height*width),h=query/width%height,w=query%width
      let t0=min(max(t-kernel[0]/2,0),frames-kernel[0]),h0=min(max(h-kernel[1]/2,0),height-kernel[1]),w0=min(max(w-kernel[2]/2,0),width-kernel[2])
      for dt in 0..<kernel[0] { for dh in 0..<kernel[1] { for dw in 0..<kernel[2] {
        output.append(Int32(((t0+dt)*height+h0+dh)*width+w0+dw))
      } } }
    }
    return output
  }
  static func referenceAttention(q:MLXArray,k:MLXArray,v:MLXArray,kernel:[Int],queryChunk:Int) throws -> MLXArray {
    guard q.shape == k.shape,q.shape == v.shape,q.ndim == 6,queryChunk>0,
      q.dtype.isFloatingPoint,k.dtype == q.dtype,v.dtype == q.dtype else { throw LTXError.invalid("DiffVAE Q/K/V layout or dtype differs.") }
    let s=q.shape,rows=try MLXDiffusionVideoPlan.product(Array(s[1...3]),limit:Int(Int32.max))
    let flatQ=q.reshaped([s[0],rows,s[4],s[5]]),flatK=k.reshaped([s[0],rows,s[4],s[5]]),flatV=v.reshaped([s[0],rows,s[4],s[5]])
    var pieces:[MLXArray]=[]
    for start in stride(from:0,to:rows,by:queryChunk) {
      try Task.checkCancellation()
      let stop=min(rows,start+queryChunk)
      let indices=MLXArray(try neighborhoodIndices(shape:s,kernel:kernel,queries:start..<stop),[stop-start,kernel.reduce(1,*)])
      let keys=take(flatK,indices,axis:1),values=take(flatV,indices,axis:1),queries=flatQ[0...,start..<stop,0...,0...]
      // Einsum preserves Python score dtype before FP32 softmax. Probabilities
      // return to the V dtype before the second contraction.
      let scores=einsum("bqhd,bqkhd->bqhk",queries,keys)*Float(1/sqrt(Double(s[5])))
      let probabilities=softmax(scores.asType(.float32),axis:-1).asType(values.dtype)
      let out=einsum("bqhk,bqkhd->bqhd",probabilities,values)
      eval(out);try Task.checkCancellation();pieces.append(out)
    }
    return concatenated(pieces,axis:1).reshaped(s)
  }
}
