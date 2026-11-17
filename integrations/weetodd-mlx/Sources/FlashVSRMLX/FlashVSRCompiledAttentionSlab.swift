import Foundation
import MLX

/// One original rank-two attention slab, with no model tensors retained.
final class FlashVSRCompiledAttentionSlab {
  static func supports(queries:MLXArray,keys:MLXArray,values:MLXArray,keySlabBlocks:Int,
    architecture:String?)->Bool {
    architecture == "applegpu_g15d" && keySlabBlocks == 8
      && queries.ndim == 5 && keys.ndim == 5 && values.shape == keys.shape
      && queries.shape.allSatisfy{$0 > 0} && keys.shape.allSatisfy{$0 > 0}
      && queries.shape[0..<2] == keys.shape[0..<2]
      && queries.shape[3...] == [128,128] && keys.shape[3...] == [128,128]
      && queries.dtype == keys.dtype && queries.dtype == values.dtype
      && (queries.dtype == .float32 || queries.dtype == .bfloat16)
  }
  // The pinned MLX compiler prints embedded Float constants with seven
  // significant digits. Keep the original scalar as a dynamic graph input.
  private let scale=MLXArray(sqrt(Float(128)))
  // Captures only the graph template. All activations and selected IDs are
  // explicit inputs; releasing this owner erases its MLX compile-cache entries.
  private let function: ([MLXArray]) -> [MLXArray] = compile { (a: [MLXArray]) in
    let query=a[0],key=a[1],value=a[2],indices=a[3]
    let peak=a[4],denominator=a[5],numerator=a[6]
    let k=take(key,indices,axis:0).reshaped([-1,128]).asType(.float32)
    let v=take(value,indices,axis:0).reshaped([-1,128]).asType(.float32)
    let scores=matmul(query,k.transposed())/a[7]
    let nextPeak=maximum(peak,max(scores,axis:1,keepDims:true))
    let correction=exp(peak-nextPeak),probabilities=exp(scores-nextPeak)
    return [numerator*correction+matmul(probabilities,v),
      denominator*correction+sum(probabilities,axis:1,keepDims:true),nextPeak]
  }

  func evaluate(query: MLXArray, key: MLXArray, value: MLXArray, indices: MLXArray,
    peak: MLXArray, denominator: MLXArray, numerator: MLXArray) throws -> [MLXArray] {
    // The enclosing consumer validates selected IDs before constructing slabs.
    guard query.shape == [128,128],query.dtype == .float32,
      key.ndim == 3,key.dim(0) > 0,key.shape[1...] == [128,128],
      value.shape == key.shape,value.dtype == key.dtype,
      (key.dtype == .float32 || key.dtype == .bfloat16),
      indices.ndim == 1,(1...8).contains(indices.size),indices.dtype == .int32,
      peak.shape == [128,1],denominator.shape == [128,1],numerator.shape == [128,128],
      peak.dtype == .float32,denominator.dtype == .float32,numerator.dtype == .float32 else {
      throw FlashVSRError.invalid("Invalid compiled attention slab geometry.")
    }
    let output=function([query,key,value,indices,peak,denominator,numerator,scale])
    guard output.count == 3 else { throw FlashVSRError.invalid("Compiled attention slab failed.") }
    return output
  }
}
