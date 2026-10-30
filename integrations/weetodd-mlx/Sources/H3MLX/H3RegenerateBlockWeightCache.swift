import MLX

/// At most the four merged projections of one block, bounded to 1 GiB.
/// Repeated token/query windows reuse the exact BF16 merge. Crossing a block
/// boundary drops every old projection; stage teardown clears the final block.
final class H3RegenerateBlockWeightCache {
  private var owner:Int?
  private var weights:[String:MLXArray]=[:]
  var count:Int { weights.count }
  func clear() { weights.removeAll();owner=nil }
  func weight(block:Int,target:String,shape:[Int],dtype:DType,
    create:() throws -> MLXArray) throws -> MLXArray {
    do {
      try Task.checkCancellation()
      guard (0..<52).contains(block),shape.count == 2,
        shape.allSatisfy({ (1...30000).contains($0) }),dtype == .bfloat16 else {
        throw H3CheckpointError.invalid("Invalid regenerate block-cache shape or precision.")
      }
      if owner != block { clear();owner=block }
      if let existing=weights[target] {
        guard existing.shape == shape,existing.dtype == dtype else {
          throw H3CheckpointError.invalid("Regenerate cached projection changed shape or precision.")
        }
        return existing
      }
      let bytes=weights.values.reduce(0) { $0+$1.size*2 }+shape[0]*shape[1]*2
      guard weights.count < 4,bytes <= 1024*1024*1024 else {
        throw H3CheckpointError.invalid("Regenerate merged weights exceed the one-block cache bound.")
      }
      let result=try create()
      guard result.shape == shape,result.dtype == dtype else {
        throw H3CheckpointError.invalid("Regenerate merge returned incorrect cache geometry.")
      }
      eval(result);try Task.checkCancellation();weights[target]=result
      return result
    } catch { clear();throw error }
  }
}
