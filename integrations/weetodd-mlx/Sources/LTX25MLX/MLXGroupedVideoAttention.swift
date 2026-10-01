import Foundation
import MLX
import LTX25Engine

/// MSR's target and reference query groups share the same keys but each has a
/// different one-row additive attention template. Split only the query axis;
/// never materialize an N×N video mask for the long reference sequence.
enum MLXGroupedVideoAttention {
  static func evaluate(q:MLXArray,k:MLXArray,v:MLXArray,
    groups:[Int],templates:MLXArray,scale:Float) throws -> MLXArray {
    guard q.ndim == 4,k.shape == q.shape,v.shape == q.shape,
      q.shape[0] == 1,(1...6).contains(groups.count),
      groups.allSatisfy({ $0 > 0 }),groups.reduce(0,+) == q.shape[2],
      templates.shape == [groups.count,q.shape[2]],
      templates.dtype == q.dtype,scale.isFinite,scale > 0,
      MLX.isFinite(templates).all().item(Bool.self),
      templates.min().item(Float.self) >= 0,templates.max().item(Float.self) <= 1 else {
      throw LTXError.invalid("MSR grouped attention needs finite, bounded query templates.")
    }
    return evaluateAdmitted(q:q,k:k,v:v,groups:groups,templates:templates,scale:scale)
  }
  static func evaluateAdmitted(q:MLXArray,k:MLXArray,v:MLXArray,
    groups:[Int],templates:MLXArray,scale:Float) -> MLXArray {
    var start=0,outputs:[MLXArray]=[]
    for (index,rows) in groups.enumerated() {
      let part=q[0...,0...,start..<(start+rows),0...]
      let mask=templates[index].reshaped([1,1,1,q.shape[2]])
      outputs.append(MLXFast.scaledDotProductAttention(queries:part,keys:k,values:v,
        scale:scale,mask:mask))
      start += rows
    }
    return concatenated(outputs,axis:2)
  }
}
