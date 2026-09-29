import Foundation
import MLX
import TensorIO
import LTX25Engine

/// Lazy native handles for a validated checkpoint. Each returned factor is
/// removed immediately; callers complete only the active layer's I/O.
final class MLXNativeWeightSource {
  private let file:SafeTensorFile
  private let url:URL
  private var pending:[String:MLXArray]=[:]
  init(file:SafeTensorFile,url:URL) { self.file=file;self.url=url }
  func clear() { pending.removeAll() }
  func read(_ name:String,_ shape:[Int]) throws -> MLXWeight {
    do {
      try Task.checkCancellation();try file.checkUnchanged(at:url)
      if pending[name] == nil {
        pending.removeAll()
        pending=try loadArrays(url:url)
        guard Set(pending.keys)==Set(file.tensors.keys) else { throw LTXError.invalid("Native checkpoint header changed.") }
      }
      let file=self.file,url=self.url
      return try MLXWeight(file:file,name:name,shape:shape,
        verifyMaterialization:{ try file.checkUnchanged(at:url) },tensor:{ key in
          guard let value=self.pending.removeValue(forKey:key) else { throw LTXError.invalid("Missing native weight factor.") }
          return value
        })
    } catch { clear();throw error }
  }
}
