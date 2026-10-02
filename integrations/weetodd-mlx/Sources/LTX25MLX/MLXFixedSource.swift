import Foundation
import LTX25Engine
import TensorIO

/// Validates all fixed denoiser weights by header. Connector and reference-only
/// weights in the same installed file belong to separate stages and are not read.
public final class MLXFixedSource {
  public let tensorCount:Int
  public let tensorBytes:UInt64
  private let file:SafeTensorFile
  private let names:[String:String]
  private let shapes:[String:[Int]]
  private let native:MLXNativeWeightSource?
  public init(url:URL,configuration:AVBlockConfiguration,nativeLoading:Bool=true,
    requireKeyframeMarker:Bool=false) throws {
    try configuration.validate()
    var shapes=DenoiserLayout.weightShapes(configuration)
    if requireKeyframeMarker { shapes["keyframes_abs_pos_embedding"]=[1,configuration.videoDimension] }
    let file=try SafeTensorFile(url:url,maximumHeaderBytes:1024*1024)
    var names:[String:String]=[:]
    for original in file.tensors.keys {
      guard let name=LTXAdapterCompatibility.normalize(original), shapes[name] != nil else { continue }
      guard names.updateValue(original,forKey:name) == nil else { throw LTXError.invalid("Duplicate fixed weight: \(name)") }
    }
    for (name,shape) in shapes {
      guard let original=names[name], let record=file.tensors[original],
        record.shape == shape.map(UInt64.init), ["BF16","F16","F32"].contains(record.dtype),
        record.byteCount <= 512*1024*1024 else {
        throw LTXError.invalid("Missing, oversized or incompatible fixed weight: \(name)")
      }
    }
    self.file=file; self.names=names; self.shapes=shapes
    native=nativeLoading ? MLXNativeWeightSource(file:file,url:url) : nil
    tensorCount=file.tensors.count; tensorBytes=file.tensors.values.reduce(0) { $0+$1.byteCount }
  }
  public func read(_ name:String,shape:[Int]) throws -> MLXWeight {
    guard shapes[name] == shape, let original=names[name] else { throw LTXError.invalid("Unvalidated fixed-weight request.") }
    if let native {
      do {
        let weight=try native.read(original,shape)
        // Fixed callers consume one parameter at a time. Finish its read and
        // source checks here, before later projection work or source release.
        try MLXWeight.materialize([weight])
        return weight
      } catch { native.clear();throw error }
    }
    return try MLXWeight(file:file,name:original,shape:shape)
  }
}
