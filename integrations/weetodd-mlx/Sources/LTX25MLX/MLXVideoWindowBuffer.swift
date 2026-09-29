import Foundation
import Metal
import MLX
import Cmlx
import LTX25Engine

private final class VideoWindowOwner {
  let buffer:MTLBuffer
  init(_ buffer:MTLBuffer) { self.buffer=buffer }
}

private func releaseVideoWindow(_ payload:UnsafeMutableRawPointer?) {
  if let payload { Unmanaged<VideoWindowOwner>.fromOpaque(payload).release() }
}

/// Assemble evaluated contiguous windows into one shared Metal allocation.
/// The buffer is writable only before publication to MLX; ownership transfers
/// through a balanced C payload finalizer and no input array is mutated.
final class MLXVideoWindowBuffer {
  let buffer:MTLBuffer
  private let shape:[Int]
  private var written=0
  private var published=false
  var allocatedBytes:Int { buffer.length }
  init(shape:[Int]) throws {
    guard shape.count==4,shape.allSatisfy({ (1...131072).contains($0) }),
      let device=MTLCreateSystemDefaultDevice() else { throw LTXError.invalid("Invalid window assembly shape/device.") }
    var count=1
    for dimension in shape {
      let (product,overflow)=count.multipliedReportingOverflow(by:dimension)
      guard !overflow,product<=device.maxBufferLength/4 else { throw LTXError.invalid("Window assembly exceeds the device buffer limit.") }
      count=product
    }
    guard let storage=device.makeBuffer(length:count*4,options:.storageModeShared) else { throw LTXError.invalid("Cannot allocate decoder window storage.") }
    self.shape=shape;buffer=storage
  }
  func append(_ value:MLXArray,start:Int) throws {
    try Task.checkCancellation()
    guard !published,start==written,value.dtype == .float32,value.ndim==4,
      Array(value.shape.dropFirst())==Array(shape.dropFirst()),value.shape[0]>0,value.shape[0]<=shape[0]-written else {
      throw LTXError.invalid("Decoder windows must be contiguous, nonoverlapping and inside the output.")
    }
    // asData synchronizes evaluation and makes at most one window contiguous.
    let data=value.asData(access:.noCopyIfContiguous).data
    let offset=written*shape[1]*shape[2]*shape[3]*4
    data.withUnsafeBytes { bytes in buffer.contents().advanced(by:offset).copyMemory(from:bytes.baseAddress!,byteCount:bytes.count) }
    written += value.shape[0]
  }
  func finish() throws -> MLXArray {
    guard !published,written==shape[0] else { throw LTXError.invalid("Cannot publish an incomplete or already transferred decoder buffer.") }
    published=true
    // The pinned Swift managed-pointer wrapper retains its closure capture
    // state after finalization. Own the payload directly so every full output
    // allocation is released when MLX drops its last reference.
    let owner=Unmanaged.passRetained(VideoWindowOwner(buffer)).toOpaque()
    return MLXArray(mlx_array_new_data_managed_payload(buffer.contents(),shape.map(Int32.init),Int32(shape.count),MLX_FLOAT32,owner,releaseVideoWindow))
  }
}
