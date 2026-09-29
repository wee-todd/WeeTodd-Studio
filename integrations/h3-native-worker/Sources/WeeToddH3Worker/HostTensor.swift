import Foundation
import NNC

var boundedHostIO:Bool { ProcessInfo.processInfo.environment["WEETODD_NNC_BUFFER_IO"]=="bounded" }

// NNC's array initializer retains its immutable backing array. Reshape creates
// a view, avoiding the second full CPU allocation made by the sequence initializer.
func hostTensor<T:TensorNumeric>(_ values:[T],shape:TensorShape) -> Tensor<T> {
  if boundedHostIO { return Tensor<T>(values).reshaped(format:.NHWC,shape:shape) }
  return Tensor<T>(values,kind:.CPU,format:.NHWC,shape:shape)
}
func writeTensorBytes(_ bytes:UnsafeRawBufferPointer,to target:URL) throws {
  if boundedHostIO,let pointer=bytes.baseAddress {
    // The synchronous atomic write completes before the owning tensor is released.
    try Data(bytesNoCopy:UnsafeMutableRawPointer(mutating:pointer),count:bytes.count,deallocator:.none).write(to:target,options:.atomic)
  } else { try Data(bytes).write(to:target,options:.atomic) }
}
