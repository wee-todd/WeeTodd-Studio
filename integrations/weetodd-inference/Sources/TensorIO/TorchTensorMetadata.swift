/// A view into an uncompressed ZIP storage. Offsets/strides are in elements;
/// storageByteRange is the absolute source-file payload range. Logical reads
/// gather strided views and return little-endian bytes suitable for safetensors.
public struct TorchTensorMetadata: Sendable, Equatable {
  public let dtype:String
  public let shape:[UInt64]
  public let storageKey:String
  public let storageByteRange:Range<UInt64>
  public let storageOffset:UInt64
  public let strides:[UInt64]
  public let byteCount:UInt64
  public let isContiguous:Bool
}
