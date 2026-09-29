import MLX

/// Float32 remains available for the original diagnostic oracle. BF16 follows
/// the installed convolutional decoder's trained storage and reference path.
public enum MLXVideoPrecision:String,Sendable {
  case float32,bfloat16
  var dtype:DType { self == .bfloat16 ? .bfloat16 : .float32 }
  var bytes:Int { self == .bfloat16 ? 2 : 4 }
  var windowElements:Int { (self == .bfloat16 ? 64 : 8)*1024*1024 }
}
