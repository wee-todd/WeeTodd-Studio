import Foundation
import MLX
import LTX25Engine

/// Versioned reproducibility boundary; older requests retain their native stream.
public enum MLXNoisePolicy:String,Codable,Sendable {
  case native = "native_box_muller_v1"
  case releasedMLX = "mlx_threefry_bf16_v1"
  public var algorithm:String { self == .native ? GaussianNoise.algorithm : rawValue }
  static func seeded(_ seed:UInt64,tokens:Int) -> MLXArray {
    let (_,draw)=MLXRandom.split(key:MLXRandom.key(seed))
    return MLXRandom.normal([1,tokens,128],key:draw).asType(.bfloat16).reshaped([tokens,128])
  }
}
