import Foundation
import MLX

/// Transport bytes outlive temporary MLX views and their allocator storage.
/// Regeneration artifacts and individual RGB frames are bounded to256MiB.
public enum H3RegenerateArtifact {
  public static func copyData(_ value:MLXArray) throws -> Data {
    guard value.nbytes > 0,value.nbytes <= 256*1024*1024 else {
      throw H3CheckpointError.invalid("Regeneration artifact exceeds the bounded byte-copy contract.")
    }
    return value.asData(access:.copy).data
  }
}
