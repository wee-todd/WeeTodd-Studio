import MLX

public enum FlashVSRScheduler {
  public static let timestep: Float = 1000
  /// The released one-evaluation shifted flow schedule has sigma endpoints [1, 0].
  /// Noise is injected by the caller, so reproducibility never assumes cross-backend RNG parity.
  public static func step(noise: MLXArray, velocity: MLXArray) throws -> MLXArray {
    guard noise.shape == velocity.shape else { throw FlashVSRError.invalid("Flow velocity and noise shapes differ.") }
    return noise - velocity
  }
}
