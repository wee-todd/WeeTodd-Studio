import Foundation
import InferenceContracts

public struct ReCaMemoryPlan: Sendable {
  public let estimatedBytes: UInt64
  public let kvBytes: UInt64
  public let packingBytes: UInt64
  public static func admit(geometry: ReCaGeometry, profile: ReCaPrecision,
    residency: ReCaResidency, budgetBytes: UInt64) throws -> Self {
    let pixels = UInt64(geometry.paddedWidth * geometry.paddedHeight)
    let kv = UInt64(geometry.tokenWidth * geometry.tokenHeight) * 811_008
    let weights: UInt64 = profile == .bf16 ? 10_603_673_624 : 6_266_935_048
    // Conservative admission includes prefix/LQ activations, bounded attention
    // tiles, cross KV, decoder masters/casts, source/encoder queues and headroom.
    let decoder: UInt64 = 500_000_000
    let base = weights + kv + pixels * 1500 + 1_000_000_000 + (residency == .resident ? decoder : 0)
    let packing = ReCaPackedTilePlan.admissionBytes(height: geometry.tokenHeight,
      width: geometry.tokenWidth, windowHeight: 22, windowWidth: 40, sliding: true,
      queryFrames: 6, keyFrames: 8, queryChannels: 3072, keyChannels: 3072, scalarBytes: 2)
      + ReCaPackedTilePlan.admissionBytes(height: geometry.latentHeight,
        width: geometry.latentWidth, windowHeight: 24, windowWidth: 24, sliding: false,
        queryFrames: 4, keyFrames: 8, queryChannels: 512, keyChannels: 128, scalarBytes: 4)
    let estimate = base + base / 4 + packing
    guard estimate <= budgetBytes else {
      throw ReCaError.invalid("ReCaVSR estimates \((estimate + (1 << 30) - 1) >> 30) GiB for this output; the budget is \(budgetBytes >> 30) GiB. Choose a smaller scale/source, INT8, or an explicitly larger budget.")
    }
    return .init(estimatedBytes: estimate, kvBytes: kv, packingBytes: packing)
  }
}
