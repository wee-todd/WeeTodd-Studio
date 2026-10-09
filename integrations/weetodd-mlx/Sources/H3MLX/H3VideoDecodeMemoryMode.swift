import Foundation

/// Existing saved-recipe memory modes. Nil direct API requests preserve eager decoding.
public enum H3VideoDecodeMemoryMode: String, Sendable {
  case normal
  case lowMemoryBF16 = "low_memory_bf16"

  static func spatialBatchSize(for mode: Self?) -> Int {
    mode == .lowMemoryBF16 ? 1 : 4
  }

  static func materializesProjection(for mode: Self?) -> Bool {
    mode == nil
  }

  static func materializesFirstResidual(for mode: Self?, spatialBatchSize: Int = 1) -> Bool {
    !(mode == .normal && (3...4).contains(spatialBatchSize))
  }

  static func blockOutputExecutionWindow(for mode: Self?, spatialBatchSize: Int = 1) -> Int {
    // Small normal batches retire each complete block. Larger normal batches
    // and the single-tile low-memory path retain the tile-head boundary.
    mode == .lowMemoryBF16 || (mode == .normal && (3...4).contains(spatialBatchSize)) ? 36 : 1
  }

  static func materializesBlockOutput(for mode: Self?, blockIndex: Int,
    spatialBatchSize: Int = 1) -> Bool {
    mode != .lowMemoryBF16 && !(mode == .normal && (3...4).contains(spatialBatchSize))
      && (blockIndex + 1).isMultiple(of: blockOutputExecutionWindow(for: mode, spatialBatchSize: spatialBatchSize))
  }

  public static func diagnostics(for mode: Self?) -> [String: Any] {
    ["memoryMode": mode?.rawValue ?? "direct_eager",
      "materializationPolicy": mode == .normal ? "adaptive_normal_spatial_batch"
        : mode == .lowMemoryBF16 ? "defer_projections_and_small_tile_residual" : "eager",
      "packedWeightsResidentWithinVideoStage": true,
      // True only when every admitted batch retires every complete block.
      "materializesBlockOutput": mode == nil,
      "blockOutputMaterializationPolicy": mode == nil ? "every_block"
        : mode == .normal ? "batches_1_2_every_block_batches_3_4_tile_head" : "deferred_until_tile_head",
      "blockOutputExecutionWindow": blockOutputExecutionWindow(for: mode, spatialBatchSize: spatialBatchSize(for: mode)),
      "maximumPendingBlockOutputs": blockOutputExecutionWindow(for: mode, spatialBatchSize: spatialBatchSize(for: mode)),
      "maximumPendingSpatialTileBlocks": blockOutputExecutionWindow(for: mode, spatialBatchSize: spatialBatchSize(for: mode)) * spatialBatchSize(for: mode),
      "normalSmallBatchExecutionWindow": mode == .normal ? 1 : NSNull() as Any,
      "normalLargeBatchExecutionWindow": mode == .normal ? 36 : NSNull() as Any,
      "normalFirstResidualPolicy": mode == .normal ? "batches_1_2_eager_batches_3_4_tile_head" : NSNull() as Any,
      "residualExecutionWindow": mode == .normal ? 36 : mode == .lowMemoryBF16 ? 2 : 1,
      "lazyResidualMaximumRows": mode == .lowMemoryBF16 ? 2048 : 0,
      "allocationCacheLimitBytes": 128 * 1024 * 1024,
      "spatialBatch": spatialBatchSize(for: mode),
      "scope": "video-only residency; released before audio; no whole-generation speed guarantee"]
  }
}
