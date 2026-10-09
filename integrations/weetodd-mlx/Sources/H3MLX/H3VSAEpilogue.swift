import MLX

/// Restore original VSA row order and apply compression with the existing
/// BF16 cast, multiply, then add boundaries. No attention reduction changes.
enum H3VSAEpilogue {
  static func evaluate(prefixOutput: MLXArray, paddedVideo: MLXArray,
    compression: MLXArray, gate: MLXArray, tiles: H3FastTiles) throws -> MLXArray {
    try Task.checkCancellation()
    let count = tiles.sizes.count
    guard gate.ndim == 4, gate.shape[0] == 1,
      (1...56).contains(gate.shape[1]), gate.shape[2] == tiles.rows,
      gate.shape[3] == 128, gate.dtype == .bfloat16,
      (1...40_000).contains(tiles.rows), (1..<tiles.rows).contains(tiles.prefixRows),
      count > 1, count <= 40_000, (1..<count).contains(tiles.prefixTiles),
      tiles.sizes.allSatisfy({ (1...64).contains($0) }),
      tiles.rowSlots.count == tiles.rows else {
      throw H3CheckpointError.invalid("Invalid H3 VSA epilogue tensors or row map.")
    }
    let heads = gate.shape[1], videoRows = (count-tiles.prefixTiles)*64
    guard prefixOutput.shape == [1,heads,tiles.prefixRows,128],
      paddedVideo.shape == [1,heads,videoRows,128],
      compression.shape == [1,heads,count,128],
      prefixOutput.dtype == .bfloat16, paddedVideo.dtype == .bfloat16,
      compression.dtype == .float32,
      tiles.rowSlots.enumerated().allSatisfy({ row, entry in
        let slot = Int(entry)
        guard slot >= 0, slot < count*64 else { return false }
        let tile = slot/64
        return slot%64 < tiles.sizes[tile]
          && ((row < tiles.prefixRows) == (tile < tiles.prefixTiles))
      }) else {
      throw H3CheckpointError.invalid("H3 VSA epilogue inputs do not match the tile layout.")
    }
    let result: MLXArray
    if Device.defaultDevice().deviceType == .gpu {
      result = kernel([prefixOutput,paddedVideo,compression,gate,MLXArray(tiles.rowSlots)],
        template:[("HEADS",heads),("ROWS",tiles.rows),
          ("PREFIX_ROWS",tiles.prefixRows),("PREFIX_TILES",tiles.prefixTiles)],
        grid:(gate.size,1,1),threadGroup:(256,1,1),
        outputShapes:[gate.shape],outputDTypes:[.bfloat16])[0]
    } else {
      let rowTiles = MLXArray(tiles.rowSlots.map { $0/64 })
      let expanded = take(compression,rowTiles,axis:2).asType(.bfloat16)
      let videoScatter = MLXArray(tiles.rowSlots.dropFirst(tiles.prefixRows)
        .map { $0-Int32(tiles.prefixTiles*64) })
      let output = concatenated([prefixOutput,take(paddedVideo,videoScatter,axis:2)],axis:2)
      result = output+expanded*gate
    }
    eval(result)
    try Task.checkCancellation()
    return result
  }

  // Independent scalar epilogue. Read runtime strides rather than materializing
  // head/row/feature views, including reversed and broadcast axes.
  private static let kernel = MLXFast.metalKernel(
    name:"weetodd_h3_vsa_original_rows_rounded_epilogue",
    inputNames:["prefix","padded","compression","gate","row_slots"],
    outputNames:["output"],source:"""
      uint i = thread_position_in_grid.x;
      if (i >= HEADS*ROWS*128) return;
      uint feature = i%128;
      uint row = (i/128)%ROWS;
      uint head = i/(ROWS*128);
      long slot = long(row_slots[row]);
      long tile = slot/64;
      bfloat attention;
      if (row < PREFIX_ROWS) {
        long po = long(head)*long(prefix_strides[1])
          + long(row)*long(prefix_strides[2])+long(feature)*long(prefix_strides[3]);
        attention = prefix[po];
      } else {
        long video_row = slot-long(PREFIX_TILES)*64;
        long vo = long(head)*long(padded_strides[1])
          + video_row*long(padded_strides[2])+long(feature)*long(padded_strides[3]);
        attention = padded[vo];
      }
      long co = long(head)*long(compression_strides[1])
        + tile*long(compression_strides[2])+long(feature)*long(compression_strides[3]);
      long go = long(head)*long(gate_strides[1])
        + long(row)*long(gate_strides[2])+long(feature)*long(gate_strides[3]);
      bfloat correction = bfloat(compression[co]);
      bfloat product = bfloat(float(correction)*float(gate[go]));
      output[i] = bfloat(float(attention)+float(product));
      """,ensureRowContiguous:false)
}
