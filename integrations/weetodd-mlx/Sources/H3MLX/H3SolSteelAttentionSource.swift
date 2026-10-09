import Foundation

// Apple MLX Steel primitives/body are MIT licensed. The source is reused from
// the existing attributed H3SteelAttentionSource, not copied from Sol Metal.
/*
MIT License

Copyright © 2023 Apple Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

/// Separate Sol traversal over the existing BQ32/BK16 Float32 Steel consumer.
/// Each valid key is represented by its fine token or one weighted coarse mean.
enum H3SolSteelAttentionSource {
  static let header = H3SteelAttentionSource.header

  private static func replaceOnce(_ source: String, _ old: String, _ new: String) -> String {
    precondition(source.components(separatedBy: old).count == 2,
      "Pinned Steel Sol adaptation seam changed")
    return source.replacingOccurrences(of: old, with: new)
  }

  static let body: String = {
    var source = H3SteelAttentionSource.indexedBody
    source = replaceOnce(source,
      "const uint query_tile = PREFIX_TILES + route_query_tile;",
      "const uint query_tile = route_query_tile;")
    let begin = source.range(of: "  // Each pair of compact 32-row query blocks")!
    let end = source.range(of: "    // Load K block and apply scale", range: begin.lowerBound..<source.endIndex)!
    source.replaceSubrange(begin.lowerBound..<end.lowerBound, with: """
      // All original rows, including protected prefix rows, have one 64-row
      // query route shared by their two 32-row Steel blocks.
      const uint SOL_J = (KEY_ROWS + 63) / 64;
      const uint COARSE_BLOCKS = (SOL_J + BK - 1) / BK;
      const ulong route_base =
          (ulong(tidl.y) * QUERY_TILES + route_query_tile) * ROUTE_WORDS;
      // Coarse traversal precedes fine traversal; max/sum/Otile persist across
      // both. The route map is compact metadata, never gathered K/V storage.
      for (uint traversal = 0; traversal < COARSE_BLOCKS + SOL_J * (64 / BK); ++traversal) {
        const bool coarse = traversal < COARSE_BLOCKS;
        const uint coarse_start = coarse ? traversal * BK : 0;
        const uint fine_block = coarse ? 0 : (traversal - COARSE_BLOCKS) / (64 / BK);
        const uint key_chunk = coarse ? 0 : (traversal - COARSE_BLOCKS) % (64 / BK);
        uint selected_size = 0;
        int kb = int(fine_block);
        if (coarse) {
          selected_size = min(uint(BK), SOL_J - coarse_start);
          bool has_coarse = false;
          for (uint column = 0; column < selected_size; ++column) {
            uint j = coarse_start + column;
            has_coarse |= ((ROUTE_BITS[route_base + j / 32] >> (j % 32)) & 1u) == 0;
          }
          // Uniform skip is essential: an entirely masked initial tile must
          // not turn finite_min-finite_min into spurious probability mass.
          if (!has_coarse) continue;
        } else {
          if (((ROUTE_BITS[route_base + fine_block / 32] >> (fine_block % 32)) & 1u) == 0) continue;
          const int remaining = int(TILE_SIZES[fine_block]) - int(key_chunk * BK);
          if (remaining <= 0) continue;
          selected_size = uint(min(remaining, BK));
        }
        const device T* selected_k = coarse
            ? KC + (ulong(tidl.y) * SOL_J + coarse_start) * BD
            : K + ulong(fine_block * 64 + key_chunk * BK) * params->K_strides[2];
        const device T* selected_v = coarse
            ? VC + (ulong(tidl.y) * SOL_J + coarse_start) * BD
            : V + ulong(fine_block * 64 + key_chunk * BK) * params->V_strides[2];
        const ulong selected_k_stride = coarse ? BD : params->K_strides[2];
        const ulong selected_v_stride = coarse ? BD : params->V_strides[2];
        KBlockLoader selected_loader_k(selected_k, selected_k_stride, Ks, simd_group_id, simd_lane_id);
        VBlockLoader selected_loader_v(selected_v, selected_v_stride, Vs, simd_group_id, simd_lane_id);

      """)
    source = replaceOnce(source, "    // Mask internal padding in every ragged prefix or video tile.", """
        // Steel logits are already in base-2 units. Add multiplicity before
        // the row maximum so both numerator and denominator remain stable.
        if (coarse) {
          STEEL_PRAGMA_UNROLL
          for (short i = 0; i < decltype(Stile)::kTileRows; ++i) {
            STEEL_PRAGMA_UNROLL
            for (short j = 0; j < decltype(Stile)::kTileCols; ++j) {
              short col_pos = sn + j * decltype(Stile)::kFragCols;
              STEEL_PRAGMA_UNROLL
              for (short jj = 0; jj < decltype(Stile)::MMAFrag_t::kElemCols; ++jj) {
                uint column = uint(col_pos + jj);
                if (column < selected_size) {
                  uint key_tile = coarse_start + column;
                  bool selected = ((ROUTE_BITS[route_base + key_tile / 32] >> (key_tile % 32)) & 1u) != 0;
                  Stile.frag_at(i, j)[jj] = selected ? Limits<AccumType>::finite_min
                    : Stile.frag_at(i, j)[jj] + log2(float(TILE_SIZES[key_tile]));
                }
              }
            }
          }
        }

        // Mask internal padding in every ragged prefix or video tile.
      """)
    return source
  }()
}
