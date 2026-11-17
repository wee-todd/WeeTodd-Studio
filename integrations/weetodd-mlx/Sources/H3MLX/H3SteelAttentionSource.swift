import Foundation
import NativeMetalSupport

// MLX Steel primitives and attention body: MIT licensed, not Apache-2.0.
// Embedded from mlx-swift 901941965d82e4a216d4d117231d847d194c563d.
// Indexed padded-tile adaptation preserves the native BQ=32/BK=16 accumulator order.
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

enum H3SteelAttentionSource {
  static let header = NativeSteelPrimitives.header
  static let indexedBody = #"""
 // clang-format on

  // Pacifying compiler
  (void)lid;

  // Move to correct block
  ulong3 tidl{tid.x, tid.y, tid.z};

  const uint route_query_tile = tid.x / 2;
  const uint query_half = tid.x % 2;
  const uint query_tile = PREFIX_TILES + route_query_tile;
  const int query_block_size = int(TILE_SIZES[query_tile]) - int(query_half * BQ);
  if (query_block_size <= 0) return;
  const uint query_offset = route_query_tile * 64
      + query_half * BQ;
  Q += tidl.z * params->Q_strides[0] + // Batch
      tidl.y * params->Q_strides[1] + // Head
      ulong(query_offset) * params->Q_strides[2]; // Compact sequence

  ulong kv_head_idx = int(tid.y) / params->gqa_factor;
  K += tidl.z * params->K_strides[0] + // Batch
      kv_head_idx * params->K_strides[1]; // Head

  V += tidl.z * params->V_strides[0] + // Batch
      kv_head_idx * params->V_strides[1]; // Head

  O += tidl.z * params->O_strides[0] + // Batch
      tidl.y * params->O_strides[1] + // Head
      ulong(query_offset) * params->O_strides[2]; // Compact sequence

  if (has_mask) {
    mask += tidl.z * mask_params->M_strides[0] + // Batch
        tidl.y * mask_params->M_strides[1]; // Head
  }

  // Prepare threadgroup memory
  constexpr short padQ = 16 / sizeof(T);
  constexpr short padK = 16 / sizeof(T);
  constexpr short padV = 16 / sizeof(T);

  constexpr short LDQ_tgp = BD + padQ;
  constexpr short LDK_tgp = BK + padK;
  constexpr short LDV_tgp = BD + padV;

  constexpr short tgp_mem_0 = (BK + padK) * (BD);
  constexpr short tgp_mem_1 = BK * (BD + padV);
  constexpr short tgp_mem_s = tgp_mem_0 > tgp_mem_1 ? tgp_mem_0 : tgp_mem_1;

  threadgroup T Q_smem[BQ * (BD + padQ)];
  threadgroup T KV_smem[tgp_mem_s];

  threadgroup T* Qs = Q_smem;
  threadgroup T* Ks = KV_smem;
  threadgroup T* Vs = KV_smem;

  // Prepare block loaders
  using QBlockLoader = BlockLoaderT<
      /* typename T = */ T,
      /* short BROWS = */ BQ,
      /* short BCOLS = */ BD,
      /* short kDstStrRow = */ LDQ_tgp,
      /* short kDstStrCol = */ 1,
      /* short reduction_dim = */ 1,
      /* short tgp_size = */ WM * WN * 32>;

  // K is loaded in transposed
  using KBlockLoader = BlockLoaderT<
      /* typename T = */ T,
      /* short BROWS = */ BK,
      /* short BCOLS = */ BD,
      /* short kDstStrRow = */ 1,
      /* short kDstStrCol = */ LDK_tgp,
      /* short reduction_dim = */ 0,
      /* short tgp_size = */ WM * WN * 32>;

  using VBlockLoader = BlockLoaderT<
      /* typename T = */ T,
      /* short BROWS = */ BK,
      /* short BCOLS = */ BD,
      /* short kDstStrRow = */ LDV_tgp,
      /* short kDstStrCol = */ 1,
      /* short reduction_dim = */ 0,
      /* short tgp_size = */ WM * WN * 32>;

  QBlockLoader loader_q(
      Q, params->Q_strides[2], Qs, simd_group_id, simd_lane_id);
  KBlockLoader loader_k(
      K, params->K_strides[2], Ks, simd_group_id, simd_lane_id);
  VBlockLoader loader_v(
      V, params->V_strides[2], Vs, simd_group_id, simd_lane_id);

  const AccumType scale = params->scale * M_LOG2E_F;

  // Prepare MMA tiles
  constexpr short kFragSize = 8; // MMAFrag size
  using MMAFrag_acc_t = BaseMMAFrag<AccumType, kFragSize, kFragSize>;

  constexpr int kNWarps = WM * WN;
  static_assert(
      BQ >= (kNWarps * kFragSize) && BQ % (kNWarps * kFragSize) == 0,
      "Each simdgroup must host atleast 1 simdgroup matrix along Q sequence.");

  // Q seq frags per warp
  constexpr int TQ = BQ / (kNWarps * kFragSize);
  // KV sequence frags (all warps load the same frags)
  constexpr int TK = BK / kFragSize;
  // HeadDim frags (all warps load the same frags)
  constexpr int TD = BD / kFragSize;

  static_assert(TQ == 1, "Check TQ");

  MMATile<AccumType, TQ, 1, MMAFrag_acc_t> Qtile;
  MMATile<AccumType, 1, TK, MMAFrag_acc_t> Ktile;
  MMATile<AccumType, TQ, TK, MMAFrag_acc_t> Stile;
  MMATile<AccumType, 1, 1, MMAFrag_acc_t> Vtile;
  MMATile<AccumType, TQ, TD, MMAFrag_acc_t> Otile;

  Otile.clear();

  // Prepare mma tile offsets
  const short2 simd_coord = MMAFrag_acc_t::get_coord(simd_lane_id);
  const short sm = simd_coord.y;
  const short sn = simd_coord.x;
  const short tm = kFragSize * TQ * simd_group_id;

  const short Qs_offset = (tm + sm) * LDQ_tgp + sn;
  const short Ks_offset = sm * LDK_tgp + sn;
  const short Vs_offset = sm * LDV_tgp + sn;

  constexpr short Qs_tile_stride = kFragSize;
  constexpr short Ks_tile_stride = kFragSize * LDK_tgp;

  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Load Q blocks
  if (query_block_size < BQ) {
    loader_q.load_safe(short2(BD, query_block_size));
  } else {
    loader_q.load_unsafe();
  }

  // Init row reduction variables
  constexpr short kRowsPT = decltype(Stile)::kRowsPerThread;

  AccumType max_score[kRowsPT];
  AccumType sum_score[kRowsPT] = {0};

  // Init to -Inf
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < kRowsPT; ++i) {
    max_score[i] = Limits<AccumType>::finite_min;
  }

  if (has_sinks) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      max_score[i] = M_LOG2E_F * static_cast<AccumType>(sinks[tidl.y]);
      sum_score[i] = 1;
    }
  }

  int kb_lim = params->NK;
  int kb_min_causal = params->NK;

  if (do_causal) {
    int q_max = (tid.x + 1) * BQ + params->qL_off;
    kb_lim = (q_max + BK - 1) / BK;
    kb_lim = min(params->NK, kb_lim);

    int q_min = tid.x * BQ + params->qL_off;
    q_min = max(0, q_min);
    kb_min_causal = (q_min / BK);
  }

  // Each pair of compact 32-row query blocks shares one trained 64-row VSA route.
  const ulong route_base =
      ((ulong(tidl.z) * HEADS + tidl.y) * VIDEO_TILES + route_query_tile) * ROUTE_TILES;
  for (uint route_block = 0; route_block < ROUTE_TILES * (64 / BK); ++route_block) {
    const uint route_slot = route_block / (64 / BK);
    const uint key_chunk = route_block % (64 / BK);
    const int kb = int(ROUTES[route_base + route_slot]);
    const int remaining = int(TILE_SIZES[kb]) - int(key_chunk * BK);
    if (remaining <= 0) continue;
    const uint selected_size = uint(min(remaining, BK));
    KBlockLoader selected_loader_k(
        K + ulong(kb * 64 + key_chunk * BK) * params->K_strides[2],
        params->K_strides[2], Ks, simd_group_id, simd_lane_id);
    VBlockLoader selected_loader_v(
        V + ulong(kb * 64 + key_chunk * BK) * params->V_strides[2],
        params->V_strides[2], Vs, simd_group_id, simd_lane_id);
    // Load K block and apply scale
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (selected_size < BK) {
      selected_loader_k.load_safe(short2(BD, selected_size));
    } else {
      selected_loader_k.load_unsafe();
    }

    // Do S = Q @ K.T
    Stile.clear();

    threadgroup_barrier(mem_flags::mem_threadgroup);

    STEEL_PRAGMA_UNROLL
    for (short dd = 0; dd < TD; dd++) {
      simdgroup_barrier(mem_flags::mem_none);

      Qtile.template load<T, 1, 1, LDQ_tgp, 1>(
          &Qs[Qs_offset + dd * Qs_tile_stride]);
      Ktile.template load<T, 1, 1, LDK_tgp, 1>(
          &Ks[Ks_offset + dd * Ks_tile_stride]);

      simdgroup_barrier(mem_flags::mem_none);

      tile_matmad(Stile, Qtile, Ktile, Stile);
    }

    // Apply scale in float32
    STEEL_PRAGMA_UNROLL
    for (short ii = 0; ii < decltype(Stile)::kElemsPerTile; ii++) {
      Stile.elems()[ii] *= scale;
    }

    // Mask internal padding in every ragged prefix or video tile.
    if (selected_size < BK) {
      using stile_t = decltype(Stile);
      using selem_t = typename stile_t::elem_type;
      constexpr auto neg_inf = Limits<selem_t>::finite_min;
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < stile_t::kTileRows; i++) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < stile_t::kTileCols; j++) {
          short col_pos = sn + (j * stile_t::kFragCols);
          STEEL_PRAGMA_UNROLL
          for (short jj = 0; jj < stile_t::MMAFrag_t::kElemCols; jj++) {
            if ((col_pos + jj) >= selected_size) {
              Stile.frag_at(i, j)[jj] = neg_inf;
            }
          }
        }
      }
    }

    // Mask out if causal
    if (do_causal && kb >= kb_min_causal) {
      using stile_t = decltype(Stile);
      using selem_t = typename stile_t::elem_type;
      constexpr auto neg_inf = Limits<selem_t>::finite_min;

      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < stile_t::kTileRows; i++) {
        const int row_pos =
            tid.x * BQ + params->qL_off + tm + sm + (i * stile_t::kFragRows);
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < stile_t::kTileCols; j++) {
          const int col_pos = kb * BK + sn + (j * stile_t::kFragCols);
          STEEL_PRAGMA_UNROLL
          for (short jj = 0; jj < stile_t::MMAFrag_t::kElemCols; jj++) {
            if (row_pos < (col_pos + jj)) {
              Stile.frag_at(i, j)[jj] = neg_inf;
            }
          }
        }
      }
    }

    // Other masking as needed
    if (has_mask) {
      using stile_t = decltype(Stile);
      using selem_t = typename stile_t::elem_type;
      constexpr auto neg_inf = Limits<selem_t>::finite_min;

      constexpr bool is_bool = is_same_v<MaskType, bool>;
      using melem_t = typename metal::conditional_t<is_bool, bool, selem_t>;

      using MMAFrag_mask_t = BaseMMAFrag<melem_t, kFragSize, kFragSize>;
      using frag_t = typename MMAFrag_mask_t::frag_type;

      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < stile_t::kTileRows; i++) {
        const int row_pos = tid.x * BQ + tm + sm + (i * stile_t::kFragRows);
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < stile_t::kTileCols; j++) {
          const int col_pos = kb * BK + sn + (j * stile_t::kFragCols);

          frag_t mfrag;

          MMAFrag_mask_t::load_safe(
              mfrag,
              mask,
              int64_t(mask_params->M_strides[2]),
              Int<1>{},
              params->qL,
              params->kL,
              row_pos,
              col_pos);

          STEEL_PRAGMA_UNROLL
          for (short jj = 0; jj < stile_t::MMAFrag_t::kElemsPerFrag; jj++) {
            if constexpr (is_bool) {
              Stile.frag_at(i, j)[jj] =
                  mfrag[jj] ? Stile.frag_at(i, j)[jj] : neg_inf;
            } else {
              Stile.frag_at(i, j)[jj] += M_LOG2E_F * selem_t(mfrag[jj]);
            }
          }
        }
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Load V blocks
    if (selected_size < BK) {
      selected_loader_v.load_safe(short2(BD, selected_size));
    } else {
      selected_loader_v.load_unsafe();
    }

    // Do softmax

    // Temp variables
    AccumType new_max[kRowsPT];
    AccumType factor[kRowsPT];
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      new_max[i] = max_score[i];
    }

    // Row max
    Stile.template row_reduce<MaxOp>(new_max);

    // exp(Si - rowmax(Si))
    Stile.template row_bin_op<ExpSubOp>(new_max);

    // Factor exp(rowmax(Si) - rowmax(Si-1))
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      factor[i] = fast::exp2(max_score[i] - new_max[i]);
    }

    // Save max for next iteration
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      max_score[i] = new_max[i];
    }

    // Row Sum
    AccumType sum_score_tmp[kRowsPT] = {0};
    Stile.template row_reduce<SumOp>(sum_score_tmp);

    // Update norm
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      sum_score[i] = sum_score[i] * factor[i] + sum_score_tmp[i];
    }

    // Update O
    Otile.template row_bin_op<MulOp>(factor);

    // Load V into registers
    threadgroup_barrier(mem_flags::mem_threadgroup);

    STEEL_PRAGMA_UNROLL
    for (short iq = 0; iq < TQ; iq++) {
      STEEL_PRAGMA_UNROLL
      for (short id = 0; id < TD; id++) {
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < TK; ik++) {
          if constexpr (BD >= 128) {
            simdgroup_barrier(mem_flags::mem_none);
          }

          const short kk = ik * kFragSize;
          const short dd = id * kFragSize;

          Vtile.template load<T, 1, 1, LDV_tgp, 1>(
              &Vs[Vs_offset + kk * LDV_tgp + dd]);

          if constexpr (BD >= 128) {
            simdgroup_barrier(mem_flags::mem_none);
          }

          MMAFrag_acc_t::mma(
              Otile.frag_at(iq, id),
              Stile.frag_at(iq, ik),
              Vtile.frag_at(0, 0),
              Otile.frag_at(iq, id));
        }
      }
    }

    // Indexed loaders are reconstructed at the selected tile address.
  }

  // Normalize output
  Otile.template row_bin_op<DivOp>(sum_score);
  threadgroup_barrier(mem_flags::mem_none);

  // Store results
  O += (tm + sm) * params->O_strides[2] + sn;

  if (query_block_size < BQ) {
    auto dst_tile_dims = short2(BD - sn, query_block_size - (tm + sm));
    if (dst_tile_dims.x <= 0 || dst_tile_dims.y <= 0) return;
    Otile.template store_safe<T, 1, 1>(O, params->O_strides[2], dst_tile_dims);
  } else {
    Otile.template store<T, 1, 1>(O, params->O_strides[2]);
  }

"""#
  // Same MMA/softmax body, with a row-indexed loader. Padding remains only
  // in the small tile metadata/output; Q/K/V stay in original row storage.
  static let originalRowLoader = #"""
  template <typename T, short BR, short BC, short DR, short DC,
      short REDUCTION, short THREADS>
  struct H3MappedBlockLoader {
    const device T* source;
    ulong stride;
    threadgroup T* destination;
    uint lane;
    const device int* rows;
    uint start;
    H3MappedBlockLoader(const device T* p, ulong s, threadgroup T* d,
        uint group, uint lane_id, const device int* r, uint offset)
      : source(p), stride(s), destination(d), lane(group*32+lane_id), rows(r), start(offset) {}
    void load_safe(short2 size) {
      constexpr uint VECTOR = BR*BC/THREADS;
      constexpr uint COLUMN_THREADS = BC/VECTOR;
      constexpr uint ROW_THREADS = THREADS/COLUMN_THREADS;
      uint row = lane/COLUMN_THREADS;
      uint column = (lane%COLUMN_THREADS)*VECTOR;
      #pragma clang loop unroll(full)
      for(uint r=row; r<BR; r+=ROW_THREADS) {
        ulong address = r<uint(size.y) ? ulong(rows[start+r])*stride+column : 0;
        #pragma clang loop unroll(full)
        for(uint c=0; c<VECTOR; ++c) {
          destination[r*DR+(column+c)*DC] = (r<uint(size.y) && column+c<uint(size.x))
            ? source[address+c] : T(0);
        }
      }
    }
    void load_unsafe() { load_safe(short2(BC,BR)); }
  };
  """#

  static let originalRowBody = indexedBody
    .replacingOccurrences(of:"ulong(query_offset) * params->Q_strides[2]",with:"ulong(0)")
    .replacingOccurrences(of:"BlockLoaderT<",with:"H3MappedBlockLoader<")
    .replacingOccurrences(of:"Q, params->Q_strides[2], Qs, simd_group_id, simd_lane_id);",
      with:"Q, params->Q_strides[2], Qs, simd_group_id, simd_lane_id, ROW_MAP, PREFIX_TILES*64+query_offset);")
    .replacingOccurrences(of:"K, params->K_strides[2], Ks, simd_group_id, simd_lane_id);",
      with:"K, params->K_strides[2], Ks, simd_group_id, simd_lane_id, ROW_MAP, 0);")
    .replacingOccurrences(of:"V, params->V_strides[2], Vs, simd_group_id, simd_lane_id);",
      with:"V, params->V_strides[2], Vs, simd_group_id, simd_lane_id, ROW_MAP, 0);")
    .replacingOccurrences(of:"K + ulong(kb * 64 + key_chunk * BK) * params->K_strides[2]",with:"K")
    .replacingOccurrences(of:"V + ulong(kb * 64 + key_chunk * BK) * params->V_strides[2]",with:"V")
    .replacingOccurrences(of:"params->K_strides[2], Ks, simd_group_id, simd_lane_id);",
      with:"params->K_strides[2], Ks, simd_group_id, simd_lane_id, ROW_MAP, kb*64+key_chunk*BK);")
    .replacingOccurrences(of:"params->V_strides[2], Vs, simd_group_id, simd_lane_id);",
      with:"params->V_strides[2], Vs, simd_group_id, simd_lane_id, ROW_MAP, kb*64+key_chunk*BK);")

  /// The original 64-row source remains byte-for-byte stable for Preview/Sol.
  /// Only mapped-row Omni instantiation changes tile address arithmetic; the
  /// BQ32/BK16 matrix operations and softmax reduction are identical.
  static func mappedBody(tileRows: Int) -> String {
    precondition(tileRows == 64 || tileRows == 128)
    return originalRowBody
      .replacingOccurrences(of: "tid.x / 2", with: "tid.x / \(tileRows / 32)")
      .replacingOccurrences(of: "tid.x % 2", with: "tid.x % \(tileRows / 32)")
      .replacingOccurrences(of: "* 64", with: "* \(tileRows)")
      .replacingOccurrences(of: "*64", with: "*\(tileRows)")
      .replacingOccurrences(of: "(64 / BK)", with: "(\(tileRows) / BK)")
  }

}
