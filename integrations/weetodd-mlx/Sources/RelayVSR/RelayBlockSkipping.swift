import Foundation
import MLX
import NativeMetalSupport

struct RelayBlockPlane {
  let rows: Range<Int>, columns: Range<Int>, window: RelayWindow
}
struct RelayBlockSkippingPlan {
  let classes: [[UInt8]]
  init(queries: [Int], height: Int, width: Int, planes: [RelayBlockPlane], keyBlockSize: Int) throws {
    guard (1...4096).contains(height), (1...4096).contains(width),
      (1...128).contains(queries.count), queries.allSatisfy({ (0..<height * width).contains($0) }),
      (1...4).contains(planes.count), [16, 32].contains(keyBlockSize), planes.allSatisfy({
        !$0.rows.isEmpty && !$0.columns.isEmpty && $0.rows.lowerBound >= 0 && $0.rows.upperBound <= height
          && $0.columns.lowerBound >= 0 && $0.columns.upperBound <= width
          && $0.window.height >= 2 && $0.window.width >= 2
      }) else { throw RelayVSRError.invalid("Invalid Relay block classification geometry.") }
    let keys = planes.reduce(0) { $0 + $1.rows.count * $1.columns.count }
    let nk = (keys + keyBlockSize - 1) / keyBlockSize, nq = (queries.count + 31) / 32
    guard nq * nk <= 1_000_000 else { throw RelayVSRError.invalid("Relay block classification exceeds its metadata bound.") }
    struct Rectangle {
      let y: Range<Int>, x: Range<Int>
      func intersects(y row: Int, left: Int, right: Int) -> Bool {
        y.contains(row) && left < x.upperBound && right > x.lowerBound
      }
      func contains(y row: Int, left: Int, right: Int) -> Bool {
        y.contains(row) && left >= x.lowerBound && right <= x.upperBound
      }
    }
    var result: [[UInt8]] = []
    for first in stride(from: 0, to: queries.count, by: 32) {
      let group = queries[first..<min(first + 32, queries.count)]
      var union: [Rectangle] = [], intersection: [Rectangle] = []
      var earliest = Array(repeating: Int.max, count: group.count), prefix = 0
      for plane in planes {
        let ys = group.map { plane.window.range(at: $0 / width, extent: height, size: plane.window.height) }
        let xs = group.map { plane.window.range(at: $0 % width, extent: width, size: plane.window.width) }
        union.append(Rectangle(y: ys.map(\.lowerBound).min()!..<ys.map(\.upperBound).max()!,
          x: xs.map(\.lowerBound).min()!..<xs.map(\.upperBound).max()!))
        let yl = ys.map(\.lowerBound).max()!, yu = ys.map(\.upperBound).min()!
        let xl = xs.map(\.lowerBound).max()!, xu = xs.map(\.upperBound).min()!
        intersection.append(Rectangle(y: yl..<max(yl, yu), x: xl..<max(xl, xu)))
        for i in ys.indices {
          let y = max(ys[i].lowerBound, plane.rows.lowerBound), x = max(xs[i].lowerBound, plane.columns.lowerBound)
          if y < min(ys[i].upperBound, plane.rows.upperBound) && x < min(xs[i].upperBound, plane.columns.upperBound) {
            earliest[i] = min(earliest[i], prefix + (y - plane.rows.lowerBound) * plane.columns.count + x - plane.columns.lowerBound)
          }
        }
        prefix += plane.rows.count * plane.columns.count
      }
      let seenAfter = earliest.max()! == Int.max ? Int.max : earliest.max()! / keyBlockSize
      var row = Array(repeating: UInt8(1), count: nk)
      for kb in 0..<nk where (kb + 1) * keyBlockSize <= keys {
        let lo = kb * keyBlockSize, hi = lo + keyBlockSize
        var any = false, every = true, offset = 0
        for (i, plane) in planes.enumerated() {
          let count = plane.rows.count * plane.columns.count
          var start = max(lo, offset) - offset
          let stop = min(hi, offset + count) - offset
          while start < stop {
            let y = plane.rows.lowerBound + start / plane.columns.count
            let x = plane.columns.lowerBound + start % plane.columns.count
            let length = min(stop - start, plane.columns.upperBound - x)
            any = any || union[i].intersects(y: y, left: x, right: x + length)
            every = every && intersection[i].contains(y: y, left: x, right: x + length)
            start += length
          }
          offset += count
        }
        // A union bounding rectangle may be conservative for a ragged Q group,
        // but can never incorrectly omit a visible key. Tail K remains mixed.
        if !any && kb > seenAfter { row[kb] = 0 }
        else if every { row[kb] = 2 }
      }
      result.append(row)
    }
    classes = result
  }
}
enum RelaySkippedAttention {
  static let traceEnabled = ProcessInfo.processInfo.environment["WEETODD_UPSCALER_TRACE"] == "1"
  static func trace(custom: Bool, safe: MLXArray) {
    if traceEnabled {
      FileHandle.standardError.write(Data("Relay attention custom=\(custom) safe=\(safe.item(Bool.self))\n".utf8))
    }
  }
  static var runtimeQualified: Bool {
    Device.defaultDevice().deviceType == .gpu && GPU.deviceInfo().architecture == "applegpu_g15d"
  }

  private static let kernel = MLXFast.metalKernel(name: "weetodd_relay_mask_blocks_v1",
    inputNames: ["q", "k", "v", "fine_mask", "block_types", "cfg", "scale_value", "allow_skipping"],
    outputNames: ["output"], source: """
      using namespace mlx::steel;
      using MaskType = bool; using AccumType = float;
      constexpr int BQ=32, BK=BD<128 ? 32:16, WM=4, WN=1;
      const int QUERY_ROWS=cfg[0], KEY_ROWS=cfg[1], HEADS=cfg[2], KV_HEADS=cfg[3];
      const bool align_Q=(QUERY_ROWS%BQ)==0, align_K=(KEY_ROWS%BK)==0;
      constexpr bool has_mask=true, do_causal=false, has_sinks=false;
      const device T* Q=q; const device T* K=k; const device T* V=v; device T* O=output;
      AttnParams params_value{
        1,HEADS,BD,QUERY_ROWS,KEY_ROWS,HEADS/KV_HEADS,0,
        (QUERY_ROWS+BQ-1)/BQ,(KEY_ROWS+BK-1)/BK,QUERY_ROWS/BQ,KEY_ROWS/BK,
        QUERY_ROWS%BQ,KEY_ROWS%BK,KEY_ROWS-QUERY_ROWS,
        {0,BD,HEADS*BD},{0,BD,KV_HEADS*BD},{0,BD,KV_HEADS*BD},{0,BD,HEADS*BD}
      };
      params_value.scale=scale_value;
      thread const AttnParams* params=&params_value;
      AttnMaskParams mask_value{{0,0,KEY_ROWS}};
      thread const AttnMaskParams* mask_params=&mask_value;
      const device bool* mask=fine_mask; const device T* sinks=nullptr;
      uint simd_lane_id=thread_index_in_simdgroup;
      uint simd_group_id=simdgroup_index_in_threadgroup;
      uint3 tid=threadgroup_position_in_grid,lid=thread_position_in_threadgroup;
      """ + RelaySkippedSteelSource.body, header: NativeSteelPrimitives.header)

  /// A device scalar is shared by every spatial tile; no host synchronization.
  static func safeInputs(query: MLXArray, key: MLXArray, value: MLXArray) -> MLXArray {
    RelayInputSafety.evaluate(query, key, value)
  }

  static func apply(query: MLXArray, key: MLXArray, value: MLXArray, mask: MLXArray,
    classes: MLXArray, allowSkipping: MLXArray) throws -> MLXArray {
    try Task.checkCancellation()
    guard query.ndim == 3, key.ndim == 3, key.shape == value.shape,
      query.shape.allSatisfy({ $0 > 0 }), key.shape.allSatisfy({ $0 > 0 }),
      query.dim(1) % key.dim(1) == 0, query.dim(2) == key.dim(2),
      [64, 128].contains(query.dim(2)), [DType.bfloat16, .float32].contains(query.dtype),
      query.dtype == key.dtype, query.dtype == value.dtype,
      mask.dtype == .bool, mask.shape == [query.dim(0), key.dim(0)],
      classes.dtype == .uint8, allowSkipping.dtype == .bool, allowSkipping.size == 1 else {
      throw RelayVSRError.invalid("Invalid Relay block-skipping attention tensors.")
    }
    let d = query.dim(2), bk = d == 64 ? 32 : 16
    guard classes.shape == [(query.dim(0) + 31) / 32, (key.dim(0) + bk - 1) / bk],
      query.size <= Int(Int32.max), key.size <= Int(Int32.max) else {
      throw RelayVSRError.invalid("Invalid Relay block-skipping metadata.")
    }
    return graph(query: query, key: key, value: value, mask: mask, classes: classes,
      allowSkipping: allowSkipping, scale: MLXArray(1 / sqrt(Float(d))))
  }

  // Only validated tensor/plan geometry reaches this nonthrowing graph boundary.
  static func graph(query: MLXArray, key: MLXArray, value: MLXArray, mask: MLXArray,
    classes: MLXArray, allowSkipping: MLXArray, scale: MLXArray) -> MLXArray {
    let d = query.dim(2)
    let cfg = MLXArray([query.dim(0), key.dim(0), query.dim(1), key.dim(1)].map(Int32.init))
    return kernel([query, key, value, mask, classes, cfg, scale, allowSkipping],
      template: [("T", query.dtype), ("BD", d)],
      grid: (((query.dim(0) + 31) / 32) * 32, query.dim(1) * 4, 1), threadGroup: (32, 4, 1),
      outputShapes: [query.shape], outputDTypes: [query.dtype])[0]
  }
}
