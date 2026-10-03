// Uses the exact released dense inverse basis and BF16 oracle; no candidate math.
import CoreFoundation
import Darwin
import Foundation
import MLX
import TensorIO
import XCTest
@testable import H3MLX

final class H3QKVDecodeBoundaryProbeTests: XCTestCase {
  func testInstalledQKVCPUCopyAndGPUDecodeBoundaries() throws {
    let environment = ProcessInfo.processInfo.environment
    guard environment["WEETODD_H3_DECODE_BOUNDARY_PROBE"] == "1",
      let source = environment["WEETODD_H3_TEST_CHECKPOINT"] else {
      throw XCTSkip("Explicit installed-checkpoint diagnostic; run only after the render pipeline is idle.")
    }
    let url = URL(fileURLWithPath: source), rows = 21504, columns = 5376
    let layout = try H3CheckpointLayout(url: url)
    guard layout.curveRank == nil, layout.quantizedProjections == 250 else {
      throw H3CheckpointError.invalid("Decode boundaries require the full-width Comfy INT8 checkpoint.")
    }
    func now() -> UInt64 { DispatchTime.now().uptimeNanoseconds }
    func elapsed(_ start: UInt64) -> Double { Double(now() - start) / 1_000_000_000 }
    let fileStart = now(), file = try SafeTensorFile(url: url)
    let openAndHeaderSeconds = elapsed(fileStart)
    let name = layout.prefix + "blocks.0.attn.qkv_proj.weight"
    guard let tensor = file.tensors[name], tensor.dtype == "I8",
      tensor.shape == [UInt64(rows), UInt64(columns)] else {
      throw H3CheckpointError.invalid("This diagnostic requires released full-width block-0 QKV INT8.")
    }
    let stem = String(name.dropLast(".weight".count))
    let metadataStart = now()
    guard let markerTensor = file.tensors[stem + ".comfy_quant"],
      markerTensor.dtype == "U8", markerTensor.byteCount > 0, markerTensor.byteCount <= 4096 else {
      throw H3CheckpointError.invalid("Invalid released quantization marker.")
    }
    let marker = try file.withTensorBytes(named: stem + ".comfy_quant") { Data($0) }
    guard let scaleTensor = file.tensors[stem + ".weight_scale"],
      scaleTensor.dtype == "F32", scaleTensor.shape == [UInt64(rows), 1],
      let dictionary = try JSONSerialization.jsonObject(with: marker) as? [String: Any],
      Set(dictionary.keys) == ["format", "convrot", "convrot_groupsize"],
      dictionary["format"] as? String == "int8_tensorwise",
      let rotated = dictionary["convrot"] as? NSNumber,
      CFGetTypeID(rotated) == CFBooleanGetTypeID(), rotated.boolValue,
      let group = dictionary["convrot_groupsize"] as? NSNumber,
      CFGetTypeID(group) != CFBooleanGetTypeID(), group.doubleValue == 256 else {
      throw H3CheckpointError.invalid("The diagnostic does not substitute another ConvRot layout.")
    }
    let scales = try file.readFloat32(named: stem + ".weight_scale")
    guard scales.count == rows, scales.allSatisfy({ $0.isFinite && $0 > 0 }) else {
      throw H3CheckpointError.invalid("Invalid released scales.")
    }
    let metadataSeconds = elapsed(metadataStart)
    var phases: [[String: Any]] = []
    var boundaries: [[String: Any]] = []
    @discardableResult
    func boundary(_ label: String, pass: Int = -1, range: Range<Int> = 0..<rows,
      clear: Bool = true) -> [String: Any] {
      let cacheBefore = Memory.cacheMemory, activeBefore = Memory.activeMemory
      let start = now()
      Stream.gpu.synchronize()
      let synchronizeSeconds = elapsed(start)
      let cacheAfterSynchronize = Memory.cacheMemory
      let clearStart = now()
      if clear { Memory.clearCache() }
      let clearSeconds = clear ? elapsed(clearStart) : 0
      let record: [String: Any] = ["boundary": label, "pass": pass,
        "startRow": range.lowerBound, "rowCount": range.count,
        "synchronizeSeconds": synchronizeSeconds, "clearExecuted": clear,
        "clearSeconds": clearSeconds, "totalSeconds": elapsed(start),
        "cacheMLXBytesBefore": cacheBefore,
        "cacheMLXBytesAfterSynchronize": cacheAfterSynchronize,
        "cacheMLXBytesAfter": Memory.cacheMemory,
        "activeMLXBytesBefore": activeBefore, "activeMLXBytesAfter": Memory.activeMemory]
      boundaries.append(record)
      return record
    }
    let oldCache = Memory.cacheLimit
    Memory.cacheLimit = 128 * 1024 * 1024
    defer {
      // This executes after the summary, and also on failure. Keep its cost
      // visible as a separate record rather than silently excluding teardown.
      let final = boundary("scope_exit_cleanup")
      if let bytes = try? JSONSerialization.data(withJSONObject: final, options: [.sortedKeys]) {
        print("H3_DECODE_BOUNDARY_FINAL_CLEANUP " + String(decoding: bytes, as: UTF8.self))
      }
      Memory.cacheLimit = oldCache
    }
    boundary("initial_cleanup")
    let residentBefore = Memory.activeMemory
    func sample(_ phase: String, pass: Int, range: Range<Int>,
      _ operation: () throws -> MLXArray) rethrows -> MLXArray {
      boundary("before_" + phase, pass: pass, range: range)
      let activeBefore = Memory.activeMemory
      Memory.peakMemory = activeBefore
      let start = now(), value = try operation()
      eval(value)
      boundary("finish_" + phase, pass: pass, range: range, clear: false)
      phases.append(["phase": phase, "pass": pass, "startRow": range.lowerBound,
        "rowCount": range.count, "seconds": elapsed(start),
        "activeMLXBytesBefore": activeBefore, "activeMLXBytesAfter": Memory.activeMemory, "peakMLXBytes": Memory.peakMemory])
      return value
    }
    // Three bounded warm-order passes over exactly two released row windows.
    // The first pass is natural-cache state, not claimed cold. No cache purge.
    for pass in 0..<3 {
      try autoreleasepool {
        let basis = sample("convrot_basis_creation_and_eval", pass: pass, range: 0..<rows) {
          H3ComfyDecodedProjection.basis(group: 256)
        }
        var parts: [MLXArray] = []
        for range in [0..<16384, 16384..<21504] {
          let part = try autoreleasepool { () throws -> MLXArray in
            let lower = UInt64(range.lowerBound * columns), upper = UInt64(range.upperBound * columns)
            // Read/touch/copy actual file pages into owned host memory. This includes
            // page faults, mapping/checks, and host copy, without a GPU graph.
            var usageBefore = rusage(), usageAfter = rusage()
            guard getrusage(RUSAGE_SELF, &usageBefore) == 0 else {
              throw H3CheckpointError.invalid("Cannot read diagnostic resource counters.")
            }
            let readStart = now()
            let owned = try file.withTensorBytes(named: name, range: lower..<upper) { Data($0) }
            guard getrusage(RUSAGE_SELF, &usageAfter) == 0 else {
              throw H3CheckpointError.invalid("Cannot read diagnostic resource counters.")
            }
            phases.append(["phase": "mmap_to_owned_host_copy", "pass": pass,
              "startRow": range.lowerBound, "rowCount": range.count,
              "sourceBytes": owned.count, "seconds": elapsed(readStart),
              "minorFaultDelta": usageAfter.ru_minflt - usageBefore.ru_minflt,
              "majorFaultDelta": usageAfter.ru_majflt - usageBefore.ru_majflt,
              "inputBlockCounterDelta": usageAfter.ru_inblock - usageBefore.ru_inblock])
            // The actual payload is now resident host bytes. MLX's byte initializer
            // allocates and copies; the source mmap lifetime does not escape.
            let quantized = sample("resident_host_to_owned_mlx_copy", pass: pass, range: range) {
              MLXArray(owned, [range.count, columns], type: Int8.self)
            }
            let scale = MLXArray(Array(scales[range]), [range.count, 1])
            eval(scale)
            boundary("scale_array_ready", pass: pass, range: range, clear: false)
            let decoded = sample("gpu_i8_to_f32_row_scale", pass: pass, range: range) {
              quantized.asType(.float32) * scale
            }
            let restored = sample("gpu_dense_convrot_f32", pass: pass, range: range) {
              matmul(decoded.reshaped([-1, 256]), basis).reshaped([range.count, columns])
            }
            let split = sample("gpu_f32_to_bf16", pass: pass, range: range) { restored.asType(.bfloat16) }
            // Exact released decoder oracle, outside timed subphases. No new tolerance.
            let oracleStart = now()
            let oracle = try H3ComfyDecodedProjection.decodeRows(checkpointURL: url,
              name: name, rows: rows, columns: columns, range: range)
            phases.append(["phase": "released_decode_rows_oracle_aggregate", "pass": pass,
              "startRow": range.lowerBound, "rowCount": range.count,
              "seconds": elapsed(oracleStart),
              "scope": "Released row decoder call including file/header/metadata, mapped copy and eval; this row API has no explicit deferred synchronization/cache clear."])
            XCTAssertEqual(max(abs(split.asType(.float32) - oracle.asType(.float32))).item(Float.self), 0)
            // Fused/evaluated graph from the resident source: barriers can affect
            // fusion, so these timings must not be added as a production estimate.
            let residentCombined = sample("resident_combined_decode_graph", pass: pass, range: range) {
              matmul((quantized.asType(.float32) * scale).reshaped([-1, 256]), basis)
                .reshaped([range.count, columns]).asType(.bfloat16)
            }
            XCTAssertEqual(max(abs(residentCombined.asType(.float32) - oracle.asType(.float32))).item(Float.self), 0)
            // Production mmap->MLX constructor control. Timing includes read faults,
            // map/check overhead, allocation and copy; source is already cache-touched.
            let direct = try sample("warm_direct_mmap_to_owned_mlx_copy", pass: pass, range: range) {
              try file.withTensorBytes(named: name, range: lower..<upper) {
                MLXArray($0, [range.count, columns], type: Int8.self)
              }
            }
            XCTAssertTrue((direct .== quantized).all().item(Bool.self))
            return split
          }
          parts.append(part)
          boundary("row_window_release", pass: pass, range: range)
        }
        let assembled = sample("gpu_concat_and_qkv_reorder", pass: pass, range: 0..<rows) {
          concatenated(parts, axis: 0).reshaped([3, 56, 128, columns])
            .transposed(1, 0, 2, 3).reshaped([rows, columns])
        }
        let releasedCacheBefore = Memory.cacheMemory
        let releasedStart = now()
        let released = try H3ComfyDecodedProjection.load(file: file, checkpointURL: url,
          name: name, rows: rows, columns: columns, reorderQKV: true, rowWindow: 16384)
        phases.append(["phase": "released_full_qkv_load_oracle_aggregate", "pass": pass,
          "startRow": 0, "rowCount": rows, "seconds": elapsed(releasedStart),
          "cacheMLXBytesBefore": releasedCacheBefore, "cacheMLXBytesAfter": Memory.cacheMemory,
          "scope": "Released aggregate call including its internal deferred synchronization/cache clear; no internal attribution."])
        XCTAssertEqual(max(abs(assembled.asType(.float32) - released.asType(.float32))).item(Float.self), 0)
      }
      boundary("pass_release", pass: pass)
    }
    try file.checkUnchanged(at: url)
    XCTAssertEqual(Memory.activeMemory, residentBefore)
    let result: [String: Any] = ["format": "weetodd-h3-decode-boundaries-v1",
      "source": source, "openHeaderSeconds": openAndHeaderSeconds,
      "metadataMarkerAndScalesSeconds": metadataSeconds, "phases": phases, "boundaries": boundaries,
      "checkpointBytes": file.fileByteCount, "int8PayloadBytes": tensor.byteCount,
      "rowWindows": [16384, 5120], "passes": 3, "convrotGroup": 256,
      "activeMLXBytesBefore": residentBefore, "activeMLXBytesAfterRelease": Memory.activeMemory,
      "cacheLimitBytes": 128 * 1024 * 1024,
      "timerScope": "Monotonic wall around operation, eval and finish synchronization. Pre-phase synchronization/cache clear is measured separately in boundaries, not silently charged to the phase. The released full-QKV load aggregate includes its internal deferred synchronization/clear.",
      "boundaryScope": "Diagnostic explicit GPU synchronize/cache-clear calls, with cache bytes before/after and separate monotonic times. The full-QKV load's internal deferred clear remains inside its aggregate timing, not attributed individually. Final scope-exit cleanup is emitted in H3_DECODE_BOUNDARY_FINAL_CLEANUP on success/failure after this summary.",
      "allocatorScope": "MLX active/peak only; peak reset to active resident bytes before each phase. Earlier outputs remain resident, so phase peaks are not isolated. Owned Data/VM/OS footprint excluded.",
      "faultCounterScope": "Process rusage deltas across mapped host copy, not isolated thread counters or physical disk read bytes. Natural filesystem cache not controlled.",
      "maximumAllowedBF16Error": 0, "scope": "one block-0 QKV payload; no sampling",
      "limitations": "Natural filesystem cache, fixed pass order, diagnostic eval barriers. CPU copy includes VM faults; block counter is not a byte counter. No additive runtime or isolated peak claim."]
    print("H3_DECODE_BOUNDARIES " + String(decoding: try JSONSerialization.data(withJSONObject: result, options: [.sortedKeys]), as: UTF8.self))
  }
}
