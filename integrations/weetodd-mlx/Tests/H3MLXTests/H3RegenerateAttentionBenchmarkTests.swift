import Foundation
import InferenceContracts
import MLX
import MLXRandom
import XCTest
@testable import H3MLX

/// Explicit diagnostics only. These tests never alter a production attention policy.
/// GPU tests require a bound frozen worker/Metal library and an exclusive GPU slot.
final class H3RegenerateAttentionBenchmarkTests: XCTestCase {
  private struct Candidate {
    let queryChunk: Int
    let headsPerGroup: Int
    var name: String { "q\(queryChunk)-h\(headsPerGroup)" }
  }
  private static let candidates = [
    Candidate(queryChunk: 256, headsPerGroup: 14),
    Candidate(queryChunk: 256, headsPerGroup: 56),
    Candidate(queryChunk: 1024, headsPerGroup: 14),
  ]

  /// Mirrors the existing per-SDPA-call conservative signed-32 admissions.
  /// Production projected() still caps queries at 256; this diagnostic calls attend().
  private static func admitted(queryRows: Int, keyRows: Int, candidate: Candidate) -> Bool {
    guard queryRows > 0, keyRows > 0, keyRows <= 390_000,
      (1...8192).contains(candidate.queryChunk),
      (1...56).contains(candidate.headsPerGroup),
      56.isMultiple(of: candidate.headsPerGroup) else { return false }
    let h = Int64(candidate.headsPerGroup), q = Int64(min(queryRows, candidate.queryChunk))
    let k = Int64(keyRows), limit = Int64(Int32.max)
    return h * q * 128 < limit && h * k * 128 < limit && h * q * k < limit
  }

  func testCPUAdmissionKeepsFull2KCandidatesAndMaximumRowFallbackSeparate() {
    for candidate in Self.candidates {
      XCTAssertTrue(Self.admitted(queryRows: 4096, keyRows: 133_727, candidate: candidate))
    }
    XCTAssertTrue(Self.admitted(queryRows: 4096, keyRows: 390_000, candidate: Self.candidates[0]))
    XCTAssertFalse(Self.admitted(queryRows: 4096, keyRows: 390_000, candidate: Self.candidates[1]))
    XCTAssertFalse(Self.admitted(queryRows: 4096, keyRows: 390_000, candidate: Self.candidates[2]))
    XCTAssertFalse(Self.admitted(queryRows: 0, keyRows: 133_727, candidate: Self.candidates[0]))
  }

  private func bindGPU() throws {
    let environment = ProcessInfo.processInfo.environment
    let worker = try XCTUnwrap(environment["WEETODD_H3_REGENERATE_ATTENTION_WORKER"],
      "Bind the frozen native worker whose adjacent Metal library matches this test build.")
    GPU.metallib = try NativeMLXMetalLibrary.resolve(executable: URL(fileURLWithPath: worker))
    guard Device.defaultDevice().deviceType == .gpu else { throw XCTSkip("Metal GPU required.") }
  }

  private func inputs(queryRows: Int, keyRows: Int) -> (MLXArray, MLXArray, MLXArray) {
    // Counter-based MLX keys avoid allocating enormous host-side integer ranges.
    let q = (MLXRandom.normal([1,56,queryRows,128], key: MLXRandom.key(841)) * 0.25).asType(.bfloat16)
    let k = (MLXRandom.normal([1,56,keyRows,128], key: MLXRandom.key(842)) * 0.25).asType(.bfloat16)
    let v = (MLXRandom.normal([1,56,keyRows,128], key: MLXRandom.key(843)) * 0.5).asType(.bfloat16)
    eval(q,k,v)
    return (q,k,v)
  }

  /// Query-major/head-group-minor order matches projected() without QKV or out_proj.
  private func evaluate(_ candidate: Candidate, q: MLXArray, k: MLXArray, v: MLXArray) throws -> MLXArray {
    guard Self.admitted(queryRows: q.shape[2], keyRows: k.shape[2], candidate: candidate),
      q.shape[0] == 1, q.shape[1] == 56, q.shape[3] == 128,
      k.shape == [1,56,k.shape[2],128], v.shape == k.shape,
      q.dtype == .bfloat16, k.dtype == .bfloat16, v.dtype == .bfloat16 else {
      throw H3CheckpointError.invalid("Attention diagnostic exceeds its exact admitted geometry.")
    }
    var queryParts: [MLXArray] = []
    for start in stride(from: 0, to: q.shape[2], by: candidate.queryChunk) {
      let rows = start..<min(start + candidate.queryChunk, q.shape[2])
      var headParts: [MLXArray] = []
      for head in stride(from: 0, to: 56, by: candidate.headsPerGroup) {
        let heads = head..<(head + candidate.headsPerGroup)
        let output = try H3RegenerateAttention.attend(
          query: q[0..<1,heads,rows,0..<128],
          keys: k[0..<1,heads,0..<k.shape[2],0..<128],
          values: v[0..<1,heads,0..<v.shape[2],0..<128], queryChunk: candidate.queryChunk)
        headParts.append(output)
      }
      let part = concatenated(headParts, axis: 1)
      eval(part)
      queryParts.append(part)
    }
    let result = concatenated(queryParts, axis: 2)
    eval(result)
    return result
  }

  private func parity(_ actual: MLXArray, _ reference: MLXArray) -> [String: Any] {
    let a = actual.asType(.float32), r = reference.asType(.float32), delta = a-r
    let maximum = max(abs(delta)).item(Float.self)
    let squaredError = sum(delta*delta).item(Float.self)
    let squaredReference = sum(r*r).item(Float.self)
    let relativeL2 = (squaredError / Swift.max(squaredReference, Float.leastNonzeroMagnitude)).squareRoot()
    let finite = all(isFinite(actual)).item(Bool.self)
    let allClose = all(abs(delta) .<= (Float(1.0/256.0) + Float(0.02)*abs(r))).item(Bool.self)
    XCTAssertEqual(actual.shape, reference.shape)
    XCTAssertTrue(finite)
    XCTAssertTrue(allClose, "Every value must satisfy atol=1/256, rtol=0.02.")
    XCTAssertLessThanOrEqual(maximum, Float(1.0/256.0))
    XCTAssertLessThanOrEqual(relativeL2, 0.02)
    return ["finite":finite,"allClose":allClose,"maximumAbsoluteDifference":maximum,
      "relativeL2":relativeL2,"bitIdentical":maximum == 0]
  }

  private func record(_ value: [String: Any]) throws {
    let data = try JSONSerialization.data(withJSONObject: value, options: [.sortedKeys])
    print("H3_REGENERATE_ATTENTION_DIAGNOSTIC \(String(decoding:data,as:UTF8.self))")
    fflush(stdout)
  }

  func testOptInNontrivialBF16D128FusedParity() throws {
    let environment = ProcessInfo.processInfo.environment
    guard environment["WEETODD_H3_REGENERATE_ATTENTION_PARITY"] == "1"
      || environment["WEETODD_H3_REGENERATE_ATTENTION_BENCH"] == "1" else {
      throw XCTSkip("Enable explicit BF16 fused attention parity after the GPU slot is released.")
    }
    try bindGPU()
    defer { Stream.gpu.synchronize(); Memory.clearCache() }
    // Ragged Q/K lengths exercise the actual BF16 D128 fused kernel and its tails.
    let (q,k,v) = inputs(queryRows: 545, keyRows: 769)
    let reference = MLXFast.scaledDotProductAttention(queries:q,keys:k,values:v,
      scale:1/Float(128).squareRoot(),mask:nil,forceFused:true)
    eval(reference)
    for candidate in Self.candidates {
      let result = try evaluate(candidate, q:q,k:k,v:v)
      var metrics = parity(result, reference)
      metrics["event"] = "small-fused-parity"; metrics["candidate"] = candidate.name
      metrics["queryRows"] = 545; metrics["keyRows"] = 769
      try record(metrics)
    }
  }

  func testOptInFull2KKeyLengthBoundedQueryBenchmark() throws {
    let environment = ProcessInfo.processInfo.environment
    guard environment["WEETODD_H3_REGENERATE_ATTENTION_BENCH"] == "1" else {
      throw XCTSkip("Enable the explicit model-free benchmark only with exclusive GPU ownership.")
    }
    let repetitions = Int(environment["WEETODD_H3_REGENERATE_ATTENTION_REPETITIONS"] ?? "3") ?? 0
    guard (1...7).contains(repetitions) else { throw H3CheckpointError.invalid("Benchmark repetitions must be 1...7.") }
    try bindGPU()
    let priorLimit = Memory.cacheLimit
    Memory.cacheLimit = Swift.min(priorLimit,128*1024*1024)
    defer { Stream.gpu.synchronize(); Memory.clearCache(); Memory.cacheLimit = priorLimit }
    let queries = 4096, keys = 133_727
    let (q,k,v) = inputs(queryRows: queries, keyRows: keys)
    Stream.gpu.synchronize(); Memory.clearCache()
    try record(["event":"begin","modelFree":true,"productionSettingsChanged":false,
      "queryRows":queries,"keyRows":keys,"heads":56,"headWidth":128,"dtype":"BF16",
      "seeds":[841,842,843],"repetitions":repetitions,"allKeysVisible":true,
      "includesQKVProjection":false,"includesOutputProjection":false,
      "includesSourceEncoding":false,"includesDecoding":false,
      "physicalMemoryBytes":ProcessInfo.processInfo.physicalMemory,
      "operatingSystem":ProcessInfo.processInfo.operatingSystemVersionString,
      "cacheLimitBytes":Memory.cacheLimit,"device":String(describing:GPU.deviceInfo()),
      "metallib":GPU.metallib?.path ?? "unavailable"])
    // One fully completed warm run per shape. Fixed baseline survives all candidates.
    let reference = try evaluate(Self.candidates[0], q:q,k:k,v:v)
    for candidate in Self.candidates.dropFirst() {
      try autoreleasepool {
        let output = try evaluate(candidate, q:q,k:k,v:v)
        _ = parity(output,reference)
      }
    }
    for iteration in 0..<repetitions {
      // Rotate order to expose thermal/order effects rather than favoring one candidate.
      for offset in 0..<Self.candidates.count {
        let candidate = Self.candidates[(iteration+offset)%Self.candidates.count]
        try autoreleasepool {
          Stream.gpu.synchronize(); Memory.clearCache()
          let activeBefore = Memory.activeMemory
          Memory.peakMemory = activeBefore
          let started = DispatchTime.now().uptimeNanoseconds
          let output = try evaluate(candidate, q:q,k:k,v:v)
          Stream.gpu.synchronize()
          let seconds = Double(DispatchTime.now().uptimeNanoseconds-started)/1e9
          let peak = Memory.peakMemory // Comparison scratch is excluded from this peak.
          var metrics = parity(output,reference)
          metrics["event"] = "warm-complete"; metrics["iteration"] = iteration
          metrics["candidate"] = candidate.name; metrics["seconds"] = seconds
          metrics["activeBytesBefore"] = activeBefore; metrics["peakMLXBytes"] = peak
          metrics["transientPeakBytes"] = Swift.max(0,peak-activeBefore)
          let queryTiles = (queries+candidate.queryChunk-1)/candidate.queryChunk
          metrics["sdpaCalls"] = queryTiles*(56/candidate.headsPerGroup)
          metrics["nonNAXThreadgroupsPerFullCall"] = ((candidate.queryChunk+31)/32)*candidate.headsPerGroup
          metrics["full2KScoreIndexProduct"] = Int64(candidate.queryChunk)*Int64(candidate.headsPerGroup)*Int64(keys)
          XCTAssertLessThanOrEqual(peak,16*1024*1024*1024)
          try record(metrics)
        }
      }
    }
  }

  /// Diagnostic-only bypass of the logical score product. The pinned Metal
  /// BF16 D128 fused kernel does not allocate or address a full QK matrix.
  /// Production attend()/projected() caps are deliberately not changed here.
  func testOptInFull2KForcedFusedWideQueries() throws {
    let environment=ProcessInfo.processInfo.environment
    guard environment["WEETODD_H3_REGENERATE_FORCED_ATTENTION_BENCH"] == "1" else {
      throw XCTSkip("Explicit pinned non-NAX BF16 D128 forced-fused diagnostic only.")
    }
    try bindGPU()
    let device=GPU.deviceInfo()
    guard Device.defaultDevice().deviceType == .gpu,device.architecture == "applegpu_g15d" else {
      throw XCTSkip("This indexing qualification targets the pinned M3 Ultra non-NAX kernel.")
    }
    let priorLimit=Memory.cacheLimit
    Memory.cacheLimit=Swift.min(priorLimit,128*1024*1024)
    defer { Stream.gpu.synchronize();Memory.clearCache();Memory.cacheLimit=priorLimit }
    let queries=8192,keys=133_727,chunks=[256,1024,4096,8192]
    let (q,k,v)=inputs(queryRows:queries,keyRows:keys)
    for tensor in [q,k,v] {
      let strides=tensor.strides,rows=tensor.shape[2]
      guard tensor.ndim == 4,tensor.shape[0] == 1,tensor.shape[1] == 56,tensor.shape[3] == 128,
        tensor.dtype == .bfloat16,tensor.size < Int(Int32.max),tensor.nbytes <= device.maxBufferSize,
        strides.count == 4,strides.allSatisfy({ $0 > 0 }),strides[3] == 1,strides[2] == 128,
        strides[1] == rows*128,
        55*Int64(strides[1])+Int64(rows-1)*Int64(strides[2])+127*Int64(strides[3])+1 == Int64(tensor.size) else {
        throw H3CheckpointError.invalid("Forced diagnostic requires owned canonical BF16 D128 arrays within real storage bounds.")
      }
    }
    func forced(_ chunk:Int) throws -> (MLXArray,[Double]) {
      guard chunks.contains(chunk),q.shape == [1,56,8192,128],k.shape == [1,56,133_727,128],v.shape == k.shape else {
        throw H3CheckpointError.invalid("Forced diagnostic geometry changed after admission.")
      }
      var parts:[MLXArray]=[],calls:[Double]=[]
      for start in stride(from:0,to:queries,by:chunk) {
        try Task.checkCancellation()
        let range=start..<min(start+chunk,queries)
        let began=DispatchTime.now().uptimeNanoseconds
        let output=try MLX.withError { error in
          let value=MLXFast.scaledDotProductAttention(queries:q[0..<1,0..<56,range,0..<128],
            keys:k,values:v,scale:1/Float(128).squareRoot(),mask:nil,forceFused:true)
          try error.check() // Do not consume an invalid array after a forced-routing error.
          eval(value)
          try error.check()
          return value
        }
        calls.append(Double(DispatchTime.now().uptimeNanoseconds-began)/1e9)
        try Task.checkCancellation()
        parts.append(output)
      }
      let output=concatenated(parts,axis:2);eval(output)
      return (output,calls)
    }
    let reference=try evaluate(Candidate(queryChunk:256,headsPerGroup:56),q:q,k:k,v:v)
    try record(["event":"forced-begin","modelFree":true,"productionSettingsChanged":false,
      "queryRows":queries,"keyRows":keys,"heads":56,"headWidth":128,"dtype":"BF16",
      "forceFused":true,"noMask":true,"equalQueryKVHeads":true,"perArraySigned32":true,
      "canonicalOwnedStorage":true,"architecture":device.architecture,"cacheLimitBytes":Memory.cacheLimit,
      "maximumDeviceBufferBytes":device.maxBufferSize,"repetitions":3,
      "includesOutputProjection":false,"logicalScoreGuardBypass":"diagnostic only; fused has no global QK storage",
      "indexingAudit":"h3-fused-attention-indexing-audit.md"])
    for chunk in chunks {
      try autoreleasepool {
        let (output,_)=try forced(chunk)
        _=parity(output,reference)
      }
    }
    for iteration in 0..<3 {
      for offset in 0..<chunks.count {
        let chunk=chunks[(iteration+offset)%chunks.count]
        try autoreleasepool {
          Stream.gpu.synchronize();Memory.clearCache()
          let before=Memory.activeMemory
          Memory.peakMemory=before
          let began=DispatchTime.now().uptimeNanoseconds
          let (output,calls)=try forced(chunk)
          Stream.gpu.synchronize()
          let seconds=Double(DispatchTime.now().uptimeNanoseconds-began)/1e9,peak=Memory.peakMemory
          var metrics=parity(output,reference)
          metrics["event"]="forced-warm-complete";metrics["iteration"]=iteration
          metrics["queryChunk"]=chunk;metrics["headsPerGroup"]=56;metrics["forceFused"]=true
          metrics["seconds"]=seconds;metrics["sdpaCalls"]=calls.count
          metrics["callSeconds"]=calls;metrics["maximumCallSeconds"]=calls.max() ?? 0
          metrics["activeBytesBefore"]=before;metrics["peakMLXBytes"]=peak
          metrics["transientPeakBytes"]=Swift.max(0,peak-before)
          metrics["nonNAXThreadgroupsPerFullCall"]=(chunk/32)*56
          XCTAssertLessThanOrEqual(peak,16*1024*1024*1024)
          try record(metrics)
        }
      }
    }
  }
}
