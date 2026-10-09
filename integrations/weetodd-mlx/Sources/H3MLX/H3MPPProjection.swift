import Foundation
import MLX

/// Swift port of WeeTodd's MLX-owned BF16 MPP projection in
/// `src/minimax_h3_mlx/projection.py`. No external tensor ownership or host copy
/// is introduced. Callers must qualify the operation before enabling it: MLX
/// Metal compilation/execution failures are fatal, not Swift catchable errors.
enum H3MPPProjection {
  /// Task qualification is independent of prepared-weight ownership. Direct
  /// state callers and unqualified task contexts retain standard MLX.
  enum ExecutionTask { case t2va, ref2va, fl2va, continuation, refinement, motionFidelity }

  static func isTaskEligible(_ task: ExecutionTask, contextFrames: Int = 0,
    isRefinement: Bool) -> Bool {
    guard contextFrames == 0, !isRefinement else { return false }
    switch task {
    case .t2va, .ref2va, .fl2va: return true
    case .continuation, .refinement, .motionFidelity: return false
    }
  }

  struct Tile: Equatable {
    let rows: Int
    let columns: Int
    let simdgroups: Int

    private init(rows: Int, columns: Int, simdgroups: Int) {
      self.rows = rows
      self.columns = columns
      self.simdgroups = simdgroups
    }

    static let standard = Tile(rows: 32, columns: 64, simdgroups: 2)
    static let feedForwardOutput = Tile(rows: 64, columns: 128, simdgroups: 8)
  }

  struct Statistics {
    let verified: Int
    let fallback: Int
    let eligibleCalls: Int
    let mppCalls: Int
    let knownFallbackCalls: Int
    let firstUseReferenceCalls: Int
  }

  private final class Verification: @unchecked Sendable {
    private let lock = NSLock()
    private var verdicts: [String: [[Int]: Bool]] = [:]
    private struct Calls {
      var eligible = 0, mpp = 0, knownFallback = 0, firstUseReference = 0
    }
    private var calls: [String: Calls] = [:]

    func project(scope: String, signature: [Int], reference: () -> MLXArray,
      candidate: () -> MLXArray) throws -> MLXArray {
      lock.lock(); defer { lock.unlock() }
      try Task.checkCancellation()
      // Reuse the verification lock: no extra synchronization or tensor evaluation.
      calls[scope, default: Calls()].eligible += 1
      if let exact = verdicts[scope]?[signature] {
        if exact {
          calls[scope, default: Calls()].mpp += 1
          return candidate()
        }
        calls[scope, default: Calls()].knownFallback += 1
        return reference()
      }
      calls[scope, default: Calls()].firstUseReference += 1
      let ordinary = reference(), proposed = candidate()
      eval(ordinary, proposed)
      let exact = H3MPPProjection.firstUseMatches(reference: ordinary, candidate: proposed)
      verdicts[scope, default: [:]][signature] = exact
      // The first use always returns the established MLX result, including
      // signatures rejected for different floating-point reduction rounding.
      try Task.checkCancellation()
      return ordinary
    }

    func status(scope: String) -> Statistics {
      lock.lock(); defer { lock.unlock() }
      let values = Array(verdicts[scope]?.values ?? [:].values)
      let count = calls[scope] ?? Calls()
      return Statistics(verified: values.filter { $0 }.count,
        fallback: values.filter { !$0 }.count, eligibleCalls: count.eligible,
        mppCalls: count.mpp, knownFallbackCalls: count.knownFallback,
        firstUseReferenceCalls: count.firstUseReference)
    }

    func forget(scope: String) {
      lock.lock(); defer { lock.unlock() }
      verdicts.removeValue(forKey: scope)
      calls.removeValue(forKey: scope)
    }
  }

  /// Bitwise BF16 comparison preserves the sign of zero and every stored bit.
  /// This empirical sample qualifies one geometry on its first operands only.
  static func firstUseMatches(reference: MLXArray, candidate: MLXArray) -> Bool {
    guard reference.dtype == .bfloat16, candidate.dtype == .bfloat16,
      reference.shape == candidate.shape else { return false }
    return all(reference.view(dtype: .uint16) .== candidate.view(dtype: .uint16)).item(Bool.self)
  }

  private static let verification = Verification()
  static func verificationStatus(scope: String) -> Statistics {
    verification.status(scope: scope)
  }
  static func forget(scope: String) { verification.forget(scope: scope) }

  /// Only M3 Ultra has a measured complete-generation BF16 MPP qualification.
  /// This pure check does not create a Metal device or compile any kernel.
  static func isEligible(macOSMajor: Int, architecture: String, isGPU: Bool,
    sourceDType: DType, weightDType: DType) -> Bool {
    macOSMajor >= 26 && architecture.lowercased() == "applegpu_g15d"
      && isGPU && sourceDType == .bfloat16 && weightDType == .bfloat16
  }

  static var isAvailable: Bool {
    #if os(macOS)
    guard Device.defaultDevice().deviceType == .gpu else { return false }
    return isEligible(macOSMajor: ProcessInfo.processInfo.operatingSystemVersion.majorVersion,
      architecture: GPU.deviceInfo().architecture, isGPU: true,
      sourceDType: .bfloat16, weightDType: .bfloat16)
    #else
    return false
    #endif
  }

  static func tile(weightShape: [Int]) -> Tile {
    weightShape == [5376, 14336] ? .feedForwardOutput : .standard
  }

  /// Compute source @ weight.T. Unsupported devices, precision and unqualified
  /// matrix widths retain standard MLX. `enabled` is an explicit caller policy;
  /// this helper does not change any sampler's optimization default.
  static func apply(source: MLXArray, weight: MLXArray,
    enabled: Bool, verificationScope: String = "standalone") throws -> MLXArray {
    guard source.ndim >= 2, weight.ndim == 2,
      source.shape.allSatisfy({ $0 > 0 }), weight.shape.allSatisfy({ $0 > 0 }),
      source.shape.last == weight.shape[1] else {
      throw H3CheckpointError.invalid("MPP projection requires positive matrix dimensions and matching input widths.")
    }
    let inputWidth = weight.shape[1], outputWidth = weight.shape[0]
    // The descriptor uses int dimensions. Bound flattened addressing before
    // allocating or submitting work rather than overflowing its Metal indices.
    var rows = 1
    for dimension in source.shape.dropLast() {
      let product = rows.multipliedReportingOverflow(by: dimension)
      guard !product.overflow, product.partialValue <= Int(Int32.max) else {
        throw H3CheckpointError.invalid("MPP projection row count exceeds supported indexing.")
      }
      rows = product.partialValue
    }
    guard [inputWidth, outputWidth].allSatisfy({ $0 <= Int(Int32.max) }),
      rows <= Int(Int32.max) / max(inputWidth, outputWidth),
      outputWidth <= Int(Int32.max) / inputWidth else {
      throw H3CheckpointError.invalid("MPP projection matrix addressing exceeds supported indexing.")
    }
    try Task.checkCancellation()
    // The installed H3 BF16 projections have aligned K/N dimensions. Ragged
    // row counts are handled by MPP tensor extents, including the final tile.
    guard enabled, isAvailable, source.dtype == .bfloat16,
      weight.dtype == .bfloat16, [[21504,5376],[5376,7168],[28672,5376],[5376,14336]].contains(weight.shape) else {
      return matmul(source, weight.T)
    }
    let selected = tile(weightShape: weight.shape)
    let threads = 32 * selected.simdgroups
    let outputShape = Array(source.shape.dropLast()) + [outputWidth]
    let signature = [rows,inputWidth,outputWidth,selected.rows,selected.columns,selected.simdgroups]
    return try verification.project(scope:verificationScope, signature:signature,
      reference: { matmul(source,weight.T) }, candidate: {
      kernel([contiguous(source), contiguous(weight)],
      template: [("ROWS", rows), ("OUTPUT_DIM", outputWidth), ("INPUT_DIM", inputWidth),
        ("TILE_M", selected.rows), ("TILE_N", selected.columns),
        ("SIMDGROUPS", selected.simdgroups)],
      grid: (((outputWidth + selected.columns - 1) / selected.columns) * threads,
        (rows + selected.rows - 1) / selected.rows, 1),
      threadGroup: (threads, 1, 1), outputShapes: [outputShape],
      outputDTypes: [.bfloat16])[0]
    })
  }

  private static let kernel = MLXFast.metalKernel(
    name: "weetodd_h3_mpp_bf16_nt_matmul", inputNames: ["source", "weight"],
    outputNames: ["output"], source: """
      // MPP requires mutable element types but does not modify either input.
      auto matrix_a = tensor(
          (device bfloat*)source,
          dextents<int, 2>{INPUT_DIM, ROWS},
          array<int, 2>{1, INPUT_DIM});
      auto matrix_b = tensor(
          (device bfloat*)weight,
          dextents<int, 2>{INPUT_DIM, OUTPUT_DIM},
          array<int, 2>{1, INPUT_DIM});
      auto matrix_c = tensor(
          (device bfloat*)output,
          dextents<int, 2>{OUTPUT_DIM, ROWS},
          array<int, 2>{1, OUTPUT_DIM});

      constexpr auto descriptor = matmul2d_descriptor(
          TILE_M, TILE_N, static_cast<int>(dynamic_extent), false, true, false);
      matmul2d<descriptor, execution_simdgroups<SIMDGROUPS>> operation;
      auto tile_a = matrix_a.slice(0, threadgroup_position_in_grid.y * TILE_M);
      auto tile_b = matrix_b.slice(0, threadgroup_position_in_grid.x * TILE_N);
      auto tile_c = matrix_c.slice(
          threadgroup_position_in_grid.x * TILE_N,
          threadgroup_position_in_grid.y * TILE_M);
      auto result = operation.template get_destination_cooperative_tensor<
          decltype(tile_a), decltype(tile_b), bfloat>();
      #pragma unroll
      for (ushort index = 0; index < result.get_capacity(); ++index) {
        result[index] = bfloat(0.0f);
      }
      operation.run(tile_a, tile_b, result);
      result.store(tile_c);
      """, header: """
      #include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
      using namespace metal;
      using namespace mpp::tensor_ops;
      """)
}
