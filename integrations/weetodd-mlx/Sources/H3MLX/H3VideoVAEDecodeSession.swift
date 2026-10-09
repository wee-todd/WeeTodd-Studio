import Foundation
import MLX
import TensorIO

/// Packed decoder weights live only for one video decode stage. This is not a
/// process-wide model cache: close drops all arrays before the audio VAE starts.
final class H3VideoVAEDecodeSession {
  struct StageTensor {
    let name: String
    let shape: [Int]
    let dtype: String
    var byteCount: Int { shape.reduce(1,*) * (dtype == "U32" ? 4 : 2) }
  }

  /// Decoder-only admission. Encoder descriptors remain unrequested lazy handles.
  static func groupedStagePlan() -> [[StageTensor]] {
    var groups: [[StageTensor]] = [[
      StageTensor(name:"post_quant_conv.weight",shape:[24,1,1,1,24],dtype:"F16"),
      StageTensor(name:"post_quant_conv.bias",shape:[24],dtype:"F16"),
      StageTensor(name:"decoder.x_embedder.weight",shape:[2048,24],dtype:"F16"),
      StageTensor(name:"decoder.x_embedder.bias",shape:[2048],dtype:"F16"),
      StageTensor(name:"decoder.register_tokens",shape:[1,4,2048],dtype:"F16"),
      StageTensor(name:"decoder.norm_out.weight",shape:[2048],dtype:"F16"),
      StageTensor(name:"decoder.norm_out.bias",shape:[2048],dtype:"F16"),
      StageTensor(name:"decoder.proj_out.weight",shape:[3072,2048],dtype:"F16"),
      StageTensor(name:"decoder.proj_out.bias",shape:[3072],dtype:"F16"),
    ]]
    for index in 0..<36 {
      let prefix = "decoder.transformer_blocks.\(index)."
      var group = ["norm1.weight","norm2.weight","scale1","scale2"].map {
        StageTensor(name:prefix + $0,shape:[2048],dtype:"F16")
      }
      for (suffix,rows,columns) in [("attn.to_qkv",6144,2048),("attn.to_out",2048,2048),
        ("ff.w1",16384,2048),("ff.w2",2048,8192)] {
        group.append(StageTensor(name:prefix + suffix + ".weight",shape:[rows,columns / 4],dtype:"U32"))
        for companion in ["scales","biases"] {
          group.append(StageTensor(name:prefix + suffix + "." + companion,shape:[rows,columns / 64],dtype:"F16"))
        }
        group.append(StageTensor(name:prefix + suffix + ".bias",shape:[rows],dtype:"F16"))
      }
      groups.append(group)
    }
    return groups
  }

  static func validateGroupedStageAdmission(tensor: (String) -> H3TensorInfo?) throws -> Int {
    let plan = groupedStagePlan()
    for request in plan.flatMap({ $0 }) {
      try Task.checkCancellation()
      guard tensor(request.name) == H3TensorInfo(dtype:request.dtype,shape:request.shape.map(UInt64.init)) else {
        throw H3CheckpointError.invalid("Missing or changed grouped H3 video stage tensor: \(request.name)")
      }
    }
    return plan.reduce(0) { $0 + $1.reduce(0) { $0 + $1.byteCount } }
  }
  static let defaultAllocationCacheLimitBytes = 128 * 1024 * 1024
  static let experimentalLowAllocationCacheLimitBytes = 512 * 1024 * 1024

  static func validateAllocationCacheLimit(_ bytes: Int,
    memoryMode: H3VideoDecodeMemoryMode?) throws {
    guard bytes == defaultAllocationCacheLimitBytes
      || (memoryMode == .lowMemoryBF16 && bytes == experimentalLowAllocationCacheLimitBytes) else {
      throw H3CheckpointError.invalid("Invalid H3 video allocation cache policy.")
    }
  }
  struct GridKey: Hashable {
    let batch: Int
    let depth: Int
    let height: Int
    let width: Int
    let dtype: DType
  }
  struct Grid {
    let positions: MLXArray
    let rotary: H3VideoVAERotary
    var storageBytes: Int { positions.nbytes + rotary.cosine.nbytes + rotary.sine.nbytes }
  }
  static let maximumGridCacheBytes = 8 * 1024 * 1024
  private var grids: [GridKey: Grid] = [:]
  private(set) var gridCacheBytes = 0
  private var maximumGridCacheBytesObserved = 0
  var gridCacheEntries: Int { grids.count }
  struct Statistics {
    let projectionLoads: Int
    let tensorLoads: Int
    let maximumResidentBytes: Int
    let closed: Bool
    let remainingResidentBytes: Int
    let maximumGridCacheBytes: Int
    let allocationCacheLimitBytes: Int
    let groupedPreparationGroups: Int
    let computePrecision: String
  }
  let memoryMode: H3VideoDecodeMemoryMode?
  let precision: H3VideoDecodePrecision
  let allocationCacheLimitBytes: Int
  let checkpointURL: URL
  private let file: SafeTensorFile
  private var tensors: [String: MLXArray] = [:]
  private var projections: [String: H3QwenQ8Projection] = [:]
  private let previousCacheLimit: Int
  private(set) var projectionLoads = 0
  private(set) var tensorLoads = 0
  private(set) var residentBytes = 0
  private var maximumResidentBytes = 0
  private(set) var isClosed = false
  private(set) var groupedPreparationGroups = 0

  init(checkpointURL: URL, memoryMode: H3VideoDecodeMemoryMode? = nil,
    precision: H3VideoDecodePrecision = .defaultPrecision,
    allocationCacheLimitBytes: Int = H3VideoVAEDecodeSession.defaultAllocationCacheLimitBytes) throws {
    try Self.validateAllocationCacheLimit(allocationCacheLimitBytes, memoryMode: memoryMode)
    try Task.checkCancellation()
    self.memoryMode = memoryMode
    self.precision = precision
    self.allocationCacheLimitBytes = allocationCacheLimitBytes
    self.checkpointURL = checkpointURL
    previousCacheLimit = Memory.cacheLimit
    file = try SafeTensorFile(url: checkpointURL)
    _ = try H3VideoVAELayout(file: file, url: checkpointURL)
    // Production retains 128 MiB; only an explicit internal low-mode trial may
    // change the reusable allocation cache. Active packed weights are separate.
    Memory.cacheLimit = allocationCacheLimitBytes
  }

  deinit { close() }

  static func withSession<T>(checkpointURL: URL,
    memoryMode: H3VideoDecodeMemoryMode? = nil,
    precision: H3VideoDecodePrecision = .defaultPrecision,
    allocationCacheLimitBytes: Int = H3VideoVAEDecodeSession.defaultAllocationCacheLimitBytes,
    groupedStageLoading: Bool = false,
    onClose: (Statistics) -> Void = { _ in },
    _ body: (H3VideoVAEDecodeSession) throws -> T) throws -> T {
    let session = try H3VideoVAEDecodeSession(checkpointURL: checkpointURL,
      memoryMode: memoryMode, precision: precision, allocationCacheLimitBytes: allocationCacheLimitBytes)
    defer {
      session.close()
      onClose(Statistics(projectionLoads: session.projectionLoads,
        tensorLoads: session.tensorLoads,
        maximumResidentBytes: session.maximumResidentBytes,
        closed: session.isClosed, remainingResidentBytes: session.residentBytes,
        maximumGridCacheBytes: session.maximumGridCacheBytesObserved,
        allocationCacheLimitBytes: session.allocationCacheLimitBytes,
        groupedPreparationGroups: session.groupedPreparationGroups,
        computePrecision: session.precision.rawValue))
    }
    if groupedStageLoading { try session.prepareGroupedStage() }
    return try body(session)
  }

  /// Prefill owned decoder arrays before any tile activation exists. Each
  /// group admits at most 71,372,800 payload bytes and drains before the next.
  func prepareGroupedStage() throws {
    guard !isClosed,tensors.isEmpty,projections.isEmpty,grids.isEmpty,
      allocationCacheLimitBytes == Self.defaultAllocationCacheLimitBytes else {
      throw H3CheckpointError.invalid("Grouped H3 video preparation requires a fresh128MiB session.")
    }
    try Task.checkCancellation()
    try checkUnchanged()
    let expectedBytes = try Self.validateGroupedStageAdmission { name in
      guard let descriptor = file.tensors[name],descriptor.byteCount <= 512 * 1024 * 1024 else { return nil }
      return H3TensorInfo(dtype:descriptor.dtype,shape:descriptor.shape)
    }
    let page = H3NativePage(file:file,url:checkpointURL)
    defer { page.clear() }
    do {
      for group in Self.groupedStagePlan() {
        try Task.checkCancellation()
        try checkUnchanged()
        var values: [String:MLXArray] = [:]
        for request in group { values[request.name] = try page.read(request.name,materialize:false) }
        eval(Array(values.values))
        try checkUnchanged()
        try Task.checkCancellation()
        for request in group where request.dtype == "U32" {
          let stem = String(request.name.dropLast(".weight".count))
          guard let packed = values[request.name],let scales = values[stem + ".scales"],
            let biases = values[stem + ".biases"] else {
            throw H3CheckpointError.invalid("Grouped H3 video projection factors were not prepared.")
          }
          let projection = try H3QwenQ8Projection(packed:packed,scales:scales,biases:biases,
            columns:request.shape[1] * 4,materializeWeights:false)
          projections[request.name] = projection
          projectionLoads += 1
          residentBytes += projection.storageBytes
        }
        for request in group where request.dtype == "F16"
          && !request.name.hasSuffix(".scales") && !request.name.hasSuffix(".biases") {
          guard let value = values[request.name] else {
            throw H3CheckpointError.invalid("Grouped H3 video tensor was not prepared.")
          }
          tensors[request.name] = value
          tensorLoads += 1
          residentBytes += value.nbytes
        }
        maximumResidentBytes = max(maximumResidentBytes,residentBytes)
        groupedPreparationGroups += 1
      }
      guard residentBytes == expectedBytes,projectionLoads == 144,tensorLoads == 297 else {
        throw H3CheckpointError.invalid("Grouped H3 video preparation changed the admitted resident byte count.")
      }
      try checkUnchanged()
      try Task.checkCancellation()
    } catch {
      close()
      throw error
    }
  }

  func checkUnchanged() throws {
    guard !isClosed else {
      throw H3CheckpointError.invalid("H3 video decoder session is closed.")
    }
    try file.checkUnchanged(at: checkpointURL)
  }

  func read(_ name: String, shape: [Int]) throws -> MLXArray {
    guard !isClosed, let descriptor = file.tensors[name], descriptor.dtype == "F16",
      descriptor.shape == shape.map(UInt64.init),
      name.hasPrefix("decoder.") || name.hasPrefix("post_quant_conv.") else {
      throw H3CheckpointError.invalid("Missing H3 video decoder session tensor: \(name)")
    }
    if let value = tensors[name] { return value }
    try Task.checkCancellation()
    let value = try file.withTensorBytes(named: name) { bytes in
      MLXArray(bytes, shape, type: Float16.self)
    }
    eval(value)
    tensors[name] = value
    tensorLoads += 1
    residentBytes += value.nbytes
    maximumResidentBytes = max(maximumResidentBytes, residentBytes)
    return value
  }

  func projection(_ name: String) throws -> H3QwenQ8Projection {
    guard !isClosed, name.hasPrefix("decoder.transformer_blocks.") else {
      throw H3CheckpointError.invalid("Invalid H3 video decoder session projection.")
    }
    if let value = projections[name] { return value }
    try Task.checkCancellation()
    let value = try H3QwenQ8Projection(file: file, name: name)
    projections[name] = value
    projectionLoads += 1
    residentBytes += value.storageBytes
    maximumResidentBytes = max(maximumResidentBytes, residentBytes)
    return value
  }

  /// Positions and RoPE depend only on tile geometry and token precision.
  /// If the bounded cache is full, the caller still receives an uncached grid.
  func preparedGrid(batch: Int, depth: Int, height: Int, width: Int, dtype: DType,
    build: () throws -> Grid) throws -> Grid {
    guard (1...4).contains(batch), (1...16).contains(depth),
      (1...32).contains(height), (1...32).contains(width),
      depth * height * width <= 4096, dtype == .float16 || dtype == .float32 else {
      throw H3CheckpointError.invalid("Invalid H3 video decoder cached grid geometry.")
    }
    try checkUnchanged()
    try Task.checkCancellation()
    let key = GridKey(batch: batch, depth: depth, height: height, width: width, dtype: dtype)
    if let cached = grids[key] { return cached }
    let grid = try build()
    let rows = depth * height * width + 5
    guard grid.positions.shape == [batch, rows, 3], grid.positions.dtype == .float32,
      grid.rotary.cosine.shape == [batch, rows, 1, 48],
      grid.rotary.sine.shape == grid.rotary.cosine.shape,
      grid.rotary.cosine.dtype == dtype, grid.rotary.sine.dtype == dtype else {
      throw H3CheckpointError.invalid("Prepared H3 video grid differs from its cache key.")
    }
    try checkUnchanged()
    try Task.checkCancellation()
    let bytes = grid.storageBytes
    if bytes <= Self.maximumGridCacheBytes - gridCacheBytes {
      grids[key] = grid
      gridCacheBytes += bytes
      maximumGridCacheBytesObserved = max(maximumGridCacheBytesObserved, gridCacheBytes)
      residentBytes += bytes
      maximumResidentBytes = max(maximumResidentBytes, residentBytes)
    }
    return grid
  }

  func materializeProjection(_ value: MLXArray) {
    if H3VideoDecodeMemoryMode.materializesProjection(for: memoryMode) { eval(value) }
  }

  func materializeFirstResidual(_ value: MLXArray, blockIndex: Int) throws {
    guard (0..<36).contains(blockIndex) else {
      throw H3CheckpointError.invalid("Invalid H3 video decoder block index.")
    }
    try Task.checkCancellation()
    // Single-tile production geometry can keep residuals lazy within a block.
    // Larger private tiles retain a bounded two-block execution window.
    if memoryMode == .lowMemoryBF16 && value.shape[0] == 1 && value.shape[1] <= 2048 {
      return
    }
    let window = 2
    if memoryMode == .lowMemoryBF16 && (blockIndex+1) % window != 0 {
      // Submit larger residuals without a CPU wait, then drain every second block.
      asyncEval(value)
    } else if H3VideoDecodeMemoryMode.materializesFirstResidual(for: memoryMode,
      spatialBatchSize: value.shape[0]) {
      eval(value)
    }
  }

  func materializeBlockOutput(_ value: MLXArray, blockIndex: Int) throws {
    guard !isClosed, (0..<36).contains(blockIndex) else {
      throw H3CheckpointError.invalid("Invalid or closed H3 video decoder block output.")
    }
    try Task.checkCancellation()
    // Retire completed attention/feed scratch at a complete block boundary.
    // Normal batches1/2 retire every block; batches3/4 and single-tile low-memory
    // execution retain the tile-head boundary. Packed weights survive.
    if H3VideoDecodeMemoryMode.materializesBlockOutput(for: memoryMode,
      blockIndex: blockIndex, spatialBatchSize: value.shape[0]) { eval(value) }
  }

  func close() {
    guard !isClosed else { return }
    Stream.gpu.synchronize()
    grids.removeAll(keepingCapacity: false)
    gridCacheBytes = 0
    projections.removeAll(keepingCapacity: false)
    tensors.removeAll(keepingCapacity: false)
    residentBytes = 0
    isClosed = true
    Memory.clearCache()
    Memory.cacheLimit = previousCacheLimit
  }
}
