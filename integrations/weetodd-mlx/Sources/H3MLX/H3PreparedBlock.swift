import Foundation
import MLX
import TensorIO

/// A serially prepared block whose weights remain owned until the caller drains
/// its bounded execution window. Projection results stay lazy so the caller can
/// choose the evaluation boundary without releasing their backing weights.
final class H3PreparedBlock {
  enum Projection {
    case denseBF16(MLXArray)
    case affineQ8(H3QwenQ8Projection)

    var storageBytes: Int {
      switch self {
      case .denseBF16(let weight): return weight.nbytes
      case .affineQ8(let weight): return weight.storageBytes
      }
    }

    var shape: [Int] {
      switch self {
      case .denseBF16(let weight): return weight.shape
      case .affineQ8(let weight): return [weight.rows, weight.columns]
      }
    }

    func project(_ activation: MLXArray, useMPP: Bool, verificationScope: String) throws -> MLXArray {
      switch self {
      case .denseBF16(let weight): return try H3MPPProjection.apply(source:activation,weight:weight,enabled:useMPP,verificationScope:verificationScope)
      case .affineQ8(let weight): return try weight.project(activation)
      }
    }
  }

  private let retainProjections: Bool
  private let useMPP: Bool
  private let verificationScope: String
  let index: Int
  let checkpointURL: URL
  let tensorURL: URL
  let file: SafeTensorFile
  let layout: H3CheckpointLayout
  private var tensors: [String: MLXArray] = [:]
  private var projections: [String: Projection] = [:]
  private var baseStorageBytes = 0
  private(set) var loRAScope: H3LoRABlock?
  private var loRABindingInitialized = false
  var storageBytes: Int { baseStorageBytes + (loRAScope?.storageBytes ?? 0) }
  private(set) var isClosed = false
  let usesGroupedBF16Loading: Bool
  let usesGroupedFastQ8Loading: Bool

  init(checkpointURL: URL, index: Int,
    projectionMode: H3ProjectionMode, rowWindow: Int = 1024, useMPP: Bool = false,
    verificationScope: String? = nil, groupedBF16Loading: Bool = true,
    groupedFastQ8Loading: Bool = true, retainProjections: Bool = false) throws {
    guard (0..<50).contains(index) else {
      throw H3CheckpointError.invalid("Invalid prepared H3 block index.")
    }
    guard projectionMode == .weightDecoded else {
      throw H3CheckpointError.invalid("Prepared H3 blocks require weight-decoded projections.")
    }
    guard [1024, 2048, 4096, 8192, 16384].contains(rowWindow) else {
      throw H3CheckpointError.invalid("Prepared H3 decode row window is unsupported.")
    }
    try Task.checkCancellation()
    let admittedLayout = try H3CheckpointLayout(url: checkpointURL)
    let admittedURL = try H3CheckpointSource.fileURL(checkpointURL, block: index)
    let admittedFile = try SafeTensorFile(url: admittedURL)
    self.retainProjections = retainProjections
    self.useMPP = useMPP
    self.verificationScope = verificationScope ?? checkpointURL.standardizedFileURL.path
    self.index = index
    self.checkpointURL = checkpointURL
    tensorURL = admittedURL
    file = admittedFile
    layout = admittedLayout

    let prefix = layout.prefix + "blocks.\(index)."
    let denseDefinitions = [
      ("attn.qkv_proj", 21504, 5376), ("attn.out_proj", 5376, 7168),
      ("mlp.fc1", 28672, 5376), ("mlp.fc2", 5376, 14336),
    ]
    usesGroupedBF16Loading = groupedBF16Loading && admittedLayout.fastVariant == nil
      && admittedLayout.curveRank != nil && denseDefinitions.allSatisfy {
        admittedFile.tensors[prefix + $0.0 + ".weight"]?.dtype == "BF16"
      }
    usesGroupedFastQ8Loading = groupedFastQ8Loading && admittedLayout.fastVariant != nil
    let groupedFastLoading = usesGroupedFastQ8Loading
    let groupedLoading = usesGroupedBF16Loading || usesGroupedFastQ8Loading
    let nativePage = layout.fastVariant != nil || usesGroupedBF16Loading
      ? H3NativePage(file: file, url: tensorURL) : nil
    defer { nativePage?.clear() }
    do {
      if usesGroupedFastQ8Loading, let variant = layout.fastVariant {
        // Admit all 16 Dense / 17 VSA factors before any lazy payload is requested.
        try Self.validateGroupedFastAdmission(prefix: prefix, variant: variant) { name in
          guard let descriptor = file.tensors[name],
            descriptor.byteCount <= 512 * 1024 * 1024 else { return nil }
          return H3TensorInfo(dtype: descriptor.dtype, shape: descriptor.shape)
        }
        try checkUnchanged()
      }
      if usesGroupedBF16Loading {
        // Admit the complete single-owner set before requesting any payload.
        let definitions = [("norm1.weight", [5376]), ("norm2.weight", [5376]),
          ("attn.q_norm.weight", [128]), ("attn.k_norm.weight", [128])]
          + denseDefinitions.map { ($0.0 + ".weight", [$0.1, $0.2]) }
        for (suffix, shape) in definitions {
          guard let descriptor = file.tensors[prefix + suffix], descriptor.dtype == "BF16",
            descriptor.shape == shape.map(UInt64.init) else {
            throw H3CheckpointError.invalid("Invalid grouped H3 BF16 block tensor: \(prefix + suffix)")
          }
        }
        try checkUnchanged()
      }
      func loadTensor(_ suffix: String, shape: [Int]) throws {
        try Task.checkCancellation()
        let name = prefix + suffix
        guard let descriptor = file.tensors[name], descriptor.dtype == "BF16",
          descriptor.shape == shape.map(UInt64.init) else {
          throw H3CheckpointError.invalid("Missing H3 block tensor: \(name)")
        }
        let value: MLXArray
        if let nativePage {
          value = try nativePage.read(name, materialize: !groupedLoading)
        }
        else {
          value = try H3TensorPayload.withTensorBytes(file: file, name: name,
            maximumBufferedBytes: 4 * 1024 * 1024) {
            MLXArray($0, shape, type: UInt16.self).view(dtype: .bfloat16)
          }
          eval(value)
        }
        tensors[name] = value
        baseStorageBytes += value.nbytes
        try Task.checkCancellation()
      }
      try loadTensor("norm1.weight", shape: [5376])
      try loadTensor("norm2.weight", shape: [5376])
      try loadTensor("attn.q_norm.weight", shape: [128])
      try loadTensor("attn.k_norm.weight", shape: [128])
      if layout.fastVariant == .vsaV1 {
        try loadTensor("attn.gate_compress.weight", shape: [7168, 5376])
      }

      for (suffix, rows, columns) in denseDefinitions {
        try Task.checkCancellation()
        let name = prefix + suffix + ".weight"
        let projection: Projection
        if file.tensors[name]?.dtype == "U32" {
          let reader: ((String) throws -> MLXArray)? = nativePage.map { page in
            { try page.read($0, materialize: !groupedFastLoading) }
          }
          let weight = try H3QwenQ8Projection(file: file, name: name, tensor: reader,
            materializeWeights: !usesGroupedFastQ8Loading)
          guard weight.rows == rows, weight.columns == columns else {
            throw H3CheckpointError.invalid("Paged H3 affine projection changed after admission.")
          }
          // Native pages already store QKV rows in head-major attention order.
          projection = .affineQ8(weight)
        } else if layout.curveRank != nil {
          guard let descriptor = file.tensors[name], descriptor.dtype == "BF16",
            descriptor.shape == [UInt64(rows), UInt64(columns)] else {
            throw H3CheckpointError.invalid("Missing H3 FL2VA block projection: \(name)")
          }
          var weight: MLXArray
          if usesGroupedBF16Loading, let nativePage {
            weight = try nativePage.read(name, materialize: false)
          } else {
            weight = try file.withTensorBytes(named: name) {
              MLXArray($0, [rows, columns], type: UInt16.self).view(dtype: .bfloat16)
            }
          }
          if suffix == "attn.qkv_proj" {
            weight = H3QKVRowOrder.forHeadMajorAttention(weight,
              heads: 56, headWidth: 128, groupedSource: false)
          }
          if !usesGroupedBF16Loading { eval(weight) }
          projection = .denseBF16(weight)
        } else {
          projection = .denseBF16(try H3ComfyDecodedProjection.load(file: file,
            checkpointURL: tensorURL, name: name, rows: rows, columns: columns,
            reorderQKV: suffix == "attn.qkv_proj", rowWindow: rowWindow))
        }
        projections[suffix] = projection
        baseStorageBytes += projection.storageBytes
        try Task.checkCancellation()
      }
      if usesGroupedBF16Loading {
        try checkUnchanged()
        // Only the eight admitted arrays enter this evaluation. Unrequested
        // checkpoint tensors remain lazy handles until nativePage is cleared.
        let weights: [MLXArray] = denseDefinitions.compactMap { definition in
          if let projection = projections[definition.0], case .denseBF16(let weight) = projection { return weight }
          return nil
        }
        eval(Array(tensors.values) + weights)
      }
      if usesGroupedFastQ8Loading {
        try checkUnchanged()
        let parameters = denseDefinitions.flatMap { definition -> [MLXArray] in
          if let projection = projections[definition.0], case .affineQ8(let weight) = projection {
            return weight.parametersToMaterialize
          }
          return []
        }
        eval(Array(tensors.values) + parameters)
      }
      try checkUnchanged()
    } catch {
      close()
      throw error
    }
  }

  deinit { close() }

  /// Header-only admission, also usable without MLX arrays for rejection tests.
  static func validateGroupedFastAdmission(prefix: String, variant: H3FastVariant,
    tensor: (String) -> H3TensorInfo?) throws {
    var definitions: [(String, [UInt64], String)] = [
      ("norm1.weight", [5376], "BF16"), ("norm2.weight", [5376], "BF16"),
      ("attn.q_norm.weight", [128], "BF16"), ("attn.k_norm.weight", [128], "BF16"),
    ]
    for (suffix, rows, columns) in [("attn.qkv_proj", 21504, 5376),
      ("attn.out_proj", 5376, 7168), ("mlp.fc1", 28672, 5376), ("mlp.fc2", 5376, 14336)] {
      definitions.append((suffix + ".weight", [UInt64(rows), UInt64(columns / 4)], "U32"))
      definitions.append((suffix + ".scales", [UInt64(rows), UInt64(columns / 64)], "BF16"))
      definitions.append((suffix + ".biases", [UInt64(rows), UInt64(columns / 64)], "BF16"))
    }
    if variant == .vsaV1 { definitions.append(("attn.gate_compress.weight", [7168, 5376], "BF16")) }
    for (suffix, shape, dtype) in definitions {
      try Task.checkCancellation()
      guard tensor(prefix + suffix) == H3TensorInfo(dtype: dtype, shape: shape) else {
        throw H3CheckpointError.invalid("Invalid grouped FastH3 factor: \(prefix + suffix)")
      }
    }
  }

  /// Existing storage references for installed bitwise qualification, never copies.
  /// Release these references before checking that close retired the owner.
  func readAffineProjectionParameters(_ suffix: String) throws -> [MLXArray] {
    try checkUnchanged()
    guard let projection = projections[suffix], case .affineQ8(let weight) = projection else {
      throw H3CheckpointError.invalid("Missing prepared affine H3 projection: \(suffix)")
    }
    return weight.parametersToMaterialize
  }

  /// Scoped matrix access for installed byte-parity qualification. The caller
  /// must release its reference before asserting that close retired the owner.
  func readDenseProjectionWeight(_ suffix: String) throws -> MLXArray {
    try checkUnchanged()
    guard let projection = projections[suffix], case .denseBF16(let weight) = projection else {
      throw H3CheckpointError.invalid("Missing prepared dense H3 projection: \(suffix)")
    }
    return weight
  }

  func read(_ fullname: String, shape: [Int], dtype: String = "BF16") throws -> MLXArray {
    try requireOpen()
    guard let descriptor = file.tensors[fullname], descriptor.dtype == dtype,
      descriptor.shape == shape.map(UInt64.init), let value = tensors[fullname] else {
      throw H3CheckpointError.invalid("Missing prepared H3 block tensor: \(fullname)")
    }
    try file.checkUnchanged(at: tensorURL)
    try Task.checkCancellation()
    return value
  }

  func project(_ suffix: String, activation: MLXArray,
    rows: Int, columns: Int, qkv: Bool = false) throws -> MLXArray {
    try requireOpen()
    guard let projection = projections[suffix], projection.shape == [rows, columns],
      qkv == (suffix == "attn.qkv_proj"), activation.ndim >= 2,
      activation.shape.last == columns, activation.dtype.isFloatingPoint else {
      throw H3CheckpointError.invalid("Invalid prepared H3 projection inputs.")
    }
    try file.checkUnchanged(at: tensorURL)
    let value = try projection.project(activation,useMPP:useMPP,verificationScope:verificationScope)
    try Task.checkCancellation()
    return value
  }

  func checkUnchanged() throws {
    try requireOpen()
    try file.checkUnchanged(at: tensorURL)
    try H3CheckpointSource.checkUnchanged(checkpointURL)
    try Task.checkCancellation()
  }

  /// Bind once to the current ordered adapter activation snapshot. Custom
  /// applying implementations without prepared support retain their old path.
  func prepareLoRA(_ application: (any H3LoRAApplying)?) throws -> (any H3LoRAApplying)? {
    try requireOpen()
    if !loRABindingInitialized {
      loRAScope = try application?.prepareBlock(index: index)
      loRABindingInitialized = true
    }
    if let loRAScope { return loRAScope }
    return application
  }

  /// The caller must drain the projection's consumers before retiring a weight.
  /// MLP weights remain reusable across all row chunks until their branch ends.
  func retireProjection(_ suffix: String) {
    guard !retainProjections else { return }
    loRAScope?.retire(target: "diffusion_model.blocks.\(index)." + suffix)
    if let projection = projections.removeValue(forKey: suffix) {
      baseStorageBytes -= projection.storageBytes
    }
  }

  /// Call on the same serial execution path that prepared and used this owner.
  /// The caller also drains its window outputs before retiring the window.
  func close() {
    guard !isClosed else { return }
    Stream.gpu.synchronize()
    loRAScope?.close()
    loRAScope = nil
    projections.removeAll()
    tensors.removeAll()
    baseStorageBytes = 0
    isClosed = true
  }

  private func requireOpen() throws {
    guard !isClosed else {
      throw H3CheckpointError.invalid("Prepared H3 block is closed.")
    }
    try Task.checkCancellation()
  }
}
