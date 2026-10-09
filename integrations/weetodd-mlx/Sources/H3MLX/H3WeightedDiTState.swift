import Foundation
import Darwin
import MLX
import TensorIO

/// One weighted denoiser for standard and reference-conditioned H3 layouts.
/// Only packing differs; projection, LoRA, blocks and final heads stay shared.
final class H3WeightedDiTState {
  enum Layout {
    case audiovisual(H3PackedLayout)
    case references(H3ReferenceLayout)

    var maximumPackedRows: Int {
      switch self {
      case .audiovisual(let value): value.maximumPackedRows
      case .references(let value): value.maximumPackedRows
      }
    }
    var tags: [Int32] {
      switch self {
      case .audiovisual(let value): value.tags
      case .references(let value): value.tags
      }
    }
    var textRows: Int {
      switch self {
      case .audiovisual(let value): value.audioStart - value.conditionVideoRows
      case .references(let value):
        value.tags.count - value.videoIndices.count - value.audioIndices.count
      }
    }
    var videoRows: Int {
      switch self {
      case .audiovisual(let value):
        value.conditionVideoRows + value.tags.count - value.videoStart
      case .references(let value): value.videoIndices.count
      }
    }
    var audioRows: Int {
      switch self {
      case .audiovisual(let value): value.videoStart - value.audioStart
      case .references(let value): value.audioIndices.count
      }
    }
    func pack(text: MLXArray, video: MLXArray, audio: MLXArray,
      timestepIndices: [Int32]) throws -> H3PackedSequence {
      switch self {
      case .audiovisual(let value):
        try H3PackedSequence(layout: value, text: text, video: video,
          audio: audio, timestepIndices: timestepIndices)
      case .references(let value):
        try H3PackedSequence(layout: value, text: text, video: video,
          audio: audio, timestepIndices: timestepIndices)
      }
    }
  }
  private let checkpointURL: URL
  private let layout: Layout
  private let blockCount: Int
  private let projectionMode: H3ProjectionMode
  private let lora: H3LoRAStack?
  private var rotaryAngles: H3RotaryAngles?
  private var text: MLXArray?
  private var timeEmbeddings: MLXArray?
  private var modulations: [MLXArray]?
  private var modulationLoRAInputs: MLXArray?
  private var funControl: H3FunControlState?
  private let vdn:H3VDNRuntime?
  private let fastTiles:H3FastTiles?
  private let preparedWindowSize: Int?
  private var transformerWeightCache:H3TransformerWeightCache?
  private let allowMPP: Bool
  private let experimentalSol: H3SolTaskPolicy?
  private let solGeometry: H3SolGeometry?
  private var solConsumers = 0
  private var solDenseBlocks = 0
  private var solExactPairs: UInt64 = 0
  private var solCoarsePairs: UInt64 = 0
  private var solMaximumPreparedBytes = 0
  private let nativeWorkerURL: URL?
  private let nativeAdapterURL: URL?
  private var nativeModelFiles: [(URL, SafeTensorFile)] = []
  private var nativeSession: H3NativeBlockSession?
  private var nativeMemoryReport: H3NativeMemoryMonitor.Report?
  private var nativeJobRoot: URL?
  private(set) var nativeEvaluationCount = 0
  /// Child prediction wall time includes its core preparation, load and compute.
  private(set) var nativeKernelSeconds: Double = 0
  /// Inclusive parent block-bridge time; nativeKernelSeconds is nested within this value.
  private(set) var nativeBridgeSeconds: Double = 0
  var nativeTransferAndManagementSeconds: Double { max(0, nativeBridgeSeconds - nativeKernelSeconds) }
  private(set) var nativeChildReaped = false
  private let projectionVerificationScope = UUID().uuidString

  var isResident: Bool {
    text != nil && timeEmbeddings != nil && modulations != nil
  }

  /// Prepared activations held across steps; block weights are streamed and
  /// therefore intentionally absent from this resident count.
  var residentActivationBytes: Int {
    (text?.nbytes ?? 0) + (timeEmbeddings?.nbytes ?? 0)
      + (modulations?.reduce(0) { $0 + $1.nbytes } ?? 0)
      + (funControl?.residentActivationBytes ?? 0) + (modulationLoRAInputs?.nbytes ?? 0)
      + (rotaryAngles.map { $0.cosine.nbytes + $0.sine.nbytes } ?? 0)
  }

  init(checkpointURL: URL, layout: Layout,
    textEmbeddings: MLXArray, timestepTable: [Float],
    blockCount: Int,
    projectionMode: H3ProjectionMode = .weightDecoded,
    allowMPP: Bool = false, nativeWorkerURL: URL? = nil, transformerWeightCacheGB:Int = 0,
    experimentalSol: H3SolTaskPolicy? = nil,
    turboLoRAURL: URL? = nil, turboLoRAStrength: Float = 1,
    additionalLoRAs: [H3LoRAAdapter] = [],
    loRAAdapters: [H3LoRAAdapter]? = nil,
    funControl: H3FunControlCondition? = nil,
    vdn:H3VDNSelection? = nil,
    preparationObserver: H3PreparationObservation.Observer? = nil,
    progress: (Int, Int) -> Void = { _, _ in }) throws {
    let textRows = layout.textRows
    guard (1...50).contains(blockCount),
      textRows > 0, textEmbeddings.shape == [1, textRows, 5120],
      textEmbeddings.dtype.isFloatingPoint,
      (1...128).contains(timestepTable.count),
      timestepTable.allSatisfy({ $0.isFinite && (0...1).contains($0) }),
      zip(timestepTable, timestepTable.dropFirst()).allSatisfy({ $0 < $1 }),
      [40_000, 64_000].contains(layout.maximumPackedRows),
      (1...layout.maximumPackedRows).contains(layout.tags.count) else {
      throw H3CheckpointError.invalid("Invalid H3 denoiser preparation request.")
    }
    if let funControl {
      guard blockCount == 50, case .audiovisual(let packed) = layout,
        packed.conditionVideoRows == 0,
        funControl.guideRows.shape == [1, layout.videoRows, 96],
        funControl.strength.isFinite, (0...1).contains(funControl.strength) else {
        throw H3CheckpointError.invalid("H3 Fun control requires a compatible dense T2VA transformer and matching guide rows.")
      }
      _ = try H3FunControlLayout(url: funControl.checkpoint,
        base: H3CheckpointLayout(url: checkpointURL))
    }
    self.checkpointURL = checkpointURL
    self.layout = layout
    self.blockCount = blockCount
    self.projectionMode = projectionMode
    self.allowMPP = allowMPP && nativeWorkerURL == nil
    self.nativeWorkerURL = nativeWorkerURL
    var adapters = additionalLoRAs
    if let turboLoRAURL {
      adapters.insert(try H3LoRAAdapter(url: turboLoRAURL,
        strength: turboLoRAStrength), at: 0)
    }
    let effective = loRAAdapters ?? adapters
    if let nativeWorkerURL {
      var workerStatus = stat()
      guard nativeWorkerURL.isFileURL,
        fstatat(AT_FDCWD, nativeWorkerURL.path, &workerStatus, 0) == 0,
        workerStatus.st_mode & S_IFMT == S_IFREG,
        FileManager.default.isReadableFile(atPath: nativeWorkerURL.path),
        FileManager.default.isExecutableFile(atPath: nativeWorkerURL.path) else {
        throw H3CheckpointError.invalid("Experimental NNC worker is not a regular readable executable.")
      }
      try Self.validateNativeConfiguration(layout: layout, blockCount: blockCount,
        projectionMode: projectionMode, hasFunControl: funControl != nil, hasVDN: vdn != nil,
        tableRows: timestepTable.count, adapters: effective)
      let adapter = effective[0].url
      try H3NativeBlockAdmission.inspect(checkpoint: checkpointURL, adapter: adapter)
      nativeAdapterURL = adapter
      nativeModelFiles = [(checkpointURL, try SafeTensorFile(url: checkpointURL)),
        (adapter, try SafeTensorFile(url: adapter))]
    } else { nativeAdapterURL = nil }
    if let vdn {
      guard blockCount == 50,funControl == nil,effective.isEmpty,
        case .audiovisual(let packed)=layout,
        (try H3CheckpointLayout(url:checkpointURL).curveRank == nil ? vdn.adalnInputGrid == nil :
          (vdn.variant == .fiftyStep || vdn.adalnInputGrid != nil)) else {
        throw H3CheckpointError.invalid("VDN denoising requires the complete T2VA backbone and correct original-width adapter coordinates without other adapters or controls.")
      }
      self.vdn=try H3VDNRuntime(selection:vdn,packed:packed)
    } else { self.vdn=nil }
    let checkpointLayout = try H3CheckpointLayout(url:checkpointURL)
    self.experimentalSol = experimentalSol
    if let experimentalSol {
      let referenceLayout:Bool
      let generatedIndices:[Int]
      switch layout {
      case .references(let references): referenceLayout=true;generatedIndices=references.targetVideoIndices
      case .audiovisual(let frames): referenceLayout=false;generatedIndices=Array(frames.videoStart..<frames.tags.count)
      }
      try H3SolTaskPolicy.validateState(referenceLayout: referenceLayout, blockCount: blockCount,
        weightDecoded: projectionMode == .weightDecoded, hasNativeWorker: nativeWorkerURL != nil,
        maximumRows: layout.maximumPackedRows, hasFast: checkpointLayout.fastVariant != nil,
        hasCurveRank: checkpointLayout.curveRank != nil, hasVDN: vdn != nil, hasFun: funControl != nil)
      let range = try H3SolTaskPolicy.generatedVideoRange(indices: generatedIndices, rows: layout.tags.count)
      solGeometry = try H3SolGeometry(rows: layout.tags.count, heads: 56,
        approximationRange: range, tau: experimentalSol.tau)
    } else { solGeometry = nil }
    // Controls and larger spatial admission retain their independently
    // qualified execution paths. Core H3 uses bounded stage-local owners.
    preparedWindowSize = nativeWorkerURL == nil && projectionMode == .weightDecoded && funControl == nil
      && vdn == nil && layout.maximumPackedRows == 40_000
      ? (checkpointLayout.fastVariant == nil ? 1:2) : nil
    if checkpointLayout.fastVariant != nil {
      guard blockCount == 50,funControl == nil,effective.isEmpty,vdn == nil,
        case .audiovisual(let packed) = layout,packed.conditionVideoRows == 0 else {
        throw H3CheckpointError.invalid("FastH3 requires unmasked T2VA without additional adapters or controls.")
      }
      self.fastTiles = checkpointLayout.fastVariant == .vsaV1
        ? try H3FastTiles(prefixSegments:[packed.audioStart,packed.videoStart-packed.audioStart],videoGrid:packed.videoGrid) : nil
    } else { self.fastTiles = nil }
    if transformerWeightCacheGB > 0 {
      guard preparedWindowSize == 1,effective.allSatisfy({ $0.startAfterEvaluations == 0 }) else {
        throw H3CheckpointError.invalid("H3 weight cache requires ordinary MLX blocks and immediate adapters.")
      }
      transformerWeightCache = try H3TransformerWeightCache(checkpointURL:checkpointURL,
        blockCount:blockCount,budgetGB:transformerWeightCacheGB,adapters:effective,
        useMPP:self.allowMPP && H3MPPProjection.isAvailable,verificationScope:projectionVerificationScope)
    } else if transformerWeightCacheGB != 0 { throw H3CheckpointError.invalid("Invalid H3 transformer cache budget.") }
    self.lora = try effective.isEmpty ? nil : H3LoRAStack(adapters: effective)
    let application: (any H3LoRAApplying)?
    if let vdn = self.vdn { application = vdn.lora } else { application = self.lora }
    self.text = nil
    self.timeEmbeddings = nil
    self.modulations = nil
    self.modulationLoRAInputs = nil
    try Task.checkCancellation()
    defer {
      H3PreparationObservation.measure(.terminalCleanup, observer: preparationObserver) {
        // Preserve the original stage-final completion and cache retirement.
        Stream.gpu.synchronize()
        Memory.clearCache()
      }
    }
    let projected = try H3PreparationObservation.measure(.conditionProjection,
      observer: preparationObserver) {
      try H3InputProjection.evaluate(checkpointURL: checkpointURL,
        kind: .condition, input: textEmbeddings)
    }
    let refined = try H3PreparationObservation.measure(.tokenRefinement,
      observer: preparationObserver) {
      try H3TokenRefiner.evaluate(checkpointURL: checkpointURL,
        input: projected, lora: application)
    }
    let (time, loraTime) = try H3PreparationObservation.measure(.timeAndSmallMetadata,
      observer: preparationObserver) {
      let time = try H3TimeEmbedding.evaluate(checkpointURL: checkpointURL,
        timesteps: MLXArray(timestepTable))
      let loraTime = try vdn?.adalnInputGrid.map {
        try H3VDNInputGrid(url:$0).evaluate(timesteps:timestepTable)
      }
      return (time, loraTime)
    }
    let tables = try H3PreparationObservation.measure(.adalnTables,
      observer: preparationObserver) {
      var tables: [MLXArray] = []
      tables.reserveCapacity(blockCount)
      for index in 0..<blockCount {
        tables.append(try H3AdaLNProjection.evaluate(
          checkpointURL: checkpointURL, blockIndex: index,
          timeEmbeddings: time, projectionMode: projectionMode,lora:application,loraInput:loraTime))
        if index + 1 < blockCount { progress(index + 1, blockCount) }
        try Task.checkCancellation()
      }
      return tables
    }
    text = refined
    timeEmbeddings = time
    modulations = tables
    modulationLoRAInputs = loraTime
    if let funControl {
      self.funControl = try H3FunControlState(condition: funControl,
        timeEmbeddings: time, base: H3CheckpointLayout(url: checkpointURL))
    }
    progress(blockCount, blockCount)
  }

  /// Settings-only admission, before any weighted preparation or child launch.
  static func validateNativeConfiguration(layout: Layout, blockCount: Int,
    projectionMode: H3ProjectionMode, hasFunControl: Bool, hasVDN: Bool,
    tableRows: Int, adapters: [H3LoRAAdapter]) throws {
    guard case .references = layout, blockCount == 50, projectionMode == .weightDecoded,
      !hasFunControl, !hasVDN, layout.maximumPackedRows == 40_000,
      (1...100).contains(tableRows) else {
      throw H3CheckpointError.invalid("Experimental NNC requires the complete ordinary Ref2VA backbone without Fun, VDN or rotated projections.")
    }
    try H3NativeBlockAdmission.validateSettings(task: "ref2va", steps: 5,
      samplingMethod: .euler, adapters: adapters, contextFrames: 0,
      isRefinement: false, packedRows: layout.tags.count)
  }

  private var completedEvaluations = 0
  private var vsaOriginalRowsIndexedCalls = 0
  private var vsaGroupedSparseCalls = 0
  private var vsaDenseCalls = 0
  private var retiredBackendReport: H3BackendReport?
  private var allocationPoolLimit = 0
  private var maximumCachedAllocationBytes = 0
  private var weightPreparationSeconds = 0.0
  private var allocationPoolCleared = true

  var backendReport: H3BackendReport {
    if let retiredBackendReport { return retiredBackendReport }
    let statistics = H3MPPProjection.verificationStatus(scope: projectionVerificationScope)
    return H3BackendReport(transformerWeightCache:transformerWeightCache?.report, preparedWindowSize: preparedWindowSize,
      eligibleProjectionCalls: statistics.eligibleCalls,
      mppProjectionCalls: statistics.mppCalls,
      knownFallbackProjectionCalls: statistics.knownFallbackCalls,
      firstUseReferenceProjectionCalls: statistics.firstUseReferenceCalls,
      verifiedMPPSignatures: statistics.verified,
      rejectedMPPSignatures: statistics.fallback,
      vsaOriginalRowsIndexedCalls: vsaOriginalRowsIndexedCalls,
      vsaGroupedSparseCalls: vsaGroupedSparseCalls, vsaDenseCalls: vsaDenseCalls,
      sol: solGeometry.map { geometry in H3SolReport(tau: geometry.tau,
        generatedVideoStart: geometry.approximationRange.lowerBound, generatedVideoEnd: geometry.approximationRange.upperBound,
        completedEvaluations: completedEvaluations, completedSolConsumers: solConsumers,
        completedDenseBlocks: solDenseBlocks, selectedExactBlockPairs: solExactPairs,
        coarseBlockPairs: solCoarsePairs, maximumPreparedLogicalBytes: solMaximumPreparedBytes) },
      nativeBlocks: nativeWorkerURL == nil ? nil : H3NativeBlockReport(
        evaluations: nativeEvaluationCount, predictionSeconds: nativeKernelSeconds,
        bridgeSeconds: nativeBridgeSeconds, childReaped: nativeChildReaped,
        physicalMemory: nativeSession?.memoryReport ?? nativeMemoryReport),
      samplingAllocationPool: preparedWindowSize == 1 && nativeWorkerURL == nil
        ? H3SamplingAllocationReport(softLimitBytes: allocationPoolLimit,
          maximumObservedCachedBytes: maximumCachedAllocationBytes,
          weightPreparationSeconds: weightPreparationSeconds,
          clearedAtEvaluationBoundary: allocationPoolCleared) : nil)
  }

  public func predict(videoLatents: MLXArray, audioLatents: MLXArray,
    timestepIndices: [Int32], progress: (Int, Int) -> Void = { _, _ in }) throws
    -> H3FinalLayer.Output {
    var succeeded = false
    defer { if (nativeWorkerURL != nil || transformerWeightCache != nil) && !succeeded { unload() } }
    guard nativeWorkerURL == nil || completedEvaluations < 4 else {
      throw H3CheckpointError.invalid("Experimental NNC supports exactly four transformer evaluations.")
    }
    guard let text, let timeEmbeddings, let modulations else {
      throw H3CheckpointError.invalid("H3 denoiser state was unloaded.")
    }
    guard videoLatents.shape == [1, layout.videoRows, 96],
      audioLatents.shape == [1, layout.audioRows, 32],
      videoLatents.dtype.isFloatingPoint,
      audioLatents.dtype.isFloatingPoint,
      timestepIndices.count == layout.tags.count,
      timestepIndices.allSatisfy({ (0..<timeEmbeddings.shape[0]).contains(Int($0)) }) else {
      throw H3CheckpointError.invalid("Invalid H3 denoiser latent or timestep rows.")
    }
    try Task.checkCancellation()
    let previousCacheLimit = Memory.cacheLimit
    if preparedWindowSize != nil || nativeWorkerURL != nil {
      let eligible = preparedWindowSize == 1 && nativeWorkerURL == nil
        && vdn == nil && funControl == nil && allowMPP && H3MPPProjection.isAvailable
      let available = eligible ? try H3TransformerCachePlan.availableMemory() : 0
      let limit = H3SamplingAllocationPolicy.limit(previous: previousCacheLimit,
        physical: Int(ProcessInfo.processInfo.physicalMemory),
        recommended: Int(GPU.deviceInfo().maxRecommendedWorkingSetSize),
        available: available, eligible: eligible)
      Memory.cacheLimit = limit
      allocationPoolLimit = limit
      allocationPoolCleared = false
    }
    defer {
      Stream.gpu.synchronize()
      Memory.clearCache()
      allocationPoolCleared = Memory.cacheMemory == 0
      Memory.cacheLimit = previousCacheLimit
    }
    lora?.evaluation = completedEvaluations
    defer { lora?.evaluation = nil }
    let application: (any H3LoRAApplying)?
    if let vdn { application = vdn.lora } else { application = lora }
    if nativeWorkerURL != nil {
      let exported = try exportProjectionPacked(videoLatents: videoLatents,
        audioLatents: audioLatents, timestepIndices: timestepIndices,
        text: text, modulations: modulations)
      // The export helper's autorelease pool and strong packed/projection locals
      // have ended. No full-width BF16 input survives into child execution/import.
      let releaseBegan = DispatchTime.now().uptimeNanoseconds
      Stream.gpu.synchronize()
      Memory.clearCache()
      nativeBridgeSeconds += Double(DispatchTime.now().uptimeNanoseconds - releaseBegan) / 1e9
      let value = try predictNative(exported: exported, progress: progress)
      let result = try H3FinalLayer.evaluate(checkpointURL: checkpointURL,
        input: value, timeEmbeddings: timeEmbeddings,
        timestepIndices: exported.timestepIndices, videoIndices: exported.videoIndices,
        audioIndices: exported.audioIndices, maximumRows: layout.maximumPackedRows,
        residualPrecision: .float32, lora: application,
        loraInput: modulationLoRAInputs, observe: { _, _ in })
      completedEvaluations += 1
      succeeded = true
      return result
    }
    let video = try H3InputProjection.evaluate(checkpointURL: checkpointURL,
      kind: .video, input: videoLatents).asType(.bfloat16)
    let audio = try H3InputProjection.evaluate(checkpointURL: checkpointURL,
      kind: .audio, input: audioLatents).asType(.bfloat16)
    let packed = try layout.pack(text: text, video: video, audio: audio,
      timestepIndices: timestepIndices)
    // Layout and positional coordinates are immutable for this stage, even
    // when per-row timestep indices change between evaluations.
    if rotaryAngles == nil {
      rotaryAngles = try H3TransformerBlock.prepareRotaryAngles(
        checkpointURL: checkpointURL, positions: packed.positions, maximumRows: layout.maximumPackedRows)
    }
    guard let rotaryAngles, rotaryAngles.rows == packed.positions.shape[0] else {
      throw H3CheckpointError.invalid("H3 prepared rotary differs from the stage layout.")
    }
    let control = try funControl?.initialize(hidden: packed.embeddings,
      targetIndices: packed.videoIndices)
    let controlBlock: ((Int, MLXArray) throws -> (MLXArray, MLXArray))? = funControl.map { state in
      { index, current in
        try state.step(index: index, control: current,
          modulationIndices: packed.modulationIndices,
          angles: rotaryAngles, audioIndices: packed.audioIndices)
      }
    }
    let cachedBlocks = transformerWeightCache?.plan.blockCount ?? 0
    let window = try (cachedBlocks < blockCount ? preparedWindowSize : nil).map {
      try H3PreparedBlockWindow(checkpointURL: checkpointURL, blockCount: blockCount,
        windowSize: $0, projectionMode: projectionMode, useMPP: allowMPP && H3MPPProjection.isAvailable,
        verificationScope: projectionVerificationScope, startIndex:cachedBlocks)
    }
    defer { window?.close() }
    let queueWindow = preparedWindowSize == 2 && packed.embeddings.shape[1] <= 16_384
    let value = try H3FunControlMath.runBlocks(input: packed.embeddings,
      blockCount: blockCount, control: control,
      injectionLayers: funControl?.injectionLayers ?? H3FunControlLayout.v1InjectionLayers,
      baseBlock: { index, input in
        let cached = index < cachedBlocks
        let preparationBegan = ProcessInfo.processInfo.systemUptime
        let owner = try cached ? transformerWeightCache?.weights(for:index) : window?.weights(for:index)
        weightPreparationSeconds += ProcessInfo.processInfo.systemUptime - preparationBegan
        let useSol = experimentalSol?.usesSol(block: index, completedEvaluations: completedEvaluations) == true
        let attendedOverride: ((MLXArray, MLXArray, MLXArray) throws -> MLXArray)?
        if useSol {
          guard let solGeometry else { throw H3CheckpointError.invalid("Missing admitted Sol geometry.") }
          attendedOverride = { query, key, value in
            // Only completed output and scalars escape; pooled/routes/stats
            // are released before the unchanged out projection and FFN.
            let result = try autoreleasepool { () throws -> (MLXArray, UInt64, UInt64, Int) in
              let prepared = try H3SolRouting.prepare(query: query, key: key, value: value, geometry: solGeometry)
              let output = try H3SolIndexedAttention.evaluate(query: query, key: key, value: value, prepared: prepared)
              let counts = prepared.exactCounts.asArray(UInt32.self)
              guard counts.count == solGeometry.heads * solGeometry.queryBlocks,
                counts.allSatisfy({ $0 <= UInt32(solGeometry.keyBlocks) }) else {
                throw H3CheckpointError.invalid("Sol route counts differ from admitted geometry.")
              }
              let exact = counts.reduce(UInt64(0)) { $0 + UInt64($1) }
              let total = UInt64(solGeometry.heads * solGeometry.queryBlocks * solGeometry.keyBlocks)
              return (output, exact, total - exact, prepared.storageBytes)
            }
            self.solConsumers += 1
            self.solExactPairs += result.1; self.solCoarsePairs += result.2
            self.solMaximumPreparedBytes = max(self.solMaximumPreparedBytes, result.3)
            try Task.checkCancellation()
            return result.0
          }
        } else { attendedOverride = nil }
        let output = try H3TransformerBlock.evaluate(checkpointURL: checkpointURL,
          index: index, input: input, modulation: modulations[index],
          modulationIndices: packed.modulationIndices,
          positions: packed.positions, projectionMode: projectionMode,
          lora: application, rotaryAngles: rotaryAngles, maximumRows: layout.maximumPackedRows,
          vdn:vdn,fastTiles:fastTiles,preparedWeights:owner,attendedOverride:attendedOverride,onAttentionBackend: { consumer in
            switch consumer {
            case .originalRowsIndexed: self.vsaOriginalRowsIndexedCalls += 1
            case .groupedSparse: self.vsaGroupedSparseCalls += 1
            case .dense: self.vsaDenseCalls += 1
            }
          },observe: { _, _ in })
        if experimentalSol != nil && !useSol { solDenseBlocks += 1 }
        if cached { try transformerWeightCache?.checkBudget() }
        else { try window?.finish(index: index, deferRetirement: queueWindow) }
        maximumCachedAllocationBytes = max(maximumCachedAllocationBytes, Memory.cacheMemory)
        return output
      }, controlBlock: controlBlock, progress: { completed, total in
        // A queued pair is completed only after its final output and weights
        // are drained. Never report the first submitted block as completed.
        if !queueWindow || completed.isMultiple(of: 2) || completed == total {
          progress(completed, total)
        }
      })
    let result = try H3FinalLayer.evaluate(checkpointURL: checkpointURL,
      input: value, timeEmbeddings: timeEmbeddings,
      timestepIndices: packed.timestepIndices,
      videoIndices: packed.videoIndices,
      audioIndices: packed.audioIndices, maximumRows: layout.maximumPackedRows,
      residualPrecision: .bfloat16, lora:application,loraInput:modulationLoRAInputs,observe: { _, _ in })
    completedEvaluations += 1
    succeeded = true
    return result
  }

  /// Only narrow index arrays leave the projection/export lifetime scope.
  private struct NativeCoordinates {
    let rows: Int
    let timestepIndices: MLXArray
    let videoIndices: MLXArray
    let audioIndices: MLXArray
  }

  private func exportProjectionPacked(videoLatents: MLXArray, audioLatents: MLXArray,
    timestepIndices: [Int32], text: MLXArray, modulations: [MLXArray]) throws -> NativeCoordinates {
    var exportBegan: UInt64?
    defer {
      if let began = exportBegan {
        nativeBridgeSeconds += Double(DispatchTime.now().uptimeNanoseconds - began) / 1e9
      }
    }
    return try autoreleasepool {
      for (url, file) in nativeModelFiles { try file.checkUnchanged(at: url) }
      let video = try H3InputProjection.evaluate(checkpointURL: checkpointURL,
        kind: .video, input: videoLatents).asType(.bfloat16)
      let audio = try H3InputProjection.evaluate(checkpointURL: checkpointURL,
        kind: .audio, input: audioLatents).asType(.bfloat16)
      let packed = try layout.pack(text: text, video: video, audio: audio,
        timestepIndices: timestepIndices)
      if rotaryAngles == nil {
        rotaryAngles = try H3TransformerBlock.prepareRotaryAngles(
          checkpointURL: checkpointURL, positions: packed.positions,
          maximumRows: layout.maximumPackedRows)
      }
      guard let rotaryAngles, rotaryAngles.rows == packed.positions.shape[0] else {
        throw H3CheckpointError.invalid("H3 prepared rotary differs from the stage layout.")
      }
      // Start after projections/packing. The outer defer includes destruction
      // of local packed/projection handles and the autorelease pool drain.
      exportBegan = DispatchTime.now().uptimeNanoseconds
      if nativeJobRoot == nil {
        let root = FileManager.default.temporaryDirectory
          .appendingPathComponent("WeeTodd-H3-Native-" + UUID().uuidString, isDirectory: true)
        guard mkdir(root.path, 0o700) == 0 else {
          throw H3CheckpointError.invalid("Cannot create a private experimental NNC job directory.")
        }
        nativeJobRoot = root
      }
      guard let root = nativeJobRoot else {
        throw H3CheckpointError.invalid("Experimental NNC export ownership is unavailable.")
      }
      if nativeSession == nil {
        try H3NativeBlockTensorIO.writeInitial(x: packed.embeddings,
          indices: packed.modulationIndices, modulations: modulations, angles: rotaryAngles,
          to: root.appendingPathComponent("initial.safetensors"))
      }
      try H3NativeBlockTensorIO.writeRequest(x: packed.embeddings,
        indices: packed.modulationIndices, tableRows: modulations[0].shape[0] * 3,
        to: root.appendingPathComponent("input-\(nativeEvaluationCount).safetensors"))
      return NativeCoordinates(rows: packed.embeddings.shape[1],
        timestepIndices: packed.timestepIndices, videoIndices: packed.videoIndices,
        audioIndices: packed.audioIndices)
    }
  }

  private func predictNative(exported: NativeCoordinates,
    progress: (Int, Int) -> Void) throws -> MLXArray {
    guard let worker = nativeWorkerURL, let adapter = nativeAdapterURL,
      let root = nativeJobRoot else {
      throw H3CheckpointError.invalid("Experimental NNC was not admitted for this stage.")
    }
    let began = DispatchTime.now().uptimeNanoseconds
    defer { nativeBridgeSeconds += Double(DispatchTime.now().uptimeNanoseconds - began) / 1e9 }
    for (url, file) in nativeModelFiles { try file.checkUnchanged(at: url) }
    if nativeSession == nil {
      let initial = root.appendingPathComponent("initial.safetensors")
      nativeSession = try H3NativeBlockSession(initialURL: initial,
        checkpointURL: checkpointURL, adapterURL: adapter,
        workspaceURL: root.appendingPathComponent("child-workspace", isDirectory: true),
        rows: exported.rows, workerURL: worker)
      nativeChildReaped = false
      try FileManager.default.removeItem(at: initial)
    }
    guard let session = nativeSession else {
      throw H3CheckpointError.invalid("Experimental NNC child ownership is unavailable.")
    }
    let input = root.appendingPathComponent("input-\(nativeEvaluationCount).safetensors")
    let output = root.appendingPathComponent("child-workspace/output-\(nativeEvaluationCount).f32")
    let report = try session.predict(input: input, output: output, progress: progress)
    nativeKernelSeconds += report.seconds
    nativeEvaluationCount += 1
    // The completed output file survives child reaping. Import it only after the
    // last child is gone, then remove the workspace after the owned MLX copy exists.
    if nativeEvaluationCount == 4 { try retireNativeChild() }
    let result = try H3NativeBlockTensorIO.readOutput(url: output, rows: exported.rows)
    for (url, file) in nativeModelFiles { try file.checkUnchanged(at: url) }
    try FileManager.default.removeItem(at: input)
    try FileManager.default.removeItem(at: output)
    if nativeEvaluationCount == 4 { try removeNativeWorkspace() }
    try Task.checkCancellation()
    return result
  }

  private func retireNativeChild() throws {
    guard let session = nativeSession else { return }
    var failure: Error?
    do { try session.close() } catch { failure = error }
    nativeChildReaped = session.childReaped
    guard nativeChildReaped else {
      throw failure ?? H3CheckpointError.invalid("Experimental NNC child was not reaped.")
    }
    nativeMemoryReport = session.memoryReport
    nativeSession = nil
    if let failure { throw failure }
  }

  private func removeNativeWorkspace() throws {
    guard nativeSession == nil else {
      throw H3CheckpointError.invalid("Experimental NNC workspace cannot be removed before child retirement.")
    }
    if let root = nativeJobRoot {
      try FileManager.default.removeItem(at: root)
      nativeJobRoot = nil
    }
  }

  /// Reap precedes deletion even when close reports a protocol failure. No primary
  /// prediction/cancellation error is replaced by this nonthrowing unload path.
  private func retireNative(includeBridgeTime: Bool = true) throws {
    guard nativeSession != nil || nativeJobRoot != nil else { return }
    let began = DispatchTime.now().uptimeNanoseconds
    defer {
      if includeBridgeTime {
        nativeBridgeSeconds += Double(DispatchTime.now().uptimeNanoseconds - began) / 1e9
      }
    }
    var failure: Error?
    do { try retireNativeChild() } catch { failure = error }
    if nativeSession == nil {
      do { try removeNativeWorkspace() } catch { if failure == nil { failure = error } }
    }
    if let failure { throw failure }
  }

  public func unload() {
    try? retireNative()
    nativeModelFiles = []
    transformerWeightCache?.close()
    // Preserve scalar execution evidence before the scoped registry is retired.
    // No tensor or owner escapes, including failure/cancellation teardown.
    if retiredBackendReport == nil { retiredBackendReport = backendReport }
    H3MPPProjection.forget(scope: projectionVerificationScope)
    rotaryAngles = nil
    text = nil
    timeEmbeddings = nil
    modulations = nil
    modulationLoRAInputs = nil
    funControl?.unload()
    funControl = nil
    Stream.gpu.synchronize()
    Memory.clearCache()
  }

  deinit { unload() }
}
