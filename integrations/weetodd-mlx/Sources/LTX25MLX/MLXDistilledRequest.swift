import Foundation
import Darwin
import LTX25Engine
import AdapterRuntime

/// Versioned developer two-stage audiovisual request. Unsupported controls are rejected,
/// never silently accepted or routed to another runtime. No payload reads here.
public struct MLXDistilledRequest:Codable,Sendable {
  public let version:Int, engine:String, task:String
  public let gemmaRoot:String, transformerRoot:String, connectorCheckpoint:String
  public let videoCheckpoint:String, audioCheckpoint:String, spatialUpscalerCheckpoint:String
  public let prompt:String, width:Int, height:Int, frames:Int, fps:Double, seed:UInt64
  public let outputDirectory:String
  public let stageOneLoras:[LoRAAdapter], stageTwoLoras:[LoRAAdapter]
  public let referenceImages:[MLXImageReference]
  public let audioReference:MLXAudioReference?
  public let unionControlGuide:MLXUnionControlGuide?
  public let ingredientsSheet:MLXIngredientsSheet?
  public let msr:MLXMSRRequest?
  public let dfr:MLXDFRRequest?
  public let icControl:MLXICControl?
  public let ingredientsSampling:MLXIngredientsSampling
  public let guidedSampling:MLXGuidedSampling?
  public let automaticDuration:MLXAutomaticDurationPolicy?
  public let singleStageSampling:MLXSingleStageSampling?
  public let diffusionVAE:MLXDiffusionVideoSettings?
  public let generatedKeyframes:Int?
  public var usesOrdinaryKeyframes:Bool { version == 14 || version == 15 }
  public var referenceFrames:[Int] {
    referenceImages.map { $0.role == "first" ? 0 : $0.role == "last" ? frames-1 : $0.frameIndex! }
  }
  public func ordinaryKeyframeLayout(geometry:AVGeometry,stage:Int) throws -> MLXOrdinaryKeyframeLayout? {
    guard usesOrdinaryKeyframes else { return nil }
    guard stage == 0 || stage == 1 else { throw LTXError.invalid("Invalid ordinary keyframe stage.") }
    return try MLXOrdinaryKeyframeLayout(geometry:geometry,
      anchors:zip(referenceFrames,referenceImages).map { .init(frame:$0.0,strength:$0.1.strength) },
      generatedCount:stage == 0 ? generatedKeyframes ?? 0 : 0)
  }
  public func singleStageControlLayout() throws -> MLXSingleStageControlLayout? {
    guard singleStageSampling != nil else { return nil }
    return try MLXSingleStageControlLayout(geometry:recipe().high,
      anchors:zip(referenceFrames,referenceImages).map { .init(frame:$0.0,strength:$0.1.strength) },
      generatedCount:generatedKeyframes ?? 0,icControl:icControl,
      unionStrength:unionControlGuide?.referenceStrength)
  }
  public let noisePolicy:MLXNoisePolicy
  enum CodingKeys:String,CodingKey,CaseIterable {
    case version,engine,task,prompt,width,height,frames,fps,seed
    case diffusionVAE="diffusion_vae"
    case referenceImages="reference_images",noisePolicy="noise_policy"
    case audioReference="audio_reference"
    case unionControlGuide="union_control_guide"
    case ingredientsSheet="ingredients_sheet"
    case msr,dfr,icControl="ic_control"
    case ingredientsSampling="ingredients_sampling",guidedSampling="guided_sampling"
    case automaticDuration="automatic_duration"
    case generatedKeyframes="generated_keyframes",singleStageSampling="single_stage_sampling"
    case gemmaRoot="gemma_root",transformerRoot="transformer_root",connectorCheckpoint="connector_checkpoint"
    case videoCheckpoint="video_checkpoint",audioCheckpoint="audio_checkpoint",spatialUpscalerCheckpoint="spatial_upscaler_checkpoint"
    case outputDirectory="output_directory",stageOneLoras="stage_one_loras",stageTwoLoras="stage_two_loras"
  }
  private struct AnyKey:CodingKey {
    let stringValue:String
    var intValue:Int? { nil }
    init(stringValue:String) { self.stringValue=stringValue }
    init?(intValue:Int) { return nil }
  }
  public init(from decoder:Decoder) throws {
    let all=try decoder.container(keyedBy:AnyKey.self)
    let requestVersion=try all.decode(Int.self,forKey:AnyKey(stringValue:"version"))
    var expected=Set(CodingKeys.allCases.map(\.rawValue))
    expected.remove("diffusion_vae")
    if requestVersion == 1 { expected.remove("reference_images") }
    if requestVersion < 3 { expected.remove("noise_policy") }
    if requestVersion < 4 { expected.remove("audio_reference") }
    if requestVersion < 5 { expected.remove("union_control_guide") }
    if requestVersion < 6 { expected.remove("ingredients_sheet") }
    if requestVersion < 7 { expected.remove("msr") }
    if requestVersion < 8 { expected.remove("dfr") }
    if requestVersion < 10 { expected.remove("ic_control") }
    if requestVersion < 11 { expected.remove("ingredients_sampling") }
    if requestVersion < 12 { expected.remove("guided_sampling") }
    if requestVersion < 13 { expected.remove("automatic_duration") }
    if requestVersion < 14 { expected.remove("generated_keyframes") }
    if requestVersion < 15 { expected.remove("single_stage_sampling") }
    guard Set(all.allKeys.map(\.stringValue)).subtracting(["diffusion_vae"]) == expected else {
      throw LTXError.invalid("Two-stage request has missing or unsupported fields.")
    }
    let c=try decoder.container(keyedBy:CodingKeys.self)
    diffusionVAE=try c.decodeIfPresent(MLXDiffusionVideoSettings.self,forKey:.diffusionVAE)
    version=try c.decode(Int.self,forKey:.version); engine=try c.decode(String.self,forKey:.engine)
    task=try c.decode(String.self,forKey:.task)
    gemmaRoot=try c.decode(String.self,forKey:.gemmaRoot)
    transformerRoot=try c.decode(String.self,forKey:.transformerRoot)
    connectorCheckpoint=try c.decode(String.self,forKey:.connectorCheckpoint)
    videoCheckpoint=try c.decode(String.self,forKey:.videoCheckpoint)
    audioCheckpoint=try c.decode(String.self,forKey:.audioCheckpoint)
    spatialUpscalerCheckpoint=try c.decode(String.self,forKey:.spatialUpscalerCheckpoint)
    prompt=try c.decode(String.self,forKey:.prompt); width=try c.decode(Int.self,forKey:.width)
    height=try c.decode(Int.self,forKey:.height); frames=try c.decode(Int.self,forKey:.frames)
    fps=try c.decode(Double.self,forKey:.fps); seed=try c.decode(UInt64.self,forKey:.seed)
    outputDirectory=try c.decode(String.self,forKey:.outputDirectory)
    stageOneLoras=try c.decode([LoRAAdapter].self,forKey:.stageOneLoras)
    stageTwoLoras=try c.decode([LoRAAdapter].self,forKey:.stageTwoLoras)
    referenceImages=version == 1 ? [] : try c.decode([MLXImageReference].self,forKey:.referenceImages)
    audioReference=version < 4 ? nil : try c.decodeIfPresent(MLXAudioReference.self,forKey:.audioReference)
    unionControlGuide=version < 5 ? nil : try c.decodeIfPresent(MLXUnionControlGuide.self,forKey:.unionControlGuide)
    ingredientsSheet=version < 6 ? nil : try c.decodeIfPresent(MLXIngredientsSheet.self,forKey:.ingredientsSheet)
    msr=version < 7 ? nil : try c.decodeIfPresent(MLXMSRRequest.self,forKey:.msr)
    dfr=version < 8 ? nil : try c.decodeIfPresent(MLXDFRRequest.self,forKey:.dfr)
    icControl=version < 10 ? nil : try c.decodeIfPresent(MLXICControl.self,forKey:.icControl)
    ingredientsSampling=version < 11 ? .deterministic : try c.decode(MLXIngredientsSampling.self,forKey:.ingredientsSampling)
    guard version == 11 || ingredientsSampling == .deterministic else {
      throw LTXError.invalid("The authored Ingredients sampler requires its dedicated version 11 request.")
    }
    guidedSampling=version < 12 ? nil : try c.decodeIfPresent(MLXGuidedSampling.self,forKey:.guidedSampling)
    automaticDuration=version < 13 ? nil : try c.decodeIfPresent(MLXAutomaticDurationPolicy.self,forKey:.automaticDuration)
    generatedKeyframes=version < 14 ? nil : try c.decodeIfPresent(Int.self,forKey:.generatedKeyframes)
    singleStageSampling=version < 15 ? nil : try c.decodeIfPresent(MLXSingleStageSampling.self,forKey:.singleStageSampling)
    noisePolicy=version < 3 ? .native : try c.decode(MLXNoisePolicy.self,forKey:.noisePolicy)
    let dfrCanvas=try (version == 8 || version == 9) ? MLXDFRCanvas(frames:frames) : nil
    let roles=referenceImages.map(\.role)
    let ordinaryKeyframes = (version == 14 || (version == 15 && singleStageSampling != nil && guidedSampling == nil && automaticDuration == nil && stageTwoLoras.isEmpty)) && noisePolicy == .releasedMLX &&
      (0...8).contains(generatedKeyframes ?? -1) && referenceImages.count <= 8 &&
      Set(referenceFrames).count == referenceFrames.count && referenceFrames.allSatisfy { (0..<frames).contains($0) } &&
      unionControlGuide == nil && ingredientsSheet == nil && msr == nil && dfr == nil && icControl == nil &&
      ((task == "t2v" && roles.isEmpty && audioReference == nil) ||
       (task == "i2v" && !roles.isEmpty && audioReference == nil) ||
       (task == "fflf" && roles.contains("first") && roles.contains("last") && audioReference == nil) ||
       (task == "a2v" && audioReference != nil && automaticDuration == nil))
    let singleStageControl = version == 15 && singleStageSampling != nil && guidedSampling == nil &&
      automaticDuration == nil && stageTwoLoras.isEmpty && noisePolicy == .releasedMLX &&
      (0...8).contains(generatedKeyframes ?? -1) && referenceImages.count<=8 &&
      Set(referenceFrames).count == referenceFrames.count && referenceFrames.allSatisfy { (0..<frames).contains($0) } &&
      ingredientsSheet == nil && msr == nil && dfr == nil && audioReference == nil &&
      ((task == "ic_control" && icControl != nil && unionControlGuide == nil) ||
       (task == "union_control" && unionControlGuide != nil && icControl == nil))
    let guidedIngredients = version == 17 && task == "ingredients" && roles.isEmpty && frames >= 121 &&
      ingredientsSheet?.referenceStrength == 1 && guidedSampling?.singleStage == true &&
      guidedSampling?.mode == .guided && ingredientsSampling == .deterministic &&
      stageOneLoras.isEmpty && stageTwoLoras.isEmpty && noisePolicy == .releasedMLX &&
      audioReference == nil && unionControlGuide == nil && msr == nil && dfr == nil && icControl == nil &&
      automaticDuration == nil && generatedKeyframes == nil && singleStageSampling == nil
    guard ordinaryKeyframes || singleStageControl || guidedIngredients || ((version == 12 && guidedSampling != nil || version == 13 && automaticDuration != nil) && noisePolicy == .releasedMLX &&
      unionControlGuide == nil && ingredientsSheet == nil && msr == nil && dfr == nil && icControl == nil &&
      ((task == "t2v" && roles.isEmpty && audioReference == nil) ||
       (task == "i2v" && roles == ["first"] && audioReference == nil) ||
       (task == "fflf" && roles == ["first","last"] && frames > 1 && audioReference == nil) ||
       (version == 12 && task == "a2v" && (roles.isEmpty || roles == ["first"]) && audioReference != nil))) ||
      (version == 4 && task == "a2v" && (roles.isEmpty || roles == ["first"]) && audioReference != nil) ||
      (version == 10 && task == "ic_control" && roles.isEmpty && audioReference == nil &&
        unionControlGuide == nil && ingredientsSheet == nil && msr == nil && dfr == nil && icControl != nil &&
        noisePolicy == .releasedMLX) ||
      (version == 5 && task == "union_control" && roles.isEmpty &&
        audioReference == nil && unionControlGuide != nil) ||
      ((version == 6 || version == 11) && task == "ingredients" && roles.isEmpty &&
        audioReference == nil && unionControlGuide == nil && ingredientsSheet != nil &&
        frames >= 121 && stageOneLoras.isEmpty && stageTwoLoras.isEmpty &&
        noisePolicy == .releasedMLX && (version == 6 ||
          (ingredientsSampling == .ancestralCFGPP && ingredientsSheet?.referenceStrength == 1 &&
            msr == nil && dfr == nil && icControl == nil))) ||
      ((version == 7 || version == 16) && task == "msr" && roles.isEmpty && audioReference == nil &&
        unionControlGuide == nil && ingredientsSheet == nil && msr != nil &&
        stageOneLoras.isEmpty && stageTwoLoras.isEmpty && noisePolicy == .releasedMLX &&
        dfr == nil && icControl == nil && guidedSampling == nil && automaticDuration == nil &&
        generatedKeyframes == nil && singleStageSampling == nil &&
        (version == 16 ? !(msr?.audioReferences.isEmpty ?? true) : msr?.audioReferences.isEmpty == true)) ||
      ((version == 8 || version == 9) && task == "dfr" &&
        (roles.isEmpty || roles == ["first"] || roles == ["first","last"]) && audioReference == nil &&
        unionControlGuide == nil && ingredientsSheet == nil && msr == nil && dfr != nil &&
        stageOneLoras.isEmpty && stageTwoLoras.isEmpty && noisePolicy == .releasedMLX &&
        dfrCanvas != nil && (version == 8 ? dfr?.temporalRounds == 0 : (1...2).contains(dfr!.temporalRounds)) &&
        (dfr.map { fps*Double(1 << $0.temporalRounds) <= 120 } ?? false)) ||
      (version == 1 && task == "t2v") || ((version == 2 || version == 3) &&
      ((task == "t2v" && roles.isEmpty) || (task == "i2v" && roles == ["first"]) || (task == "fflf" && roles == ["first","last"] && frames>1))) else {
      throw LTXError.invalid("Request version/task must match its explicit ordered endpoint references.")
    }
    guard engine == "ltx25", prompt.utf8.count <= 65536,
      stageOneLoras.count <= 16, stageTwoLoras.count <= 16 else {
      throw LTXError.invalid("Only bounded LTX2.5 distilled audiovisual requests are supported.")
    }
    if let guidedSampling {
      guard guidedSampling.singleStage == guidedIngredients else {
        throw LTXError.invalid("Single-stage Dev guidance requires its dedicated Ingredients request.")
      }
      guard stageTwoLoras.count < 16,
        !(stageOneLoras+stageTwoLoras).contains(where: { $0.path == guidedSampling.distilledAdapterPath }) else {
        throw LTXError.invalid("The Dev refinement adapter must appear only in its dedicated stage-two slot.")
      }
      _ = try guidedSampling.schedule(videoTokens:guidedIngredients ? recipe().high.videoTokens*2 : recipe().low.videoTokens)
    }
    if let automaticDuration {
      _ = try automaticDuration.maximumFrames(fps:fps)
    }
    guard version == 14 || version == 15 || !roles.contains("keyframe") else {
      throw LTXError.invalid("Timed image references require the ordinary-keyframe request version.")
    }
    if let unionControlGuide {
      guard stageOneLoras.count < 16,
        !(stageOneLoras + stageTwoLoras).contains(where: { $0.path == unionControlGuide.adapterPath }) else {
        throw LTXError.invalid("The Union task adapter must appear exactly once in its dedicated stage-one slot.")
      }
    }
    if let ingredientsSheet {
      guard !(stageOneLoras + stageTwoLoras).contains(where: { $0.path == ingredientsSheet.adapterPath }) else {
        throw LTXError.invalid("The Ingredients task adapter must appear only in its dedicated single-stage slot.")
      }
    }
    if let msr {
      guard !(stageOneLoras + stageTwoLoras).contains(where: { $0.path == msr.adapterPath }) else {
        throw LTXError.invalid("MSR adapter must appear only in its dedicated single-stage slot.")
      }
    }
    if let dfr {
      guard !(stageOneLoras + stageTwoLoras).contains(where: { $0.path == dfr.adapterPath }) else {
        throw LTXError.invalid("DFR detailing adapter must appear only in its dedicated second-stage slot.")
      }
    }
    if let control=icControl {
      let taskPaths=Set(control.adapters.map(\.path))
      guard !(stageOneLoras+stageTwoLoras).contains(where:{ taskPaths.contains($0.path) }),
        stageOneLoras.count+control.adapters.count<=16 else {
        throw LTXError.invalid("IC task adapters must appear only in their dedicated stage-one stack.")
      }
      _ = try control.guideGeometry(target:recipe().low)
    }
    guard singleStageSampling?.method != .cfgpp || audioReference == nil else { throw LTXError.invalid("Single-stage CFG++ cannot freeze source audio.") }
    var requiredPaths=[gemmaRoot,transformerRoot,connectorCheckpoint,videoCheckpoint,audioCheckpoint,outputDirectory]
    if !spatialUpscalerCheckpoint.isEmpty || ![6,7,11,15,16,17].contains(version) { requiredPaths.append(spatialUpscalerCheckpoint) }
    for path in requiredPaths {
      guard path.hasPrefix("/"), path.utf8.count <= 4096, !path.utf8.contains(0) else {
        throw LTXError.invalid("Model and output paths must be explicit absolute local paths.")
      }
    }
    _ = try recipe()
    _ = try singleStageControlLayout()
  }
  public func encode(to encoder:Encoder) throws {
    var c=encoder.container(keyedBy:CodingKeys.self)
    try c.encode(version,forKey:.version);try c.encode(engine,forKey:.engine);try c.encode(task,forKey:.task)
    try c.encode(gemmaRoot,forKey:.gemmaRoot);try c.encode(transformerRoot,forKey:.transformerRoot)
    try c.encode(connectorCheckpoint,forKey:.connectorCheckpoint);try c.encode(videoCheckpoint,forKey:.videoCheckpoint)
    try c.encode(audioCheckpoint,forKey:.audioCheckpoint);try c.encode(spatialUpscalerCheckpoint,forKey:.spatialUpscalerCheckpoint)
    try c.encodeIfPresent(diffusionVAE,forKey:.diffusionVAE)
    try c.encode(prompt,forKey:.prompt);try c.encode(width,forKey:.width);try c.encode(height,forKey:.height)
    try c.encode(frames,forKey:.frames);try c.encode(fps,forKey:.fps);try c.encode(seed,forKey:.seed)
    try c.encode(outputDirectory,forKey:.outputDirectory);try c.encode(stageOneLoras,forKey:.stageOneLoras);try c.encode(stageTwoLoras,forKey:.stageTwoLoras)
    if version >= 2 { try c.encode(referenceImages,forKey:.referenceImages) }
    if version >= 3 { try c.encode(noisePolicy,forKey:.noisePolicy) }
    if version >= 4 { try c.encode(audioReference,forKey:.audioReference) }
    if version >= 5 { try c.encode(unionControlGuide,forKey:.unionControlGuide) }
    if version >= 6 { try c.encode(ingredientsSheet,forKey:.ingredientsSheet) }
    if version >= 7 { try c.encode(msr,forKey:.msr) }
    if version >= 8 { try c.encode(dfr,forKey:.dfr) }
    if version >= 10 { try c.encode(icControl,forKey:.icControl) }
    if version >= 11 { try c.encode(ingredientsSampling,forKey:.ingredientsSampling) }
    if version >= 12 { try c.encode(guidedSampling,forKey:.guidedSampling) }
    if version >= 13 { try c.encode(automaticDuration,forKey:.automaticDuration) }
    if version >= 14 { try c.encode(generatedKeyframes,forKey:.generatedKeyframes) }
    if version >= 15 { try c.encode(singleStageSampling,forKey:.singleStageSampling) }
  }
  /// Resolves geometry through the same strict versioned contract. Last-image
  /// references remain endpoint roles, so they follow the effective last frame.
  public func replacingFrames(_ effectiveFrames:Int,automaticHeadHeaderSHA256:String?=nil) throws -> Self {
    var object=try JSONSerialization.jsonObject(with:JSONEncoder().encode(self)) as! [String:Any]
    object["frames"]=effectiveFrames
    if let automaticHeadHeaderSHA256 {
      guard var policy=object["automatic_duration"] as? [String:Any] else {
        throw LTXError.invalid("A duration-head pin requires an automatic request.")
      }
      policy["head_header_sha256"]=automaticHeadHeaderSHA256
      object["automatic_duration"]=policy
    }
    return try JSONDecoder().decode(Self.self,from:JSONSerialization.data(withJSONObject:object))
  }
  public func recipe() throws -> DistilledTwoStageRecipe {
    try DistilledTwoStageRecipe(width:width,height:height,
      frames:dfr == nil ? frames : MLXDFRCanvas(frames:frames).frames,fps:fps,seed:seed,singleStage:singleStageSampling != nil)
  }
  public static func load(_ url:URL) throws -> Self {
    guard url.isFileURL else { throw LTXError.invalid("Request must be a local file.") }
    let descriptor=Darwin.open(url.path,O_RDONLY | O_CLOEXEC | O_NONBLOCK)
    guard descriptor >= 0 else { throw LTXError.invalid("Cannot open request file.") }
    let handle=FileHandle(fileDescriptor:descriptor,closeOnDealloc:true)
    defer { try? handle.close() }
    var status=stat()
    guard fstat(descriptor,&status) == 0, status.st_mode & S_IFMT == S_IFREG else {
      throw LTXError.invalid("Request must be a regular file.")
    }
    guard status.st_size <= 1024*1024 else { throw LTXError.invalid("Request exceeds1MiB.") }
    let data=try handle.read(upToCount:1024*1024+1) ?? Data()
    guard data.count <= 1024*1024 else { throw LTXError.invalid("Request exceeds1MiB.") }
    return try JSONDecoder().decode(Self.self,from:data)
  }
}

/// Version 6 preserves the original deterministic recipe. Version 11 opts into
/// the authored CFG++ recipe with Float32 sampler state and BF16 model inputs.
public enum MLXIngredientsSampling:String,Codable,Sendable {
  case deterministic = "deterministic_bf16_v1"
  case ancestralCFGPP = "euler_ancestral_cfg_pp_float32_v1"
  public var transformerEvaluations:Int { self == .ancestralCFGPP ? 16 : 8 }
}

/// The official Pixel-Spatial x2 adapter applies to DFR's full-resolution stage.
public struct MLXDFRRequest:Codable,Sendable {
  public let adapterPath:String
  public let adapterStrength:Float
  public let temporalUpscalerPath:String?
  public let temporalRounds:Int
  enum CodingKeys:String,CodingKey,CaseIterable {
    case adapterPath="adapter_path",adapterStrength="adapter_strength"
    case temporalUpscalerPath="temporal_upscaler_path",temporalRounds="temporal_rounds"
  }
  public init(from decoder:Decoder) throws {
    let c=try decoder.container(keyedBy:CodingKeys.self)
    let fields=Set(c.allKeys.map(\.stringValue))
    let spatial:Set<String>=[CodingKeys.adapterPath.rawValue,CodingKeys.adapterStrength.rawValue]
    guard fields == spatial || fields == Set(CodingKeys.allCases.map(\.rawValue)) else {
      throw LTXError.invalid("DFR needs its exact detailing-adapter fields.")
    }
    adapterPath=try c.decode(String.self,forKey:.adapterPath)
    adapterStrength=try c.decode(Float.self,forKey:.adapterStrength)
    temporalUpscalerPath=fields == spatial ? nil : try c.decode(String.self,forKey:.temporalUpscalerPath)
    temporalRounds=fields == spatial ? 0 : try c.decode(Int.self,forKey:.temporalRounds)
    guard adapterPath.hasPrefix("/"),adapterPath.utf8.count <= 4096,!adapterPath.utf8.contains(0),
      adapterStrength.isFinite,adapterStrength > 0,adapterStrength <= 3,
      (0...2).contains(temporalRounds),
      (temporalRounds == 0) == (temporalUpscalerPath == nil),
      temporalUpscalerPath.map({ $0.hasPrefix("/") && $0.utf8.count <= 4096 && !$0.utf8.contains(0) }) ?? true else {
      throw LTXError.invalid("DFR detailing adapter path or strength is invalid.")
    }
  }
  public func encode(to encoder:Encoder) throws {
    var c=encoder.container(keyedBy:CodingKeys.self)
    try c.encode(adapterPath,forKey:.adapterPath)
    try c.encode(adapterStrength,forKey:.adapterStrength)
    if let temporalUpscalerPath {
      try c.encode(temporalUpscalerPath,forKey:.temporalUpscalerPath)
      try c.encode(temporalRounds,forKey:.temporalRounds)
    }
  }
}

/// One source image repeated through the full causal clip before VAE encoding.
public struct MLXIngredientsSheet:Codable,Sendable {
  public let path:String,sourceSHA256:String,adapterPath:String
  public let adapterStrength:Float,referenceStrength:Float
  enum CodingKeys:String,CodingKey,CaseIterable {
    case path,sourceSHA256="source_sha256",adapterPath="adapter_path",adapterStrength="adapter_strength",
      referenceStrength="reference_strength"
  }
  public init(from decoder:Decoder) throws {
    let c=try decoder.container(keyedBy:CodingKeys.self)
    guard Set(c.allKeys.map(\.stringValue)) == Set(CodingKeys.allCases.map(\.rawValue)) else {
      throw LTXError.invalid("Ingredients needs its exact sheet and adapter fields.")
    }
    path=try c.decode(String.self,forKey:.path)
    sourceSHA256=try c.decode(String.self,forKey:.sourceSHA256)
    adapterPath=try c.decode(String.self,forKey:.adapterPath)
    adapterStrength=try c.decode(Float.self,forKey:.adapterStrength)
    referenceStrength=try c.decode(Float.self,forKey:.referenceStrength)
    guard [path,adapterPath].allSatisfy({ $0.hasPrefix("/") && $0.utf8.count <= 4096 && !$0.utf8.contains(0) }),
      sourceSHA256.utf8.count == 64,
      sourceSHA256.utf8.allSatisfy({ (48...57).contains($0) || (97...102).contains($0) }),
      adapterStrength.isFinite,adapterStrength > 0,adapterStrength <= 3,
      referenceStrength.isFinite,(0...1).contains(referenceStrength) else {
      throw LTXError.invalid("Ingredients sheet path, digest or strength is invalid.")
    }
  }
}

/// One dedicated MSR adapter and one to five ordered still references.
public struct MLXMSRRequest:Codable,Sendable {
  public let adapterPath:String
  public let adapterStrength:Float
  public let references:[MLXMSRReference]
  public let audioReferences:[MLXMSRAudioReference]
  enum CodingKeys:String,CodingKey,CaseIterable {
    case adapterPath="adapter_path",adapterStrength="adapter_strength",references,audioReferences="audio_references"
  }
  public init(from decoder:Decoder) throws {
    let all=try decoder.container(keyedBy:MSRAnyKey.self)
    let c=try decoder.container(keyedBy:CodingKeys.self)
    guard Set(all.allKeys.map(\.stringValue)).subtracting(["audio_references"]) == Set(CodingKeys.allCases.map(\.rawValue)).subtracting(["audio_references"]) else {
      throw LTXError.invalid("MSR requires exact adapter and reference fields.")
    }
    adapterPath=try c.decode(String.self,forKey:.adapterPath)
    adapterStrength=try c.decode(Float.self,forKey:.adapterStrength)
    references=try c.decode([MLXMSRReference].self,forKey:.references)
    audioReferences=(try c.decodeIfPresent([MLXMSRAudioReference].self,forKey:.audioReferences) ?? []).sorted { $0.imageSlot < $1.imageSlot }
    guard adapterPath.hasPrefix("/"),adapterPath.utf8.count <= 4096,!adapterPath.utf8.contains(0),
      adapterStrength.isFinite,adapterStrength > 0,adapterStrength <= 3,
      (1...5).contains(references.count),references.filter({ $0.role == "background" }).count <= 1,
      audioReferences.count<=2,Set(audioReferences.map(\.imageSlot)).count == audioReferences.count,
      audioReferences.allSatisfy({ $0.imageSlot<=references.count && references[$0.imageSlot-1].role != "background" }),
      audioReferences.isEmpty || references.map(\.role) == (references.filter { $0.role != "background" } + references.filter { $0.role == "background" }).map(\.role) else {
      throw LTXError.invalid("MSR adapter or one-to-five reference count is invalid.")
    }
  }
}

private struct MSRAnyKey:CodingKey {
  let stringValue:String
  var intValue:Int? { nil }
  init(stringValue:String) { self.stringValue=stringValue }
  init?(intValue:Int) { return nil }
}

/// A bounded voice identity reference, explicitly paired to visual image 1 or 2.
/// The reference is conditioning only; the model generates the output soundtrack.
public struct MLXMSRAudioReference:Codable,Sendable {
  public let path:String,sourceSHA256:String,imageSlot:Int
  public let sourceStartSeconds:Double,sourceDurationSeconds:Double
  public var effectiveDurationSeconds:Double { min(sourceDurationSeconds,5) }
  enum CodingKeys:String,CodingKey,CaseIterable {
    case path,sourceSHA256="source_sha256",imageSlot="image_slot"
    case sourceStartSeconds="source_start_seconds",sourceDurationSeconds="source_duration_seconds"
  }
  public init(from decoder:Decoder) throws {
    let all=try decoder.container(keyedBy:MSRAnyKey.self)
    guard Set(all.allKeys.map(\.stringValue)) == Set(CodingKeys.allCases.map(\.rawValue)) else {
      throw LTXError.invalid("MSR voice reference contains missing or unsupported fields.")
    }
    let c=try decoder.container(keyedBy:CodingKeys.self)
    path=try c.decode(String.self,forKey:.path);sourceSHA256=try c.decode(String.self,forKey:.sourceSHA256)
    imageSlot=try c.decode(Int.self,forKey:.imageSlot)
    sourceStartSeconds=try c.decode(Double.self,forKey:.sourceStartSeconds)
    sourceDurationSeconds=try c.decode(Double.self,forKey:.sourceDurationSeconds)
    guard path.hasPrefix("/"),path.utf8.count<=4096,!path.utf8.contains(0),
      sourceSHA256.utf8.count == 64,sourceSHA256.utf8.allSatisfy({ (48...57).contains($0) || (97...102).contains($0) }),
      (1...2).contains(imageSlot),sourceStartSeconds.isFinite,sourceStartSeconds>=0,
      sourceDurationSeconds.isFinite,sourceDurationSeconds>0,sourceDurationSeconds<=86400,
      (sourceStartSeconds+sourceDurationSeconds).isFinite else {
      throw LTXError.invalid("MSR voice path, digest, image slot or source interval is invalid.")
    }
  }
}

public struct MLXMSRReference:Codable,Sendable {
  public let path:String,sourceSHA256:String,role:String,priority:String,sizePolicy:String,referenceFrames:String
  public let strength:Float,attentionStrength:Float
  enum CodingKeys:String,CodingKey,CaseIterable {
    case path,sourceSHA256="source_sha256",role,priority
    case sizePolicy="size_policy",referenceFrames="reference_frames"
    case strength,attentionStrength="attention_strength"
  }
  public init(from decoder:Decoder) throws {
    let c=try decoder.container(keyedBy:CodingKeys.self)
    guard Set(c.allKeys.map(\.stringValue)) == Set(CodingKeys.allCases.map(\.rawValue)) else {
      throw LTXError.invalid("MSR image reference fields are incomplete.")
    }
    path=try c.decode(String.self,forKey:.path)
    sourceSHA256=try c.decode(String.self,forKey:.sourceSHA256)
    role=try c.decode(String.self,forKey:.role)
    priority=try c.decode(String.self,forKey:.priority)
    sizePolicy=try c.decode(String.self,forKey:.sizePolicy)
    referenceFrames=try c.decode(String.self,forKey:.referenceFrames)
    strength=try c.decode(Float.self,forKey:.strength)
    attentionStrength=try c.decode(Float.self,forKey:.attentionStrength)
    guard path.hasPrefix("/"),path.utf8.count <= 4096,!path.utf8.contains(0),
      sourceSHA256.utf8.count == 64,
      sourceSHA256.utf8.allSatisfy({ (48...57).contains($0) || (97...102).contains($0) }),
      ["subject","object","clothing","background"].contains(role),
      ["auto","primary","supporting","background"].contains(priority),
      ["sol_auto","quality","balanced","speed"].contains(sizePolicy),
      ["auto","25","33"].contains(referenceFrames),
      strength.isFinite,(0...1).contains(strength),
      attentionStrength.isFinite,(0...1).contains(attentionStrength) else {
      throw LTXError.invalid("MSR reference path, role, sizing or strength is invalid.")
    }
  }
}

/// A preprocessed, full-timeline RGB24 Union guide. Pixel preparation belongs
/// to the caller; the worker verifies the exact stage-one reference byte count.
public struct MLXUnionControlGuide:Codable,Sendable {
  public let path:String,sourceSHA256:String,adapterPath:String
  public let adapterStrength:Float,referenceStrength:Float
  enum CodingKeys:String,CodingKey,CaseIterable {
    case path,sourceSHA256="source_sha256",adapterPath="adapter_path",adapterStrength="adapter_strength",
      referenceStrength="reference_strength"
  }
  public init(from decoder:Decoder) throws {
    let c=try decoder.container(keyedBy:CodingKeys.self)
    guard Set(c.allKeys.map(\.stringValue)) == Set(CodingKeys.allCases.map(\.rawValue)) else {
      throw LTXError.invalid("Union Control requires its exact guide and adapter fields.")
    }
    path=try c.decode(String.self,forKey:.path)
    sourceSHA256=try c.decode(String.self,forKey:.sourceSHA256)
    adapterPath=try c.decode(String.self,forKey:.adapterPath)
    adapterStrength=try c.decode(Float.self,forKey:.adapterStrength)
    referenceStrength=try c.decode(Float.self,forKey:.referenceStrength)
    guard [path,adapterPath].allSatisfy({ $0.hasPrefix("/") && $0.utf8.count <= 4096 && !$0.utf8.contains(0) }),
      adapterStrength.isFinite, adapterStrength > 0, adapterStrength <= 2,
      sourceSHA256.utf8.count == 64,
      sourceSHA256.utf8.allSatisfy({ (48...57).contains($0) || (97...102).contains($0) }),
      referenceStrength.isFinite, (0...1).contains(referenceStrength) else {
      throw LTXError.invalid("Union Control guide paths or strengths are invalid.")
    }
  }
}

/// One bounded source interval; publication remains the original source waveform.
public struct MLXAudioReference:Codable,Sendable {
  public let path:String
  public let sourceStartSeconds:Double
  public let sourceDurationSeconds:Double
  enum CodingKeys:String,CodingKey,CaseIterable {
    case path,sourceStartSeconds="source_start_seconds",sourceDurationSeconds="source_duration_seconds"
  }
  public init(from decoder:Decoder) throws {
    let c=try decoder.container(keyedBy:CodingKeys.self)
    guard Set(c.allKeys.map(\.stringValue)) == Set(CodingKeys.allCases.map(\.rawValue)) else {
      throw LTXError.invalid("A2V audio reference must specify its exact source interval.")
    }
    path=try c.decode(String.self,forKey:.path)
    sourceStartSeconds=try c.decode(Double.self,forKey:.sourceStartSeconds)
    sourceDurationSeconds=try c.decode(Double.self,forKey:.sourceDurationSeconds)
    guard path.hasPrefix("/"),path.utf8.count <= 4096,!path.utf8.contains(0),
      sourceStartSeconds.isFinite,(0...86400).contains(sourceStartSeconds),
      sourceDurationSeconds.isFinite,(0.001...86400).contains(sourceDurationSeconds) else {
      throw LTXError.invalid("Invalid A2V source path or interval.")
    }
  }
}

/// Explicit preprocessing strength/CRF accompany each saved endpoint.
public struct MLXImageReference:Codable,Sendable {
  public let role:String,path:String
  public let strength:Float
  public let crf:Int
  public let frameIndex:Int?
  enum CodingKeys:String,CodingKey {
    case role,path,strength,crf
    case frameIndex="frame_index"
  }
  private struct Key:CodingKey {
    let stringValue:String;var intValue:Int? { nil }
    init(stringValue:String) { self.stringValue=stringValue }
    init?(intValue:Int) { nil }
  }
  public init(from decoder:Decoder) throws {
    let c=try decoder.container(keyedBy:Key.self)
    role=try c.decode(String.self,forKey:Key(stringValue:"role"));path=try c.decode(String.self,forKey:Key(stringValue:"path"))
    let expected:Set<String> = role == "keyframe" ? ["role","path","strength","crf","frame_index"] : ["role","path","strength","crf"]
    guard Set(c.allKeys.map(\.stringValue)) == expected else { throw LTXError.invalid("Unknown or missing image reference fields.") }
    frameIndex=role == "keyframe" ? try c.decode(Int.self,forKey:Key(stringValue:"frame_index")) : nil
    strength=try c.decode(Float.self,forKey:Key(stringValue:"strength"));crf=try c.decode(Int.self,forKey:Key(stringValue:"crf"))
    guard ["first","last","keyframe"].contains(role),frameIndex.map({ $0 >= 0 }) ?? true,
      path.hasPrefix("/"),path.utf8.count<=4096,!path.utf8.contains(0),
      strength.isFinite,(0...1).contains(strength),(0...51).contains(crf) else { throw LTXError.invalid("Invalid image reference settings.") }
  }
}
