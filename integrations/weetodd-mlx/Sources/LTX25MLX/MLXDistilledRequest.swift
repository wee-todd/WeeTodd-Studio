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
  public let noisePolicy:MLXNoisePolicy
  enum CodingKeys:String,CodingKey,CaseIterable {
    case version,engine,task,prompt,width,height,frames,fps,seed
    case referenceImages="reference_images",noisePolicy="noise_policy"
    case audioReference="audio_reference"
    case unionControlGuide="union_control_guide"
    case ingredientsSheet="ingredients_sheet"
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
    if requestVersion == 1 { expected.remove("reference_images") }
    if requestVersion < 3 { expected.remove("noise_policy") }
    if requestVersion < 4 { expected.remove("audio_reference") }
    if requestVersion < 5 { expected.remove("union_control_guide") }
    if requestVersion < 6 { expected.remove("ingredients_sheet") }
    guard Set(all.allKeys.map(\.stringValue)) == expected else {
      throw LTXError.invalid("Two-stage request has missing or unsupported fields.")
    }
    let c=try decoder.container(keyedBy:CodingKeys.self)
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
    ingredientsSheet=version < 6 ? nil : try c.decode(MLXIngredientsSheet.self,forKey:.ingredientsSheet)
    noisePolicy=version < 3 ? .native : try c.decode(MLXNoisePolicy.self,forKey:.noisePolicy)
    let roles=referenceImages.map(\.role)
    guard (version == 4 && task == "a2v" && (roles.isEmpty || roles == ["first"]) && audioReference != nil) ||
      (version == 5 && task == "union_control" && roles.isEmpty &&
        audioReference == nil && unionControlGuide != nil) ||
      (version == 6 && task == "ingredients" && roles.isEmpty &&
        audioReference == nil && unionControlGuide == nil && ingredientsSheet != nil &&
        frames >= 121 && stageOneLoras.isEmpty && stageTwoLoras.isEmpty &&
        noisePolicy == .releasedMLX) ||
      (version == 1 && task == "t2v") || ((version == 2 || version == 3) &&
      ((task == "t2v" && roles.isEmpty) || (task == "i2v" && roles == ["first"]) || (task == "fflf" && roles == ["first","last"] && frames>1))) else {
      throw LTXError.invalid("Request version/task must match its explicit ordered endpoint references.")
    }
    guard engine == "ltx25", prompt.utf8.count <= 65536,
      stageOneLoras.count <= 16, stageTwoLoras.count <= 16 else {
      throw LTXError.invalid("Only bounded LTX2.5 distilled audiovisual requests are supported.")
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
    for path in [gemmaRoot,transformerRoot,connectorCheckpoint,videoCheckpoint,audioCheckpoint,spatialUpscalerCheckpoint,outputDirectory] {
      guard path.hasPrefix("/"), path.utf8.count <= 4096, !path.utf8.contains(0) else {
        throw LTXError.invalid("Model and output paths must be explicit absolute local paths.")
      }
    }
    _ = try recipe()
  }
  public func encode(to encoder:Encoder) throws {
    var c=encoder.container(keyedBy:CodingKeys.self)
    try c.encode(version,forKey:.version);try c.encode(engine,forKey:.engine);try c.encode(task,forKey:.task)
    try c.encode(gemmaRoot,forKey:.gemmaRoot);try c.encode(transformerRoot,forKey:.transformerRoot)
    try c.encode(connectorCheckpoint,forKey:.connectorCheckpoint);try c.encode(videoCheckpoint,forKey:.videoCheckpoint)
    try c.encode(audioCheckpoint,forKey:.audioCheckpoint);try c.encode(spatialUpscalerCheckpoint,forKey:.spatialUpscalerCheckpoint)
    try c.encode(prompt,forKey:.prompt);try c.encode(width,forKey:.width);try c.encode(height,forKey:.height)
    try c.encode(frames,forKey:.frames);try c.encode(fps,forKey:.fps);try c.encode(seed,forKey:.seed)
    try c.encode(outputDirectory,forKey:.outputDirectory);try c.encode(stageOneLoras,forKey:.stageOneLoras);try c.encode(stageTwoLoras,forKey:.stageTwoLoras)
    if version >= 2 { try c.encode(referenceImages,forKey:.referenceImages) }
    if version >= 3 { try c.encode(noisePolicy,forKey:.noisePolicy) }
    if version >= 4 { try c.encode(audioReference,forKey:.audioReference) }
    if version >= 5 { try c.encode(unionControlGuide,forKey:.unionControlGuide) }
    if version >= 6 { try c.encode(ingredientsSheet,forKey:.ingredientsSheet) }
  }
  public func recipe() throws -> DistilledTwoStageRecipe {
    try DistilledTwoStageRecipe(width:width,height:height,frames:frames,fps:fps,seed:seed)
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
  private struct Key:CodingKey {
    let stringValue:String;var intValue:Int? { nil }
    init(stringValue:String) { self.stringValue=stringValue }
    init?(intValue:Int) { nil }
  }
  public init(from decoder:Decoder) throws {
    let c=try decoder.container(keyedBy:Key.self)
    guard Set(c.allKeys.map(\.stringValue)) == ["role","path","strength","crf"] else { throw LTXError.invalid("Unknown or missing image reference fields.") }
    role=try c.decode(String.self,forKey:Key(stringValue:"role"));path=try c.decode(String.self,forKey:Key(stringValue:"path"))
    strength=try c.decode(Float.self,forKey:Key(stringValue:"strength"));crf=try c.decode(Int.self,forKey:Key(stringValue:"crf"))
    guard ["first","last"].contains(role),path.hasPrefix("/"),path.utf8.count<=4096,!path.utf8.contains(0),
      strength.isFinite,(0...1).contains(strength),(0...51).contains(crf) else { throw LTXError.invalid("Invalid image reference settings.") }
  }
}
