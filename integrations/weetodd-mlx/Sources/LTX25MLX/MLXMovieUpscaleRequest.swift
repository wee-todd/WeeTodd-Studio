import CryptoKit
import Foundation
import LTX25Engine
import InferenceContracts

/// Dedicated movie contract. Parsing/admission never invokes a second sampler
/// or relabels a movie as DFR. Prepared RGB is frozen independently of its movie.
public struct MLXMovieUpscaleRequest: Sendable {
  public struct Source: Codable, Sendable {
    public let path:String,sha256:String,rgbPath:String,rgbSHA256:String
    public let width:Int,height:Int,frames:Int,fps:Double,startSeconds:Double,durationSeconds:Double
    enum CodingKeys:String,CodingKey,CaseIterable {
      case path,sha256,width,height,frames,fps
      case rgbPath="rgb_path",rgbSHA256="rgb_sha256",startSeconds="start_seconds",durationSeconds="duration_seconds"
    }
    public init(from decoder:Decoder) throws {
      try MLXMovieUpscaleRequest.exactKeys(decoder,CodingKeys.allCases.map(\.rawValue))
      let c=try decoder.container(keyedBy:CodingKeys.self)
      path=try c.decode(String.self,forKey:.path);sha256=try c.decode(String.self,forKey:.sha256)
      rgbPath=try c.decode(String.self,forKey:.rgbPath);rgbSHA256=try c.decode(String.self,forKey:.rgbSHA256)
      width=try c.decode(Int.self,forKey:.width);height=try c.decode(Int.self,forKey:.height)
      frames=try c.decode(Int.self,forKey:.frames);fps=try c.decode(Double.self,forKey:.fps)
      startSeconds=try c.decode(Double.self,forKey:.startSeconds);durationSeconds=try c.decode(Double.self,forKey:.durationSeconds)
      guard MLXMovieUpscaleRequest.validPath(path),MLXMovieUpscaleRequest.validPath(rgbPath),
        MLXMovieUpscaleRequest.validSHA(sha256),MLXMovieUpscaleRequest.validSHA(rgbSHA256),
        startSeconds.isFinite,startSeconds>=0,durationSeconds.isFinite,durationSeconds>0,
        fps.isFinite,(1...60).contains(fps),(32...8192).contains(width),(32...8192).contains(height),frames>0,frames<=1_000_000,
        abs(durationSeconds-Double(frames)/fps)<=0.05 else {
        throw LTXError.invalid("Movie source requires bounded frozen paths/hashes and an exact frame interval.")
      }
    }
  }
  public enum Anchors:String,Codable,Sendable { case none,first,firstLast="first_last" }
  public enum AudioPolicy:String,Codable,Sendable { case source,sidecar,silence }
  public struct AudioSource:Codable,Sendable {
    public let path:String,sha256:String,startSeconds:Double,durationSeconds:Double?
    enum CodingKeys:String,CodingKey,CaseIterable { case path,sha256;case startSeconds="start_seconds",durationSeconds="duration_seconds" }
    public init(from decoder:Decoder) throws {
      try MLXMovieUpscaleRequest.exactKeys(decoder,CodingKeys.allCases.map(\.rawValue))
      let c=try decoder.container(keyedBy:CodingKeys.self)
      path=try c.decode(String.self,forKey:.path);sha256=try c.decode(String.self,forKey:.sha256)
      startSeconds=try c.decode(Double.self,forKey:.startSeconds);durationSeconds=try c.decodeIfPresent(Double.self,forKey:.durationSeconds)
      guard MLXMovieUpscaleRequest.validPath(path),MLXMovieUpscaleRequest.validSHA(sha256),startSeconds.isFinite,startSeconds>=0,
        durationSeconds.map({ $0.isFinite && $0>0 }) ?? true else { throw LTXError.invalid("Movie sidecar requires a frozen valid audio interval.") }
    }
  }

  private struct Body: Decodable {
    let diffusionVAE:MLXDiffusionVideoSettings?
    let version:Int,engine:String,task:String
    let source:Source,components:[String:String],mode:MLXMovieUpscalePlan.Mode
    let sizePolicy:MLXMovieUpscalePlan.SizePolicy,outputDirectory:String,prompt:String,seed:UInt64
    let refinementStrength:Double,anchorStrength:Float,pixelStrength:Float,maximumAudioDriftSeconds:Double
    let anchors:Anchors,referenceImages:[MLXImageReference]
    let audioPolicy:AudioPolicy,audioSource:AudioSource?,referenceImageSHA256:[String:String]
    let chunking:Bool,chunkFrameMegapixelBudget:Double,resume:Bool,keepChunks:Bool
    enum CodingKeys:String,CodingKey,CaseIterable {
      case diffusionVAE="diffusion_vae"
      case version,engine,task,source,components,mode,prompt,seed,anchors,resume
      case audioPolicy="audio_policy",audioSource="audio_source",referenceImageSHA256="reference_image_sha256"
      case sizePolicy="size_policy",outputDirectory="output_directory",refinementStrength="refinement_strength"
      case anchorStrength="anchor_strength",pixelStrength="pixel_strength",referenceImages="reference_images"
      case maximumAudioDriftSeconds="maximum_audio_drift_seconds",chunking,chunkFrameMegapixelBudget="chunk_frame_megapixel_budget",keepChunks="keep_chunks"
    }
    init(from decoder:Decoder) throws {
      try MLXMovieUpscaleRequest.exactKeys(decoder,CodingKeys.allCases.map(\.rawValue),optional:["audio_policy","audio_source","reference_image_sha256","diffusion_vae"])
      let c=try decoder.container(keyedBy:CodingKeys.self)
      diffusionVAE=try c.decodeIfPresent(MLXDiffusionVideoSettings.self,forKey:.diffusionVAE)
      audioPolicy=try c.decodeIfPresent(AudioPolicy.self,forKey:.audioPolicy) ?? .source
      audioSource=try c.decodeIfPresent(AudioSource.self,forKey:.audioSource)
      referenceImageSHA256=try c.decodeIfPresent([String:String].self,forKey:.referenceImageSHA256) ?? [:]
      version=try c.decode(Int.self,forKey:.version);engine=try c.decode(String.self,forKey:.engine);task=try c.decode(String.self,forKey:.task)
      source=try c.decode(Source.self,forKey:.source);components=try c.decode([String:String].self,forKey:.components)
      mode=try c.decode(MLXMovieUpscalePlan.Mode.self,forKey:.mode);sizePolicy=try c.decode(MLXMovieUpscalePlan.SizePolicy.self,forKey:.sizePolicy)
      outputDirectory=try c.decode(String.self,forKey:.outputDirectory);prompt=try c.decode(String.self,forKey:.prompt);seed=try c.decode(UInt64.self,forKey:.seed)
      refinementStrength=try c.decode(Double.self,forKey:.refinementStrength);anchors=try c.decode(Anchors.self,forKey:.anchors)
      anchorStrength=try c.decode(Float.self,forKey:.anchorStrength);pixelStrength=try c.decode(Float.self,forKey:.pixelStrength)
      maximumAudioDriftSeconds=try c.decode(Double.self,forKey:.maximumAudioDriftSeconds);referenceImages=try c.decode([MLXImageReference].self,forKey:.referenceImages)
      chunking=try c.decode(Bool.self,forKey:.chunking);chunkFrameMegapixelBudget=try c.decode(Double.self,forKey:.chunkFrameMegapixelBudget)
      resume=try c.decode(Bool.self,forKey:.resume);keepChunks=try c.decode(Bool.self,forKey:.keepChunks)
    }
  }
  private struct Key:CodingKey {
    let stringValue:String;var intValue:Int? { nil }
    init(stringValue:String) { self.stringValue=stringValue };init?(intValue:Int) { nil }
  }
  fileprivate static func exactKeys(_ decoder:Decoder,_ keys:[String],optional:Set<String>=[]) throws {
    let c=try decoder.container(keyedBy:Key.self)
    let actual=Set(c.allKeys.map(\.stringValue)),allowed=Set(keys)
    guard actual.isSubset(of:allowed),allowed.subtracting(optional).isSubset(of:actual) else {
      throw LTXError.invalid("Movie upscale has missing or unsupported fields.")
    }
  }
  static func validPath(_ path:String)->Bool { path.hasPrefix("/") && path.utf8.count<=4096 && !path.utf8.contains(0) }
  static func validSHA(_ value:String)->Bool { value.utf8.count==64 && value.utf8.allSatisfy { (48...57).contains($0) || (97...102).contains($0) } }
  public let source:Source,components:[String:String],plan:MLXMovieUpscalePlan
  public let outputDirectory:String,prompt:String,seed:UInt64,anchors:Anchors,anchorStrength:Float,pixelStrength:Float
  public let referenceImages:[MLXImageReference],maximumAudioDriftSeconds:Double,chunking:Bool,chunkFrameMegapixelBudget:Double,resume:Bool,keepChunks:Bool
  public let audioPolicy:AudioPolicy,audioSource:AudioSource?,referenceImageSHA256:[String:String]
  public let diffusionVAE:MLXDiffusionVideoSettings?
  public let contractSHA256:String
  public init(data:Data,outputDirectory:URL) throws {
    guard data.count<=1024*1024 else { throw LTXError.invalid("Movie request exceeds its 1 MiB metadata bound.") }
    let body=try JSONDecoder().decode(Body.self,from:data)
    guard body.version==1,body.engine=="ltx25",body.task=="video_upscale",
      Self.validPath(body.outputDirectory),outputDirectory.isFileURL,
      try MLXTransformerPageConverter.canonicalLocalURL(URL(fileURLWithPath:body.outputDirectory)) == MLXTransformerPageConverter.canonicalLocalURL(outputDirectory),
      body.prompt.utf8.count<=65536,!body.prompt.utf8.contains(0),
      body.anchorStrength.isFinite,(0...1).contains(body.anchorStrength),
      body.pixelStrength.isFinite,(0.05...2).contains(body.pixelStrength),
      body.maximumAudioDriftSeconds.isFinite,(0...0.5).contains(body.maximumAudioDriftSeconds),
      body.chunkFrameMegapixelBudget.isFinite,body.chunkFrameMegapixelBudget>0,
      body.referenceImages.count<=2,Set(body.referenceImages.map(\.role)).count==body.referenceImages.count,
      body.referenceImages.allSatisfy({ ["first","last"].contains($0.role) && $0.strength==body.anchorStrength && Self.validPath($0.path) }),
      Set(body.referenceImageSHA256.keys)==Set(body.referenceImages.map(\.path)),body.referenceImageSHA256.values.allSatisfy(Self.validSHA),
      (body.audioPolicy == .sidecar) == (body.audioSource != nil),
      !body.resume || body.chunking else {
      throw LTXError.invalid("Movie upscale version, source, output, endpoint or explicit chunk settings are invalid.")
    }
    let common:Set<String>=["video_checkpoint","spatial_upscaler_checkpoint"]
    let refinement:Set<String>=["gemma_root","connector_checkpoint","transformer_root","audio_checkpoint"]
    let needed=common.union(body.mode == .latentOnly ? [] : refinement).union(body.mode == .pixelSpatial ? ["pixel_spatial_adapter"] : [])
    guard Set(body.components.keys)==needed,body.components.values.allSatisfy(Self.validPath),
      body.mode != .latentOnly || (body.prompt.isEmpty && body.anchors == .none && body.referenceImages.isEmpty),
      body.mode == .latentOnly || !body.prompt.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty else {
      throw LTXError.invalid("Movie mode requires its exact native components; latent-only cannot silently ignore prompt or anchors.")
    }
    diffusionVAE=body.diffusionVAE
    source=body.source;components=body.components
    plan=try MLXMovieUpscalePlan(mode:body.mode,width:source.width,height:source.height,frames:source.frames,fps:source.fps,
      sizePolicy:body.sizePolicy,refinementStrength:body.refinementStrength)
    guard plan.size.width<=2048,plan.size.height<=2048 else { throw LTXError.invalid("Movie processed source exceeds its native encoder canvas.") }
    audioPolicy=body.audioPolicy;audioSource=body.audioSource;referenceImageSHA256=body.referenceImageSHA256
    self.outputDirectory=body.outputDirectory;prompt=body.prompt;seed=body.seed;anchors=body.anchors
    anchorStrength=body.anchorStrength;pixelStrength=body.pixelStrength;referenceImages=body.referenceImages
    maximumAudioDriftSeconds=body.maximumAudioDriftSeconds;chunking=body.chunking;chunkFrameMegapixelBudget=body.chunkFrameMegapixelBudget
    resume=body.resume;keepChunks=body.keepChunks
    // Output relocation is the only allowed change in a resumable generation contract.
    var json=try JSONSerialization.jsonObject(with:data) as! [String:Any];json.removeValue(forKey:"output_directory")
    json.removeValue(forKey:"resume");json.removeValue(forKey:"keep_chunks")
    contractSHA256=SHA256.hash(data:try JSONSerialization.data(withJSONObject:json,options:[.sortedKeys])).map { String(format:"%02x",$0) }.joined()
  }
  public func chunkPlans(cutFrames:[Int]=[]) throws -> [MLXMovieUpscalePlan.Chunk] {
    let nativeAudioMaximum=Int(floor(min(4097,1501*plan.fps/25)))
    let nativePaddedMaximum=1+8*max(0,(nativeAudioMaximum-1)/8)
    let perFrame=Double(plan.size.outputWidth)*Double(plan.size.outputHeight)/1_000_000
    // Workload partitioning must also obey the trained audio clock. Padding is
    // included; a long low-resolution source must not bypass this stage bound.
    let audioBudget=perFrame*(Double(nativePaddedMaximum)+0.25)
    let budget=chunking ? min(chunkFrameMegapixelBudget,audioBudget) : max(1,plan.outputFrameMegapixels)
    let chunks=try plan.chunks(frameMegapixelBudget:budget,cutFrames:cutFrames)
    guard chunks.allSatisfy({ $0.paddedFrames<=4097 && (try? AVGeometry(width:plan.size.outputWidth,height:plan.size.outputHeight,frames:$0.paddedFrames,fps:plan.fps)) != nil }) else {
      throw LTXError.invalid("Movie chunks exceed native padded video/audio token admission. Enable a smaller chunk budget before inference.")
    }
    return chunks
  }
  public func validateMediaSources() throws {
    try NativeMediaSource(path:source.path,sha256:source.sha256).verify()
    try NativeMediaSource(path:source.rgbPath,sha256:source.rgbSHA256).verify()
    if let audioSource { try NativeMediaSource(path:audioSource.path,sha256:audioSource.sha256).verify() }
    for (path,sha) in referenceImageSHA256 { try NativeMediaSource(path:path,sha256:sha).verify() }
    let expected=Int64(plan.frames)*Int64(plan.size.width)*Int64(plan.size.height)*3
    guard (try FileManager.default.attributesOfItem(atPath:source.rgbPath)[.size] as? Int64)==expected else {
      throw LTXError.invalid("Frozen movie RGB does not match its processed grid and visible frame count.")
    }
  }
}
