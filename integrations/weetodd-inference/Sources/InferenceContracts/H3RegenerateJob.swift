import CryptoKit
import Foundation

/// Lossless worker transport, separate from the engine recipe whose derived
/// embedding/noise/silence identities are known only after preparation.
public struct H3RegenerateJob:Codable,Equatable,Sendable {
  public enum SourceKind:String,Codable,Sendable { case rgb24,normalizedVideoRows }
  public enum EncoderProfile:String,Codable,Sendable { case nativeNormalizedLatents,nativePosteriorMeanF16,publisherPosteriorFP32 }
  public enum Schedule:String,Codable,Sendable { case publisherThreeStep,publisherTwoStep }
  public enum UpscaleMode:String,Codable,Sendable { case directPublisher,multiHopNative }
  public enum Decoder:String,Codable,Sendable { case releasedH3,fastH3 }
  public struct File:Codable,Equatable,Sendable {
    public let path:String,sha256:String
    public init(path:String,sha256:String) { self.path=path;self.sha256=sha256 }
  }
  public struct Component:Codable,Equatable,Sendable {
    public let path:String,identity:NativeComponentIdentity
    public init(path:String,identity:NativeComponentIdentity) { self.path=path;self.identity=identity }
  }
  public struct Source:Codable,Equatable,Sendable {
    public let kind:SourceKind,payload:File,originalSHA256:String,audioSHA256:String
    public let width:Int,height:Int,visibleFrames:Int,paddedFrames:Int,fps:Double,encoderProfile:EncoderProfile
    public init(kind:SourceKind,payload:File,originalSHA256:String,audioSHA256:String,width:Int,height:Int,
      visibleFrames:Int,paddedFrames:Int,fps:Double,encoderProfile:EncoderProfile) {
      self.kind=kind;self.payload=payload;self.originalSHA256=originalSHA256;self.audioSHA256=audioSHA256
      self.width=width;self.height=height;self.visibleFrames=visibleFrames;self.paddedFrames=paddedFrames
      self.fps=fps;self.encoderProfile=encoderProfile
    }
    public var latentFrames:Int { 2+5*((paddedFrames-5)/17) }
    public var payloadBytes:UInt64 {
      kind == .rgb24 ? UInt64(visibleFrames*width*height*3):UInt64(latentFrames*(width/32)*(height/32)*96*4)
    }
  }
  public struct Qwen:Codable,Equatable,Sendable {
    public let checkpointRoot:String,tokenizer:File,files:[File]
    public init(checkpointRoot:String,tokenizer:File,files:[File]) { self.checkpointRoot=checkpointRoot;self.tokenizer=tokenizer;self.files=files }
  }
  public struct Components:Codable,Equatable,Sendable {
    public let transformer:Component,refiner:Component,sharpCurve:Component,sharpInput:Component
    public let latentUpscaler:Component,videoVAE:Component,audioVAE:Component
    public let publisherEncoder:Component?,fastDecoder:Component?
    public init(transformer:Component,refiner:Component,sharpCurve:Component,sharpInput:Component,
      latentUpscaler:Component,videoVAE:Component,audioVAE:Component,publisherEncoder:Component?=nil,fastDecoder:Component?=nil) {
      self.transformer=transformer;self.refiner=refiner;self.sharpCurve=sharpCurve;self.sharpInput=sharpInput
      self.latentUpscaler=latentUpscaler;self.videoVAE=videoVAE;self.audioVAE=audioVAE
      self.publisherEncoder=publisherEncoder;self.fastDecoder=fastDecoder
    }
    public var all:[Component] { [transformer,refiner,sharpCurve,sharpInput,latentUpscaler,videoVAE,audioVAE]+[publisherEncoder,fastDecoder].compactMap{$0} }
  }
  public struct Target:Codable,Equatable,Sendable {
    public let width:Int,height:Int
    public init(width:Int,height:Int) { self.width=width;self.height=height }
  }
  public struct Budget:Codable,Equatable,Sendable {
    public let maximumStageBytes:UInt64,availableTemporaryBytes:UInt64
    public init(maximumStageBytes:UInt64,availableTemporaryBytes:UInt64) { self.maximumStageBytes=maximumStageBytes;self.availableTemporaryBytes=availableTemporaryBytes }
  }
  public let version:Int,engine:String,task:String,source:Source,prompt:String,promptSHA256:String,qwen:Qwen,components:Components
  public let schedule:Schedule,seed:UInt64,posteriorSeed:UInt64?,target:Target,latentUpscaleMode:UpscaleMode,decoder:Decoder,budget:Budget
  public init(source:Source,prompt:String,promptSHA256:String,qwen:Qwen,components:Components,schedule:Schedule,
    seed:UInt64,posteriorSeed:UInt64?,target:Target,latentUpscaleMode:UpscaleMode,decoder:Decoder,budget:Budget) throws {
    version=1;engine="h3";task="h3_regenerate";self.source=source;self.prompt=prompt;self.promptSHA256=promptSHA256
    self.qwen=qwen;self.components=components;self.schedule=schedule;self.seed=seed;self.posteriorSeed=posteriorSeed
    self.target=target;self.latentUpscaleMode=latentUpscaleMode;self.decoder=decoder;self.budget=budget
    try validate()
  }
  public func validate() throws {
    func digest(_ value:String)->Bool { value.utf8.count == 64 && value.utf8.allSatisfy { (48...57).contains($0) || (97...102).contains($0) } }
    func path(_ value:String)->Bool { value.hasPrefix("/") && value.utf8.count <= 4096 && !value.unicodeScalars.contains { $0.value < 32 } }
    func file(_ value:File)->Bool { path(value.path) && digest(value.sha256) }
    guard version == 1,engine == "h3",task == "h3_regenerate",file(source.payload),digest(source.originalSHA256),digest(source.audioSHA256),
      [source.width,source.height].allSatisfy({ (32...4096).contains($0) && $0.isMultiple(of:32) }),
      [target.width,target.height].allSatisfy({ (32...4096).contains($0) }),
      target.width >= source.width,target.height >= source.height,
      (5...362).contains(source.paddedFrames),(source.paddedFrames-5).isMultiple(of:17),
      (1...source.paddedFrames).contains(source.visibleFrames),source.paddedFrames-source.visibleFrames <= 16,
      source.fps == 24,prompt.utf8.count <= 100_000,digest(promptSHA256),
      SHA256.hash(data:Data(prompt.utf8)).map({String(format:"%02x",$0)}).joined() == promptSHA256,
      path(qwen.checkpointRoot),file(qwen.tokenizer),(1...53).contains(qwen.files.count),qwen.files.allSatisfy(file),
      Set(qwen.files.map { URL(fileURLWithPath:$0.path).standardizedFileURL.path }).count == qwen.files.count,
      budget.maximumStageBytes > 0,budget.maximumStageBytes <= 192*1024*1024*1024,budget.availableTemporaryBytes > 0 else {
      throw ContractError.invalid("Invalid immutable H3 regeneration source, timing, model identities, prompt or budget.")
    }
    guard (source.kind == .normalizedVideoRows) == (source.encoderProfile == .nativeNormalizedLatents),
      source.kind != .rgb24 || source.width*source.height <= 1024*1024,
      (source.encoderProfile == .publisherPosteriorFP32) == (posteriorSeed != nil),
      (source.encoderProfile == .publisherPosteriorFP32) == (components.publisherEncoder != nil),
      (decoder == .fastH3) == (components.fastDecoder != nil) else {
      throw ContractError.invalid("H3 regeneration preparation or decoder assets differ from the selected explicit profile.")
    }
    let roles:[NativeComponentRole]=[.transformer,.refiner,.sharpCurve,.sharpInput,.latentUpscaler,.videoVAE,.audioVAE]
    let required=[components.transformer,components.refiner,components.sharpCurve,components.sharpInput,components.latentUpscaler,components.videoVAE,components.audioVAE]
    for (component,role) in zip(required,roles) {
      try component.identity.validate()
      guard path(component.path),component.identity.family == .h3,component.identity.role == role else {
        throw ContractError.invalid("H3 regeneration component has the wrong role or path.")
      }
    }
    for (component,role) in [(components.publisherEncoder,NativeComponentRole.videoVAE),(components.fastDecoder,.decoder)] {
      if let component {
        try component.identity.validate()
        guard path(component.path),component.identity.family == .h3,component.identity.role == role else {
          throw ContractError.invalid("H3 preparation/decoder component has the wrong identity role.")
        }
      }
    }
    try NativeCapabilityDescriptor.h3Regenerate.validate(components:required.map(\.identity),sampling:NativeCapabilityDescriptor.h3Regenerate.sampling)
  }
  public static func decode(_ data:Data) throws -> Self {
    func object(_ value:Any?,_ keys:Set<String>) throws -> [String:Any] {
      guard let object=value as? [String:Any],Set(object.keys).isSubset(of:keys) else { throw ContractError.invalid("Unknown H3 regeneration worker field.") }
      return object
    }
    guard data.count <= 1024*1024 else { throw ContractError.invalid("H3 regeneration wrapper exceeds 1 MiB.") }
    let root=try object(JSONSerialization.jsonObject(with:data),["version","engine","task","source","prompt","promptSHA256","qwen","components","schedule","seed","posteriorSeed","target","latentUpscaleMode","decoder","budget"])
    let source=try object(root["source"],["kind","payload","originalSHA256","audioSHA256","width","height","visibleFrames","paddedFrames","fps","encoderProfile"])
    _ = try object(source["payload"],["path","sha256"])
    let qwen=try object(root["qwen"],["checkpointRoot","tokenizer","files"])
    _ = try object(qwen["tokenizer"],["path","sha256"])
    guard let files=qwen["files"] as? [Any] else { throw ContractError.invalid("Missing Qwen identity set.") }
    for file in files { _ = try object(file,["path","sha256"]) }
    let components=try object(root["components"],["transformer","refiner","sharpCurve","sharpInput","latentUpscaler","videoVAE","audioVAE","publisherEncoder","fastDecoder"])
    for value in components.values where !(value is NSNull) { _ = try object(value,["path","identity"]) }
    _ = try object(root["target"],["width","height"])
    _ = try object(root["budget"],["maximumStageBytes","availableTemporaryBytes"])
    let result=try JSONDecoder().decode(Self.self,from:data);try result.validate();return result
  }
}
