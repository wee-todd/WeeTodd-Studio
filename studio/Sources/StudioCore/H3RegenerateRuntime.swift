import CryptoKit
import Foundation
import InferenceContracts

public struct H3RegenerateRuntime: Codable, Equatable, Sendable {
  public var ffmpegPath = "", ffprobePath = "", workerPath = ""
  public init() {}
  public init(ffmpegPath:String, ffprobePath:String, workerPath:String) {
    self.ffmpegPath=ffmpegPath; self.ffprobePath=ffprobePath; self.workerPath=workerPath
  }
  private enum CodingKeys:String,CodingKey,CaseIterable {case ffmpegPath,ffprobePath,workerPath}
  public init(from decoder:Decoder) throws {
    try FlashVSRStrictObject.check(decoder,allowed:Set(CodingKeys.allCases.map(\.rawValue)))
    let c=try decoder.container(keyedBy:CodingKeys.self)
    ffmpegPath=try c.decodeIfPresent(String.self,forKey:.ffmpegPath) ?? ""
    ffprobePath=try c.decodeIfPresent(String.self,forKey:.ffprobePath) ?? ""
    workerPath=try c.decodeIfPresent(String.self,forKey:.workerPath) ?? ""
  }
  var mediaRuntime:FlashVSRRuntime {
    var result=FlashVSRRuntime(); result.ffmpegPath=ffmpegPath; result.ffprobePath=ffprobePath; return result
  }
  func resolved() throws -> Self {
    var result=self
    if result.workerPath.isEmpty {
      result.workerPath=Bundle.main.executableURL?.deletingLastPathComponent().appendingPathComponent("WeeToddH3MLXWorker").path ?? ""
    }
    for path in [result.ffmpegPath,result.ffprobePath,result.workerPath] {
      guard path.hasPrefix("/"),!path.utf8.contains(0),FileManager.default.isExecutableFile(atPath:path) else {
        throw StudioError.invalid("Connect the native H3 worker and existing FFmpeg/FFprobe utilities before restoration.")
      }
    }
    return result
  }
}

/// A complete normalized source interval, never an implicit arbitrary latent file.
public struct H3RegenerateLatentSource: Codable, Equatable, Sendable {
  public var version=1
  public var file:FlashVSRFile,sourceSHA256:String,firstFrame:Int,visibleFrames:Int,paddedFrames:Int,width:Int,height:Int
  public var normalization="h3_sampler_normalized_v1",dtype="F32LE",fps=24
  public init(file:FlashVSRFile,sourceSHA256:String,firstFrame:Int,visibleFrames:Int,paddedFrames:Int,width:Int,height:Int) {
    self.file=file;self.sourceSHA256=sourceSHA256;self.firstFrame=firstFrame;self.visibleFrames=visibleFrames
    self.paddedFrames=paddedFrames;self.width=width;self.height=height
  }
  private enum CodingKeys:String,CodingKey,CaseIterable {case version,file,sourceSHA256,firstFrame,visibleFrames,paddedFrames,width,height,normalization,dtype,fps}
  private struct FileIdentity:Decodable {
    let path:String,sha256:String,bytes:Int64
    private enum CodingKeys:String,CodingKey,CaseIterable {case path,sha256,bytes}
    init(from decoder:Decoder) throws {
      try FlashVSRStrictObject.check(decoder,allowed:Set(CodingKeys.allCases.map(\.rawValue)))
      let c=try decoder.container(keyedBy:CodingKeys.self)
      path=try c.decode(String.self,forKey:.path);sha256=try c.decode(String.self,forKey:.sha256);bytes=try c.decode(Int64.self,forKey:.bytes)
    }
  }
  public init(from decoder:Decoder) throws {
    try FlashVSRStrictObject.check(decoder,allowed:Set(CodingKeys.allCases.map(\.rawValue)))
    let c=try decoder.container(keyedBy:CodingKeys.self),f=try c.decode(FileIdentity.self,forKey:.file)
    file = .init(path:f.path,sha256:f.sha256,bytes:f.bytes)
    version=try c.decode(Int.self,forKey:.version);sourceSHA256=try c.decode(String.self,forKey:.sourceSHA256)
    firstFrame=try c.decode(Int.self,forKey:.firstFrame);visibleFrames=try c.decode(Int.self,forKey:.visibleFrames)
    paddedFrames=try c.decode(Int.self,forKey:.paddedFrames);width=try c.decode(Int.self,forKey:.width);height=try c.decode(Int.self,forKey:.height)
    normalization=try c.decode(String.self,forKey:.normalization);dtype=try c.decode(String.self,forKey:.dtype);fps=try c.decode(Int.self,forKey:.fps)
    try validate(sourceSHA256:sourceSHA256,firstFrame:firstFrame,frames:visibleFrames)
  }
  public func validate(sourceSHA256:String,firstFrame:Int,frames:Int) throws {
    guard version==1,normalization=="h3_sampler_normalized_v1",dtype=="F32LE",fps==24,
      self.sourceSHA256==sourceSHA256,self.firstFrame==firstFrame,firstFrame>=0,visibleFrames==frames,
      sourceSHA256.count==64,sourceSHA256.utf8.allSatisfy({(48...57).contains($0)||(97...102).contains($0)}),
      file.sha256.count==64,file.sha256.utf8.allSatisfy({(48...57).contains($0)||(97...102).contains($0)}),file.path.hasPrefix("/"),file.path.utf8.count<=4096,!file.path.utf8.contains(0),
      (5...362).contains(paddedFrames),(paddedFrames-5)%17==0,(1...paddedFrames).contains(frames),paddedFrames-frames<=16,
      [width,height].allSatisfy({(32...4096).contains($0)&&$0%32==0}),
      file.bytes==Int64((2+5*((paddedFrames-5)/17))*(width/32)*(height/32)*96*4) else {
      throw StudioError.invalid("The explicit normalized H3 latent artifact differs from the source interval, grid, normalization or F32 row layout. No movie fallback was applied.")
    }
  }
}
public struct H3RegenerateRequest: Codable, Equatable, Sendable {
  public var format="weetodd-h3-regenerate-v1"
  public var sourcePath:String,sourceSHA256:String?,firstFrame=0,frameCount:Int?
  public var settings:H3RegenerateSettings,runtime:H3RegenerateRuntime,nativeLatents:H3RegenerateLatentSource?
  public init(sourcePath:String,settings:H3RegenerateSettings = .init(),runtime:H3RegenerateRuntime) {
    self.sourcePath=sourcePath;self.settings=settings;self.runtime=runtime
  }
  private enum CodingKeys:String,CodingKey,CaseIterable {case format,sourcePath,sourceSHA256,firstFrame,frameCount,settings,runtime,nativeLatents}
  public init(from decoder:Decoder) throws {
    try FlashVSRStrictObject.check(decoder,allowed:Set(CodingKeys.allCases.map(\.rawValue)))
    let c=try decoder.container(keyedBy:CodingKeys.self)
    format=try c.decode(String.self,forKey:.format);sourcePath=try c.decode(String.self,forKey:.sourcePath)
    sourceSHA256=try c.decodeIfPresent(String.self,forKey:.sourceSHA256);firstFrame=try c.decodeIfPresent(Int.self,forKey:.firstFrame) ?? 0
    frameCount=try c.decodeIfPresent(Int.self,forKey:.frameCount);settings=try c.decode(H3RegenerateSettings.self,forKey:.settings)
    runtime=try c.decode(H3RegenerateRuntime.self,forKey:.runtime);nativeLatents=try c.decodeIfPresent(H3RegenerateLatentSource.self,forKey:.nativeLatents)
    guard format=="weetodd-h3-regenerate-v1",sourcePath.hasPrefix("/"),sourcePath.utf8.count<=4096,!sourcePath.utf8.contains(0),firstFrame>=0,frameCount==nil || frameCount!>0 else {throw StudioError.invalid("Invalid H3 source request interval or local path.")}
  }

}
public struct H3RegenerateResources: Codable, Sendable {
  public let components:H3RegenerateJob.Components,qwen:H3RegenerateJob.Qwen
  public let modelFiles:[FlashVSRFile],engineFiles:[FlashVSRFile],profile:FlashVSRFile
  public let runtime:H3RegenerateRuntime
  public var files:[FlashVSRFile] { [profile]+modelFiles+engineFiles }
}
public struct H3RegenerateResolvedPlan: Codable, Sendable {
  public let request:H3RegenerateRequest,source:FlashVSRFile,media:FlashVSRMediaInfo,resources:H3RegenerateResources
  public let frames:Int,paddedFrames:Int,sourceWidth:Int,sourceHeight:Int,temporaryBytes:Int64
  public let preparationPolicy:String
  public var fingerprint:String { get throws {
    let encoder=JSONEncoder();encoder.outputFormatting=[.sortedKeys]
    return H3RegeneratePreparation.hash(try encoder.encode(self))
  } }
}
public struct H3RegenerateReceipt: Codable, Sendable {
  public var format="weetodd-h3-regenerate-result-v1"
  public let requestFingerprint:String,plan:H3RegenerateResolvedPlan
  public let source:FlashVSRFile,video:FlashVSRFile,restored:FlashVSRFile,audio:FlashVSRFile?
  public let workerResult:FlashVSRFile,workerJob:FlashVSRFile,refinedLatents:FlashVSRFile
  public let refinedLatentsShape:[Int],elapsedSeconds:Double
  public var frames:Int { plan.frames }
  public var fps:FlashVSRFrameRate { plan.media.fps }
  public var settings:H3RegenerateSettings { plan.request.settings }
  public var workerFiles:[FlashVSRFile] { plan.resources.engineFiles }
}
