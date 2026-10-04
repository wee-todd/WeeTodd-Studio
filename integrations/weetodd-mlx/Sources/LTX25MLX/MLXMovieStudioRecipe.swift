import Foundation
import LTX25Engine

/// A frozen Studio wrapper authenticates the dedicated movie request and its
/// exact media inventory. Compilation performs no weighted model work.
public enum MLXMovieStudioRecipe {
  public struct Compiled:Sendable {
    public let request:MLXMovieUpscaleRequest
    public let originalRecipeBytes:Data
    public let requestBytes:Data
    public let isStudioWrapper:Bool
  }
  private struct Input:Equatable {
    let kind:String,sha256:String
  }
  public static func compile(_ data:Data,outputDirectory:URL) throws -> Compiled {
    guard data.count<=1024*1024,outputDirectory.isFileURL,
      MLXMovieUpscaleRequest.validPath(outputDirectory.path),
      let root=try JSONSerialization.jsonObject(with:data) as? [String:Any] else {
      throw LTXError.invalid("Movie Studio recipe needs bounded metadata and an absolute authenticated output.")
    }
    guard root["movie_upscale"] != nil else {
      let request=try MLXMovieUpscaleRequest(data:data,outputDirectory:outputDirectory)
      try request.validateMediaSources()
      return Compiled(request:request,originalRecipeBytes:data,requestBytes:data,isStudioWrapper:false)
    }
    guard Set(root.keys)==["format","engine","prompt","config","components","conditioning","movie_upscale"],
      root["format"] as? String == "weetodd-headless-v2",root["engine"] as? String == "ltx25",
      let prompt=root["prompt"] as? String,let config=root["config"] as? [String:Any],config.isEmpty,
      let components=root["components"] as? [String:String],
      let conditioning=root["conditioning"] as? [String:Any],
      Set(conditioning.keys)==["task","inputs"],conditioning["task"] as? String == "video_upscale",
      let inputs=conditioning["inputs"] as? [[String:Any]],inputs.count<=5,
      var nested=root["movie_upscale"] as? [String:Any],
      let originalOutput=nested["output_directory"] as? String,MLXMovieUpscaleRequest.validPath(originalOutput) else {
      throw LTXError.invalid("Movie Studio wrapper has missing, conflicting or unsupported fields.")
    }
    // Admit the original request before relocation; no other transport field is
    // repaired. JSONDecoder rejects Boolean integer fields in this strict v1.
    let originalRequest=try MLXMovieUpscaleRequest(
      data:JSONSerialization.data(withJSONObject:nested,options:[.sortedKeys,.withoutEscapingSlashes]),
      outputDirectory:URL(fileURLWithPath:originalOutput))
    guard prompt==originalRequest.prompt,components==originalRequest.components else {
      throw LTXError.invalid("Movie Studio prompt and components conflict with the frozen nested request.")
    }
    var expected:[String:Input]=[:]
    func append(_ path:String,_ kind:String,_ sha256:String) throws {
      let key=try canonicalPath(path),value=Input(kind:kind,sha256:sha256)
      guard expected[key]==nil else { throw LTXError.invalid("Movie media roles require distinct canonical source paths.") }
      expected[key]=value
    }
    try append(originalRequest.source.path,"video",originalRequest.source.sha256)
    try append(originalRequest.source.rgbPath,"rgb24",originalRequest.source.rgbSHA256)
    if let audio=originalRequest.audioSource { try append(audio.path,"audio",audio.sha256) }
    for image in originalRequest.referenceImages {
      guard let sha=originalRequest.referenceImageSHA256[image.path] else {
        throw LTXError.invalid("Movie endpoint is missing its frozen hash.")
      }
      try append(image.path,"image",sha)
    }
    var actual:[String:Input]=[:]
    for input in inputs {
      try Task.checkCancellation()
      guard Set(input.keys)==["path","kind","sha256"],let path=input["path"] as? String,
        let kind=input["kind"] as? String,["video","rgb24","audio","image"].contains(kind),
        let sha=input["sha256"] as? String,MLXMovieUpscaleRequest.validSHA(sha) else {
        throw LTXError.invalid("Movie Studio input inventory has missing or unsupported fields.")
      }
      let key=try canonicalPath(path)
      guard actual[key]==nil else { throw LTXError.invalid("Movie Studio input inventory repeats a canonical path.") }
      actual[key]=Input(kind:kind,sha256:sha)
    }
    guard actual==expected else {
      throw LTXError.invalid("Movie Studio input inventory does not exactly cover its frozen media roles and hashes.")
    }
    try originalRequest.validateMediaSources()
    nested["output_directory"]=try MLXTransformerPageConverter.canonicalLocalURL(outputDirectory).path
    let bytes=try JSONSerialization.data(withJSONObject:nested,options:[.sortedKeys,.withoutEscapingSlashes])
    let request=try MLXMovieUpscaleRequest(data:bytes,outputDirectory:outputDirectory)
    return Compiled(request:request,originalRecipeBytes:data,requestBytes:bytes,isStudioWrapper:true)
  }
  private static func canonicalPath(_ path:String) throws -> String {
    guard MLXMovieUpscaleRequest.validPath(path) else { throw LTXError.invalid("Movie Studio inputs require bounded absolute paths.") }
    return try MLXTransformerPageConverter.canonicalLocalURL(URL(fileURLWithPath:path)).path
  }
}
