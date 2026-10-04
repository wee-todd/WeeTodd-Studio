import CoreFoundation
import Foundation

/// A source-movie edit adopts its exact visible source interval, rather than
/// inheriting an unrelated default timeline duration or AAC container padding.
public enum NativeLTXMovieAcceptance {
  public static func duration(prepared:[String:Any]?,result:[String:Any],media:[String:Any])throws->Double? {
    let metadata=result["metadata"] as? [String:Any] ?? [:]
    guard prepared?["task"] as? String == "video_upscale" else {
      guard metadata["task"] as? String != "video_upscale" else { throw StudioError.invalid("Unexpected source-movie result without its prepared contract.") }
      return nil
    }
    func number(_ value:Any?)throws->Double {
      guard let value=value as? NSNumber,CFGetTypeID(value) != CFBooleanGetTypeID(),value.doubleValue.isFinite else {
        throw StudioError.invalid("Movie acceptance requires finite numeric media timing and geometry.")
      }
      return value.doubleValue
    }
    let frames=try number(prepared?["sourceFrames"]),fps=try number(prepared?["fps"])
    let width=try number(prepared?["width"]),height=try number(prepared?["height"])
    let expected=frames/fps
    guard frames>0,frames.rounded()==frames,(1...60).contains(fps),width>0,height>0,
      result["nativeRuntime"] as? String == "swift-mlx",result["use_complete_duration"] as? Bool == true,
      metadata["task"] as? String == "video_upscale",metadata["pythonModelInference"] as? Bool == false,
      try number(metadata["frames"])==frames,try number(metadata["fps"])==fps,
      try number(metadata["width"])==width,try number(metadata["height"])==height,
      let source=prepared?["sourceMovieSHA256"] as? String,source.count==64,
      source==metadata["source_movie_sha256"] as? String,
      let rgb=prepared?["sourceRGBSHA256"] as? String,rgb.count==64,rgb==metadata["source_rgb_sha256"] as? String,
      abs(try number(result["usable_duration"])-expected)<=1e-7,
      try number(result["usable_source_in"])==0,
      abs(try number(media["fps"])-fps)<0.001,
      try number(media["width"])==width,try number(media["height"])==height,
      abs(try number(media["videoDuration"])-expected)<=0.05/fps,
      try number(media["duration"])+0.05/fps>=expected else {
      throw StudioError.invalid("Source-movie output differs from its frozen interval, canvas, source hashes or frame rate.")
    }
    return expected
  }
}
