import Foundation

/// Phase progress for actual movie chunks. Fractions are not timing estimates;
/// completion is reserved for the worker's successful atomic publication.
public struct MLXMovieStudioProgress {
  private let chunks:Int
  private var chunk=0,steps=0,fraction=0.0
  public init(chunks:Int) { self.chunks=max(1,chunks) }
  public mutating func event(stage:String,completed:Int,total:Int)->[String:Any] {
    let part=total>0 ? min(1,max(0,Double(completed)/Double(total))) : 0
    var phase=0.0,message=stage.replacingOccurrences(of:"_",with:" ")
    switch stage {
    case "movie_chunk_started":chunk=min(chunks-1,max(0,completed));steps=0
    case "movie_chunk_completed","movie_chunk_reused":
      fraction=max(fraction,0.03+0.89*Double(min(chunks,max(0,completed)))/Double(chunks))
      message="Movie chunk \(completed)/\(chunks) \(stage.hasSuffix("reused") ? "reused":"complete")"
    case "movie_sampling":
      steps=completed;phase=0.35+0.40*part
      message="Refining movie chunk \(chunk+1)/\(chunks) · \(completed)/\(total) updates"
    case "movie_video_layers":phase=0.75+0.20*part;message="Decoding movie chunk \(chunk+1)/\(chunks) · layer \(completed)/\(total)"
    case "movie_video_weights_released":phase=0.97
    case "ready_to_publish":fraction=max(fraction,0.995);message="Publishing enhanced movie"
    default:
      if stage.hasPrefix("movie_refine:") {
        phase=0.35+0.40*min(1,(Double(steps)+part)/3)
        message="Refining movie chunk \(chunk+1)/\(chunks) · update \(min(steps+1,3))/3 · block \(completed)/\(total)"
      } else if stage.hasPrefix("movie_upscale:") || stage == "movie_upscaler_weights_released" { phase=0.35;message="Upscaling movie latents" }
      else if stage.hasPrefix("movie_text:") || stage == "movie_text_weights_released" { phase=0.15+0.10*part;message="Encoding movie prompt · \(completed)/\(total)" }
      else if stage.hasPrefix("movie_audio_context") { phase=0.25+0.05*part;message="Encoding source audio · \(completed)/\(total)" }
      else if stage.hasPrefix("movie_endpoint") { phase=0.15;message="Preparing movie endpoint references" }
      else if stage.hasPrefix("movie_source_video") { phase=0.10*part;message="Encoding source movie chunk \(chunk+1)/\(chunks) · \(completed)/\(total)" }
    }
    fraction=max(fraction,min(0.995,0.03+0.89*(Double(chunk)+phase)/Double(chunks)))
    return ["event":"progress","stage":stage,"message":message,"fraction":fraction,"completed":completed,"total":total]
  }
}
