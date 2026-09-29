import Foundation

/// Shared, backend-independent uniform-timestep 2.5 denoiser contract.
public enum DenoiserLayout {
  public struct Head: Sendable {
    public let name: String
    public let input: String
    public let dimension: Int
    public let parameters: Int
  }
  public static func heads(_ c: AVBlockConfiguration) -> [Head] {
    [("adaln_single", "video_modulation", c.videoDimension, 9),
     ("audio_adaln_single", "audio_modulation", c.audioDimension, 9),
     ("prompt_adaln_single", "video_prompt_modulation", c.videoDimension, 2),
     ("audio_prompt_adaln_single", "audio_prompt_modulation", c.audioDimension, 2),
     ("av_ca_video_scale_shift_adaln_single", "video_av_modulation", c.videoDimension, 4),
     ("av_ca_audio_scale_shift_adaln_single", "audio_av_modulation", c.audioDimension, 4),
     ("av_ca_a2v_gate_adaln_single", "video_av_gate", c.videoDimension, 1),
     ("av_ca_v2a_gate_adaln_single", "audio_av_gate", c.audioDimension, 1)
    ].map { Head(name:$0.0,input:$0.1,dimension:$0.2,parameters:$0.3) }
  }
  public static func weightShapes(_ c: AVBlockConfiguration) -> [String:[Int]] {
    var shapes:[String:[Int]]=[:]
    func linear(_ name:String,_ input:Int,_ output:Int) {
      shapes[name+".weight"]=[output,input]; shapes[name+".bias"]=[output]
    }
    for head in heads(c) {
      linear(head.name+".emb.timestep_embedder.linear1",256,head.dimension)
      linear(head.name+".emb.timestep_embedder.linear2",head.dimension,head.dimension)
      linear(head.name+".linear",head.dimension,head.dimension*head.parameters)
    }
    for (prefix,dim) in [("",c.videoDimension),("audio_",c.audioDimension)] {
      linear(prefix+"patchify_proj",128,dim); linear(prefix+"proj_out",dim,128)
      shapes[prefix+"scale_shift_table"]=[2,dim]
    }
    return shapes
  }
  public static func inputShapes(_ c: AVBlockConfiguration) -> [String:[Int]] {
    ["video_latent":[c.videoTokens,128],"audio_latent":[c.audioTokens,128],
     "video_text":[c.textTokens,c.videoDimension],"audio_text":[c.textTokens,c.audioDimension],
     "video_positions":[c.videoTokens,3],"audio_positions":[c.audioTokens,1]]
  }
}
