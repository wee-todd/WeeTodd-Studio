import Foundation
import CryptoKit
import TensorIO

/// Admission for the publisher's unconverted 297-tensor fast video decoder.
/// The weights have no embedded metadata: normalization is explicitly bound to
/// an original H3 configuration or an admitted native H3 video-VAE header.
public final class H3FastDecoderLayout {
  public static let sourceSHA256="8658f1e85b5997c4ba716406a2f148bfa36cf97d4d5fcd402c7bd341f54e77a1"
  public static let selectedOriginalLayers=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,19,20,21,31,32,35]
  public struct Configuration: Sendable {
    public let mean:[Float], standardDeviation:[Float]
    public let width=2048, heads=32, headDimension=64, window=16, temporalBlock=5
    public let epsilon:Float=1e-5, rotaryTheta:Float=100, rotaryRatio:Float=0.75
    public let normalizationIdentity:String
    init(mean:[Float],standardDeviation:[Float]) throws {
      guard mean.count == 24,standardDeviation.count == 24,mean.allSatisfy(\.isFinite),standardDeviation.allSatisfy({$0.isFinite && $0 > 0}) else { throw H3CheckpointError.invalid("Invalid fast H3 latent normalization.") }
      self.mean=mean;self.standardDeviation=standardDeviation
      var bytes=Data()
      for value in mean+standardDeviation { var bits=value.bitPattern.littleEndian; withUnsafeBytes(of:&bits) { bytes.append(contentsOf:$0) } }
      normalizationIdentity=SHA256.hash(data:bytes).map { String(format:"%02x",$0) }.joined()
    }
    public init(json:[String:Any]) throws {
      guard json["_class_name"] as? String == "AutoencoderKLMiniMaxH3",
        json["latent_channels"] as? Int == 24,json["decoder_num_layers"] as? Int == 36,
        json["decoder_num_attention_heads"] as? Int == 32,json["decoder_attention_head_dim"] as? Int == 64,
        json["decoder_num_register_tokens"] as? Int == 4,json["decoder_ffn_mult"] as? Int == 4,
        (json["decoder_rope_theta"] as? NSNumber)?.doubleValue == 100,
        (json["decoder_rope_dim_ratio"] as? NSNumber)?.doubleValue == 0.75,
        (json["decoder_norm_eps"] as? NSNumber)?.doubleValue == 1e-5,
        json["clip_length"] as? Int == 17,json["token_drop"] as? Int == 3,
        let mean=json["latents_mean"] as? [NSNumber],let std=json["latents_std"] as? [NSNumber] else {
        throw H3CheckpointError.invalid("Fast H3 requires the original 24-channel / 32-head H3 VAE configuration.")
      }
      try self.init(mean:mean.map(\.floatValue),standardDeviation:std.map(\.floatValue))
    }
  }
  public let weightsURL:URL, configuration:Configuration
  let file:SafeTensorFile
  public convenience init(weightsURL:URL,baseVAEURL:URL) throws {
    let base=try H3VideoVAELayout(url:baseVAEURL);try base.validateVideoUse()
    try self.init(weightsURL:weightsURL,configuration:.init(mean:base.latentsMean,standardDeviation:base.latentsStandardDeviation))
  }
  public convenience init(weightsURL:URL,configurationURL:URL) throws {
    let handle=try FileHandle(forReadingFrom:configurationURL);defer { try? handle.close() }
    let bytes=try handle.read(upToCount:1_048_577) ?? Data()
    guard bytes.count <= 1_048_576,let json=try JSONSerialization.jsonObject(with:bytes) as? [String:Any] else { throw H3CheckpointError.invalid("Invalid original H3 VAE configuration JSON.") }
    try self.init(weightsURL:weightsURL,configuration:.init(json:json))
  }
  public init(weightsURL:URL,configuration:Configuration) throws {
    let source=try SafeTensorFile(url:weightsURL)
    try Self.validateSchema(shapes:source.tensors.mapValues{$0.shape.map(Int.init)},dtypes:source.tensors.mapValues(\.dtype))
    guard try H3OmniCheckpoint.checkedDigest(file:source,url:weightsURL) == Self.sourceSHA256 else { throw H3CheckpointError.invalid("Fast H3 decoder content differs from the selected publisher checkpoint.") }
    self.weightsURL=weightsURL;self.configuration=configuration;file=source
  }
  public func checkUnchanged() throws { try file.checkUnchanged(at:weightsURL) }
  static func validateSchema(shapes:[String:[Int]],dtypes:[String:String]) throws {
    guard shapes == expectedShapes,Set(dtypes.keys) == Set(expectedShapes.keys),dtypes.values.allSatisfy({$0 == "F32"}) else { throw H3CheckpointError.invalid("Fast H3 requires all 297 original F32 tensors in the released layout.") }
  }
  static let expectedShapes:[String:[Int]] = {
    var result:[String:[Int]]=["post_quant_conv.weight":[24,24,1,1,1],"post_quant_conv.bias":[24],"x_embedder.weight":[2048,24],"x_embedder.bias":[2048],"register_tokens":[1,4,2048],"norm_out.weight":[2048],"norm_out.bias":[2048],"proj_out.weight":[3072,2048],"proj_out.bias":[3072]]
    for layer in 0..<24 {
      let p="blocks.\(layer)."
      for key in ["norm1.weight","norm2.weight","scale1","scale2"] { result[p+key]=[2048] }
      for (key,rows,cols) in [("attn.to_qkv",6144,2048),("attn.to_out",2048,2048),("ff.w1",16384,2048),("ff.w2",2048,8192)] {
        result[p+key+".weight"]=[rows,cols];result[p+key+".bias"]=[rows]
      }
    }
    return result
  }()
}
