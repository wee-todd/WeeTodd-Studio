import Foundation
import CoreFoundation
import TensorIO
import LTX25Engine
import LTX25Video

/// Component/header admission is independent of backend choice. A selected
/// DiffVAE never falls back to Conv or another runtime after an error.
public enum MLXVideoDecoderSelection {
  case convolutional
  case diffusion(MLXDiffusionVideoCheckpoint)
  public init(checkpoint:URL,settings:MLXDiffusionVideoSettings?=nil) throws {
    let file=try SafeTensorFile(url:checkpoint,maximumHeaderBytes:1024*1024)
    guard let raw=file.metadata["config"],let root=try JSONSerialization.jsonObject(with:Data(raw.utf8)) as? [String:Any],
      let vae=root["vae"] as? [String:Any] else { throw LTXError.invalid("Missing native LTX video VAE architecture metadata.") }
    if vae["_class_name"] as? String == "CausalDiffusionVAE" {
      self = .diffusion(try MLXDiffusionVideoCheckpoint(checkpoint:checkpoint))
    } else {
      guard settings?.isDefault ?? true else { throw LTXError.invalid("Diffusion VAE optimization settings cannot be applied to a convolutional VAE.") }
      _=try VideoDecoder(checkpoint:checkpoint)
      self = .convolutional
    }
  }
  public var isDiffusion:Bool { if case .diffusion=self { return true };return false }
}

/// The distributed DiffVAE embeds the SAME84 convolutional encoder parameter
/// shapes. Mean-channel encoding is unchanged; only its architecture metadata
/// nests settings and declares a constant discarded log-variance channel.
enum MLXVideoEncoderConfiguration {
  static func compatible(_ metadata:[String:String]) throws -> Bool {
    guard metadata["model_version"] == "2.5.0",let raw=metadata["config"],
      let config=try JSONSerialization.jsonObject(with:Data(raw.utf8)) as? [String:Any],let vae=config["vae"] as? [String:Any] else { return false }
    if vae["_class_name"] as? String == "CausalVideoAutoencoder" {
      return vae["spatial_padding_mode"] as? String == "zeros" && vae["norm_layer"] as? String == "pixel_norm" &&
        vae["patch_size"] as? Int == 4 && vae["encoder_base_channels"] as? Int == 128 && vae["latent_channels"] as? Int == 128 &&
        vae["latent_log_var"] as? String == "uniform" && vae["use_quant_conv"] as? Bool == false
    }
    func integral(_ value:Any?) -> Int? {
      guard let n=value as? NSNumber,CFGetTypeID(n) != CFBooleanGetTypeID(),n.doubleValue.isFinite,
        n.doubleValue>=0,n.doubleValue<4096,n.doubleValue.rounded(.towardZero)==n.doubleValue else { return nil }
      return n.intValue
    }
    guard vae["_class_name"] as? String == "CausalDiffusionVAE",let encoder=vae["encoder"] as? [String:Any],
      encoder["_class_name"] as? String == "Encoder",integral(encoder["dims"]) == 3,
      integral(encoder["in_channels"]) == 3,integral(encoder["out_channels"]) == 128,
      integral(encoder["base_channels"]) == 128,integral(encoder["patch_size"]) == 4,
      encoder["norm_layer"] as? String == "pixel_norm",encoder["spatial_padding_mode"] as? String == "zeros",
      encoder["latent_log_var"] as? String == "constant",let variance=encoder["latent_log_var_value"] as? Double,
      variance.isFinite,abs(variance-(-7.824046010856292))<0.000000000001,
      let blocks=encoder["blocks"] as? [[Any]],blocks.count == 9 else { return false }
    let kinds=["res_x","compress_space_res","res_x","compress_time_res","res_x","compress_all_res","res_x","compress_all_res","res_x"]
    let values=[4,2,6,2,4,2,2,1,2]
    return blocks.enumerated().allSatisfy { index,row in
      row.count == 2 && row[0] as? String == kinds[index] &&
        (row[1] as? [String:Any]).map { parameters in
          let key=index%2 == 0 ? "num_layers":"multiplier"
          return Set(parameters.keys)==[key] && integral(parameters[key]) == values[index]
        } == true
    }
  }
}
