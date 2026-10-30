/// The only supported source grammars are the three pinned FlashVSR archives.
/// An archive hash is part of admission, not a hint or a cache lookup shortcut.
public struct TorchTensorImportProfile: Sendable, Equatable {
  public let id:String
  public let archiveRoot:String
  public let sourceSHA256:String
  public let dtype:String
  let maximumTensors:Int
  let maximumStorages:Int
  let rootTensor:Bool
  let allowsMetadata:Bool

  public static let tinyDecoder = Self(id:"flashvsr-tiny-decoder",archiveRoot:"step-step=54000_lora",
    sourceSHA256:"e224bdcf2f52745cbf4d393ff5374c2ba09e90285d5d19062d2bf63b915b6161",
    dtype:"F32",maximumTensors:66,maximumStorages:1,rootTensor:false,allowsMetadata:false)
  public static let wanVAE = Self(id:"flashvsr-wan21-vae",archiveRoot:"wanx_vae",
    sourceSHA256:"38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981",
    dtype:"F32",maximumTensors:194,maximumStorages:4,rootTensor:false,allowsMetadata:true)
  public static let positivePrompt = Self(id:"flashvsr-positive-prompt",archiveRoot:"posi_prompt",
    sourceSHA256:"4601107a11e4e11a936a6b79df579e54dbc99872132bf542151f0ffd65b4b1ef",
    dtype:"BF16",maximumTensors:1,maximumStorages:1,rootTensor:true,allowsMetadata:false)
}
