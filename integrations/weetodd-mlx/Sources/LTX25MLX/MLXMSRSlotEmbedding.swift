import Foundation
import MLX
import TensorIO
import LTX25Engine

/// The five small learned tensors are loaded separately from the streamed
/// rank-128 LoRA factors. Each one-based image slot gets a distinct 128-channel
/// embedding before its VAE latents enter the denoiser.
enum MLXMSRSlotEmbedding {
  static let shapes:[String:[Int]]=[
    "frequencies":[16],"net.0.weight":[256,33],"net.0.bias":[256],
    "net.2.weight":[128,256],"net.2.bias":[128]]

  static func load(_ url:URL) throws -> [String:MLXArray] {
    let file=try SafeTensorFile(url:url,maximumHeaderBytes:4*1024*1024)
    _ = try LTXAdapterCompatibility.msrPlan(file:file,strength:1)
    let prefixes=["diffusion_model.reference_slot_embedding.","reference_slot_embedding."]
    guard let prefix=prefixes.first(where: { file.tensors[$0+"frequencies"] != nil }) else {
      throw LTXError.invalid("MSR learned-slot tensor prefix is missing.")
    }
    var state:[String:MLXArray]=[:]
    for (name,shape) in shapes {
      let value=try MLXWeight.read(file,prefix+name,access:.buffered).asType(.float32)
      guard value.shape == shape else { throw LTXError.invalid("MSR slot tensor shape changed: \(name).") }
      eval(value)
      state[name]=value
    }
    return state
  }

  static func embedding(slotID:Int,state:[String:MLXArray]) throws -> MLXArray {
    guard (1...5).contains(slotID),Set(state.keys) == Set(shapes.keys),
      shapes.allSatisfy({ state[$0.key]?.shape == $0.value }) else {
      throw LTXError.invalid("MSR slot identity or learned tensor shapes are invalid.")
    }
    let scaled=MLXArray([Float(slotID)/16])
    let phases=scaled[0]*state["frequencies"]!
    let features=concatenated([scaled,MLX.sin(phases),MLX.cos(phases)],axis:0)
    var hidden=matmul(features.reshaped([1,33]),state["net.0.weight"]!.T)
      .reshaped([256])+state["net.0.bias"]!
    hidden=hidden*sigmoid(hidden)
    let result=matmul(hidden.reshaped([1,256]),state["net.2.weight"]!.T)
      .reshaped([128])+state["net.2.bias"]!
    eval(result)
    guard MLX.isFinite(result).all().item(Bool.self) else {
      throw LTXError.invalid("MSR learned slot is non-finite.")
    }
    return result
  }
}
