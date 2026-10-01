import MLX
import LTX25Engine

/// An argument layout only: no checkpoint arrays or owning block are captured
/// by the compiled closure. Each invocation supplies the current block, prompt,
/// modulation and ordered adapter factors explicitly.
struct MLXBlockGraph {
  struct Weight {
    let start:Int
    let quantized:Bool
    let adapters:[Int]
  }
  let configuration:AVBlockConfiguration
  let inputs:[String:Int]
  let weights:[String:Weight]
  let videoAttentionGroups:[Int]

  static func bind(configuration:AVBlockConfiguration,inputs:[String:MLXArray],
    weights:[String:MLXWeight],adapters:[String:[MLXLoRA]],
    videoAttentionGroups:[Int]=[]) -> (Self,[MLXArray],[Int]) {
    var arrays:[MLXArray]=[],inputSlots:[String:Int]=[:],weightSlots:[String:Weight]=[:]
    var signature=[inputs["video_modulation_indices"] == nil ? 0 : 1,
      inputs["audio_modulation_indices"] == nil ? 0 : 1]
    for name in inputs.keys.sorted() { inputSlots[name]=arrays.count;arrays.append(inputs[name]!) }
    for name in weights.keys.sorted() {
      let packed=weights[name]!.graphArrays,start=arrays.count
      arrays += packed
      var factors:[Int]=[]
      for item in adapters[name] ?? [] where item.scale != 0 {
        factors.append(arrays.count)
        arrays += [item.down,item.up,MLXArray(item.scale)]
      }
      signature += [packed.count,factors.count]
      weightSlots[name]=Weight(start:start,quantized:packed.count == 3,adapters:factors)
    }
    signature += [videoAttentionGroups.count]+videoAttentionGroups
    return (Self(configuration:configuration,inputs:inputSlots,weights:weightSlots,
      videoAttentionGroups:videoAttentionGroups),arrays,signature)
  }

  func call(_ arrays:[MLXArray]) -> [MLXArray] {
    let x=inputs.mapValues { arrays[$0] }
    func parameter(_ name:String) -> MLXArray { arrays[weights[name]!.start].asType(.float32) }
    func linear(_ name:String,_ input:MLXArray) -> MLXArray {
      let w=weights[name+".weight"]!,value=arrays[w.start]
      var result:MLXArray
      if w.quantized {
        result=quantizedMM(input,value,scales:arrays[w.start+1].asType(input.dtype),
          biases:arrays[w.start+2].asType(input.dtype),groupSize:64,bits:8)
      } else { result=matmul(input,value.T) }
      for offset in w.adapters {
        let low=matmul(input,arrays[offset].asType(input.dtype).T)
        result=result+matmul(low,arrays[offset+1].asType(input.dtype).T)*arrays[offset+2]
      }
      if let bias=weights[name+".bias"] { result=result+arrays[bias.start].asType(.float32) }
      return result
    }
    let result=MLXAVBlock.forward(configuration:configuration,x,parameter:parameter,linear:linear,
      videoAttentionGroups:videoAttentionGroups)
    return [result["video"]!,result["audio"]!]
  }
}
