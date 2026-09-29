import Foundation
import NNC
// FP16 score storage can overflow before the kernel applies its scalar. Apply
// 1/sqrt(head width) to Q first so finite normalized scores stay representable.
func nativeH3Attention(_ q:Model.IO,_ k:Model.IO,_ v:Model.IO,half:Bool,preScaleQuery:Bool=true) -> Model.IO {
 let factor=1/sqrt(Float(128)),prescale=half && preScaleQuery
 let layer=ScaledDotProductAttention(scale:prescale ? 1 : factor,flags:half && ProcessInfo.processInfo.environment["WEETODD_NNC_ATTENTION_ACCUMULATION"] != "fp32" ? [.Float16] : [])
 return layer(prescale ? factor * q : q,k,v)
}
