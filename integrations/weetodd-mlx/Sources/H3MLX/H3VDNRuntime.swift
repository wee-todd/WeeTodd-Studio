import Foundation
import MLX

public struct H3VDNSelection:Sendable {
  public enum Variant:String,Sendable { case eightStep="8_step",fiftyStep="50_step" }
  public let stage:URL
  public let variant:Variant
  public let adalnInputGrid:URL?
  public var schedulePoints:Int { variant == .eightStep ? 9 : 51 }
  public init(stage:URL,variant:Variant,adalnInputGrid:URL? = nil) throws {
    guard stage.isFileURL,stage.path.hasPrefix("/"),
      adalnInputGrid == nil || (variant == .eightStep && adalnInputGrid!.isFileURL && adalnInputGrid!.path.hasPrefix("/")) else {
      throw H3CheckpointError.invalid("VDN requires an absolute installed stage directory.")
    }
    self.stage=stage;self.variant=variant;self.adalnInputGrid=adalnInputGrid
  }
  func preflight() throws {
    _ = try H3VDNCheckpoint(stage:stage)
    _ = try H3VDNLoRAStack(selection:self)
    if let adalnInputGrid { _ = try H3VDNInputGrid(url:adalnInputGrid).evaluate(timesteps:[0,1]) }
  }
}

final class H3VDNLoRAStack:H3LoRAApplying {
  private let files:[H3VDNLoRAFile]
  init(selection:H3VDNSelection) throws {
    var files=[try H3VDNLoRAFile(directory:selection.stage.appendingPathComponent("adapters/default"),kind:.standard)]
    if selection.variant == .eightStep {
      files.append(try H3VDNLoRAFile(directory:selection.stage.appendingPathComponent("adapters/turbo"),kind:.turbo))
    }
    self.files=files
  }
  func apply(base:MLXArray,input:MLXArray,target:String,reorderQKV:Bool=false) throws -> MLXArray {
    var output=base
    for file in files { output=try file.apply(base:output,input:input,target:target,reorderQKV:reorderQKV) }
    return output
  }
}

/// Only headers and file identities live across steps; branch tensors stream
/// through the same H3 block scope as backbone projections.
final class H3VDNRuntime {
  let checkpoint:H3VDNCheckpoint
  let layout:H3VDNLayout
  let lora:H3VDNLoRAStack
  init(selection:H3VDNSelection,packed:H3PackedLayout) throws {
    layout=try H3VDNLayout(packed:packed)
    checkpoint=try H3VDNCheckpoint(stage:selection.stage)
    lora=try H3VDNLoRAStack(selection:selection)
  }
  func attention(block:Int,input:MLXArray,qkv:MLXArray,query:MLXArray,key:MLXArray,
    value:MLXArray,project:(MLXArray) throws -> MLXArray) throws -> MLXArray {
    try H3VDNAttention.evaluate(input:input,
      raw:[qkv[.ellipsis,0,0..<128],qkv[.ellipsis,1,0..<128],qkv[.ellipsis,2,0..<128]],
      query:query,key:key,value:value,layout:layout,
      weights:try checkpoint.readBlock(block),projectSoftmax:project)
  }
}
