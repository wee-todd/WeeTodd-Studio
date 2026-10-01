import Foundation
import MLX
import LTX25Engine

/// Independently encoded MSR image groups. The target remains first, then
/// learned slots one through five occupy prepended negative reference times.
struct MLXMSRLayout {
  struct Group:Sendable {
    let geometry:AVGeometry
    let strength:Float
    let attentionStrength:Float
    init(geometry:AVGeometry,strength:Float,attentionStrength:Float) {
      self.geometry=geometry;self.strength=strength;self.attentionStrength=attentionStrength
    }
  }
  let target:AVGeometry
  let groups:[Group]
  let groupRows:[Int]
  let videoTokens:Int
  let positions:[Float]

  init(target:AVGeometry,groups:[Group]) throws {
    guard (1...5).contains(groups.count),groups.allSatisfy({ group in
      group.geometry.fps == target.fps && [25,33].contains(group.geometry.frames) &&
      group.strength.isFinite && (0...1).contains(group.strength) &&
      group.attentionStrength.isFinite && (0...1).contains(group.attentionStrength)
    }) else { throw LTXError.invalid("MSR needs one to five finite 25/33-frame image groups.") }
    let rows=[target.videoTokens]+groups.map(\.geometry.videoTokens)
    let count=rows.reduce(0,+)
    guard count <= 131072 else { throw LTXError.invalid("MSR references exceed the admitted video token budget.") }
    var values=target.videoPositions
    values.reserveCapacity(count*3)
    for (index,group) in groups.enumerated() {
      let offset=Float(index-groups.count)/Float(target.fps)
      let source=group.geometry.videoPositions
      for token in 0..<group.geometry.videoTokens {
        values += [source[token*3]+offset,source[token*3+1],source[token*3+2]]
      }
    }
    self.target=target;self.groups=groups;groupRows=rows;videoTokens=count;positions=values
  }

  func prepare(generated:MLXArray,references:[MLXArray]) throws
    -> (latent:MLXArray,condition:MLXVideoDenoiseCondition,attentionTemplates:MLXArray) {
    guard generated.dtype == .float32,generated.shape == [target.videoTokens,128],
      references.count == groups.count,
      zip(references,groups).allSatisfy({ item,group in
        item.dtype == .float32 && item.shape == [group.geometry.videoTokens,128]
      }),MLX.isFinite(generated).all().item(Bool.self),
      references.allSatisfy({ MLX.isFinite($0).all().item(Bool.self) }) else {
      throw LTXError.invalid("MSR reference latents differ from their admitted image groups.")
    }
    let latent=concatenated([generated]+references,axis:0)
    let clean=concatenated([MLXArray.zeros([target.videoTokens,128])]+references,axis:0)
    let mask=[Float](repeating:1,count:target.videoTokens)+zip(groups,groupRows.dropFirst())
      .flatMap { group,rows in [Float](repeating:1-group.strength,count:rows) }
    let targetTemplate=[Float](repeating:1,count:target.videoTokens)+zip(groups,groupRows.dropFirst())
      .flatMap { group,rows in [Float](repeating:group.attentionStrength,count:rows) }
    var templates=targetTemplate
    for group in groups {
      templates += [Float](repeating:group.attentionStrength,count:target.videoTokens)
      templates += [Float](repeating:1,count:videoTokens-target.videoTokens)
    }
    eval(latent,clean)
    return (latent,try MLXVideoDenoiseCondition(clean:clean,mask:mask),
      MLXArray(templates,[groups.count+1,videoTokens]))
  }
}
