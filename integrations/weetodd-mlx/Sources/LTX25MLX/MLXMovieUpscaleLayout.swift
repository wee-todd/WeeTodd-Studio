import Foundation
import MLX
import LTX25Engine

/// One full-resolution movie canvas, optional complete half-resolution Pixel
/// guide, then the true visible last image. There are no generated slots.
public struct MLXMovieUpscaleLayout: Sendable {
  public let geometry: AVGeometry, sourceGeometry: AVGeometry
  public let visibleFrames: Int, pixelReference: Bool
  public let firstStrength: Float?, lastStrength: Float?
  public let positions: [Float], groupRows: [Int], mask: [Float], attentionTemplates: [Float]
  public var frameTokens: Int { geometry.latentHeight * geometry.latentWidth }
  public var referenceTokens: Int { pixelReference ? sourceGeometry.videoTokens : 0 }
  public var endpointTokens: Int { lastStrength == nil ? 0 : frameTokens }
  public var videoTokens: Int { geometry.videoTokens + referenceTokens + endpointTokens }
  public init(plan: MLXMovieUpscalePlan, firstStrength: Float?, lastStrength: Float?) throws {
    guard plan.mode != .latentOnly, plan.paddedFrames <= 4097,
      [firstStrength,lastStrength].allSatisfy({ $0.map { $0.isFinite && (0...1).contains($0) } ?? true }),
      lastStrength == nil || plan.frames > 1 else {
      throw LTXError.invalid("Movie refinement requires legal endpoint strengths and a bounded causal canvas.")
    }
    geometry = try AVGeometry(width:plan.size.outputWidth,height:plan.size.outputHeight,
      frames:plan.paddedFrames,fps:plan.fps)
    sourceGeometry = try AVGeometry(width:plan.size.width,height:plan.size.height,
      frames:plan.paddedFrames,fps:plan.fps)
    visibleFrames=plan.frames; pixelReference=plan.mode == .pixelSpatial
    self.firstStrength=firstStrength; self.lastStrength=lastStrength
    let mainRows=geometry.videoTokens, guideRows=pixelReference ? sourceGeometry.videoTokens : 0
    let imageRows=lastStrength == nil ? 0 : geometry.latentHeight*geometry.latentWidth
    let count=mainRows+guideRows+imageRows
    guard count <= 131072 else { throw LTXError.invalid("Movie canvas and references exceed native token admission.") }
    var values=geometry.videoPositions
    if pixelReference {
      let guide=sourceGeometry.videoPositions
      for row in 0..<guideRows { values += [guide[row*3],guide[row*3+1]*2,guide[row*3+2]*2] }
    }
    if lastStrength != nil {
      let time=Float((Double(plan.frames-1)+0.5)/plan.fps)
      for h in 0..<geometry.latentHeight { for w in 0..<geometry.latentWidth {
        values += [time,Float(h*32+16),Float(w*32+16)]
      } }
    }
    positions=values
    groupRows=[mainRows]+(guideRows>0 ? [guideRows] : [])+(imageRows>0 ? [imageRows] : [])
    var factors=[Float](repeating:1,count:mainRows)
    if let firstStrength { for row in 0..<geometry.latentHeight*geometry.latentWidth { factors[row]=1-firstStrength } }
    factors += [Float](repeating:0,count:guideRows)
    factors += [Float](repeating:1-(lastStrength ?? 0),count:imageRows)
    mask=factors
    // Each reference sees itself and the main canvas, never another reference.
    var templates=[Float](repeating:1,count:count)
    if groupRows.count>1 {
      for index in 1..<groupRows.count {
        var row=[Float](repeating:0,count:count)
        for token in 0..<mainRows { row[token]=1 }
        let start=groupRows.prefix(index).reduce(0,+)
        for token in start..<start+groupRows[index] { row[token]=1 }
        templates += row
      }
    }
    attentionTemplates=groupRows.count>1 ? templates : []
  }
  public func prepare(generated:MLXArray,source:MLXArray?,first:MLXArray?,last:MLXArray?) throws
    -> (latent:MLXArray,condition:MLXVideoDenoiseCondition) {
    try Task.checkCancellation()
    func valid(_ value:MLXArray?,rows:Int) -> Bool {
      value.map { $0.dtype == .float32 && $0.shape == [rows,128] && MLX.isFinite($0).all().item(Bool.self) } ?? false
    }
    guard valid(generated,rows:geometry.videoTokens),
      pixelReference ? valid(source,rows:sourceGeometry.videoTokens) : source == nil,
      firstStrength != nil ? valid(first,rows:frameTokens) : first == nil,
      lastStrength != nil ? valid(last,rows:frameTokens) : last == nil else {
      throw LTXError.invalid("Movie reference latents differ from frozen canvas, guide or endpoint admission.")
    }
    var parts:[MLXArray]=[],clean:[MLXArray]=[]
    if let first {
      parts.append(first.reshaped([frameTokens,128])); clean.append(first.reshaped([frameTokens,128]))
      if geometry.videoTokens>frameTokens {
        parts.append(generated[frameTokens..<geometry.videoTokens]);clean.append(.zeros([geometry.videoTokens-frameTokens,128]))
      }
    } else { parts.append(generated.reshaped(generated.shape));clean.append(.zeros(generated.shape)) }
    if let source { parts.append(source.reshaped(source.shape));clean.append(source.reshaped(source.shape)) }
    if let last { parts.append(last.reshaped(last.shape));clean.append(last.reshaped(last.shape)) }
    let latent=concatenated(parts,axis:0),reference=concatenated(clean,axis:0)
    eval(latent,reference);try Task.checkCancellation()
    return (latent,try MLXVideoDenoiseCondition(clean:reference,mask:mask))
  }
}
