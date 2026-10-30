import MLX

/// A state belongs to exactly one clip. Reset is required on failure or a new source.
public final class FlashVSRStreamState {
  public private(set) var sourceID: String
  public private(set) var groupIndex = 0
  private var tokenShape: [Int]?
  private var modelIdentity: String?
  var caches: [Int: (MLXArray, MLXArray)] = [:]
  public init(sourceID: String) { self.sourceID = sourceID }
  public func validate(sourceID: String, tokenShape: [Int]) throws {
    guard !sourceID.isEmpty, sourceID == self.sourceID else { throw FlashVSRError.invalid("FlashVSR stream belongs to another source; reset it first.") }
    guard tokenShape.count == 3, tokenShape[0] == 4, tokenShape[1] > 0, tokenShape[2] > 0,
      tokenShape[1] % 8 == 0, tokenShape[2] % 8 == 0 else { throw FlashVSRError.invalid("FlashVSR requires four latent frames and an 8-aligned token canvas.") }
    guard self.tokenShape == nil || self.tokenShape == tokenShape else { throw FlashVSRError.invalid("FlashVSR stream dimensions changed.") }
    self.tokenShape = tokenShape
  }
  func bindModel(_ identity: String) throws {
    guard modelIdentity == nil || modelIdentity == identity else { throw FlashVSRError.invalid("FlashVSR model changed; reset the stream first.") }
    modelIdentity=identity
  }
  public func finishGroup() { groupIndex += 1 }
  public func reset(sourceID: String) { self.sourceID = sourceID; groupIndex = 0; tokenShape = nil; modelIdentity = nil; caches.removeAll() }
  public var cachedLayers: Int { caches.count }
  public var cachedBytes: Int { caches.values.reduce(0) { $0 + $1.0.nbytes + $1.1.nbytes } }
}
