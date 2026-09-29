import Foundation

public struct TextEncodingConfiguration: Sendable {
  /// Admission for owned arrays, weight slabs and Metal buffers. Driver/compiler
  /// caches, mapped checkpoint residency and allocator overhead are additional;
  /// this is not a hard limit on process footprint or system memory pressure.
  public var maximumOwnedBufferBytes = 3 * 1024 * 1024 * 1024
  public init() {}
}

/// Conservative simultaneous-ownership bounds for the validated 48-layer pack.
/// Arithmetic follows shape validation, so hostile dimensions cannot overflow.
public struct TextEncodingPlan: Sendable {
  /// Reservation for the already constructed tokenizer and checkpoint metadata.
  /// Foundation container/allocator overhead is estimated, not introspectable.
  public let metadataReserveBytes = 512*1024*1024
  public let interleavedBytes: Int
  public let gemmaBytes: Int
  public let aggregationBytes: Int
  public let connectorBytes: Int
  public let ownedBufferBytes: Int

  public init(promptTokens n: Int, configuration: TextEncodingConfiguration = .init()) throws {
    guard (1...1024).contains(n), configuration.maximumOwnedBufferBytes > 0 else {
      throw TextEncodingError.invalid("Invalid text token count or memory budget.")
    }
    interleavedBytes = n*3840*49*4
    // Source mapping/read, decoded slab, conversion scratch, uploaded weight.
    // This deliberately overcounts Q8 source storage at its expanded size.
    let slabs = 4*64*1024*1024
    let hidden = n*3840*4, mlp = n*15360*4
    let query = n*16*512*4, kv = n*8*256*4
    let attention = 4*query + 4*kv + n*n*4
    // Lexically live CPU activations and rotary/norm copies, plus GPU attention
    // and the largest prepared-left/result pair. Sum disjoint phases for safety.
    gemmaBytes = interleavedBytes + 12*hidden + 5*mlp + 8*query + 8*kv
      + attention + 3*mlp + slabs
    // Both interleaved CPU states and their prepared Metal copy coexist. Retain
    // both modality outputs, the active projection result and CPU/GPU row outputs.
    let projection = n*4096*4
    aggregationBytes = 2*interleavedBytes + n*6144*4 + 3*projection + hidden + slabs
    // Connectors always run 1,024 tokens, even for a one-token prompt. Include
    // both projections, the completed video output, registers, attention buffers,
    // and the 4x MLP expansion plus prepared input and output copies.
    let context = 1024*4096*4
    connectorBytes = 1024*6144*4 + 128*4096*4 + 24*context + 6*(4*context)
      + 8*context + 1024*1024*4 + slabs
    ownedBufferBytes = metadataReserveBytes + max(gemmaBytes, aggregationBytes, connectorBytes)
    guard ownedBufferBytes <= configuration.maximumOwnedBufferBytes else {
      throw TextEncodingError.invalid("Text encoding requires an owned-buffer budget of at least \(ownedBufferBytes) bytes.")
    }
  }
}
