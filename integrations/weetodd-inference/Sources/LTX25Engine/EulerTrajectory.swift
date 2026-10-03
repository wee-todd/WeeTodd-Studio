import Foundation
import Metal

public struct SamplingSchedule: Sendable {
  public let sigmas: [Double]
  public let steps: [EulerStep]
  public let eta: Double
  public let noiseStrength: Double
  let predictionSigmas: [Float]
  public init(sigmas: [Double], eta: Double = 1, noiseStrength: Double = 1) throws {
    guard (2...257).contains(sigmas.count), sigmas.last == 0 else {
      throw LTXError.invalid(
        "A complete sampling schedule needs 2–257 descending sigma points ending at zero.")
    }
    steps = try zip(sigmas, sigmas.dropFirst()).map {
      try EulerStep(sigma: $0, nextSigma: $1, eta: eta, noiseStrength: noiseStrength)
    }
    predictionSigmas = sigmas.dropLast().map { value in
      let bits = Float(value).bitPattern
      return Float(bitPattern: (bits &+ 0x7fff &+ ((bits >> 16) & 1)) & 0xffff_0000)
    }
    guard predictionSigmas.allSatisfy({ $0.isFinite && $0 > 0 && $0 <= 1 }) else {
      throw LTXError.invalid("Sampling sigmas must remain positive at the model BF16 boundary.")
    }
    self.sigmas = sigmas
    self.eta = eta
    self.noiseStrength = noiseStrength
  }
}

public struct AVLatents: Sendable {
  public let video: [Float]
  public let audio: [Float]
  public init(video: [Float], audio: [Float]) {
    self.video = video
    self.audio = audio
  }
}

/// Float32 trajectories with explicit noise: RNG policy is separate from model arithmetic.
/// Buffers are reused across steps and released on return. The prediction callback is
/// synchronous; the owning pipeline controls model residency across predictions.
/// Optional previews receive completed latent states, not decoded pixels. They do
/// not imply that a reused transformer session has released its weighted slot;
/// callers must explicitly admit any weighted preview decoder overlap.
public final class EulerTrajectory {
  public enum Modality: String { case video, audio }
  public struct Progress {
    public let completedSteps: Int
    public let totalSteps: Int
    public let sigma: Double
    public let nextSigma: Double
  }
  public typealias Noise = (Int, Modality, Int) throws -> [Float]
  private let device: MTLDevice
  private let queue: MTLCommandQueue
  private let sampler: MetalEulerSampler
  private var active = false
  public init() throws {
    guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
      throw LTXError.invalid("Metal is unavailable for LTX sampling.")
    }
    self.device = device
    self.queue = queue
    sampler = try MetalEulerSampler(device: device)
  }

  public func evaluate(
    video: [Float], audio: [Float], schedule: SamplingSchedule,
    noise: Noise? = nil, predict: (AVLatents, Float) throws -> AVLatents,
    progress: (Progress) throws -> Void = { _ in },
    preview: ((AVLatents, Progress) throws -> Void)? = nil
  ) throws -> AVLatents {
    guard !active else { throw LTXError.invalid("The sampler is already evaluating.") }
    active = true
    defer { active = false }
    let counts = [video.count, audio.count]
    guard counts.allSatisfy({ $0 > 0 && $0 <= 16 * 1024 * 1024 }),
      video.allSatisfy(\.isFinite), audio.allSatisfy(\.isFinite),
      !schedule.steps.contains(where: \.ancestral) || noise != nil
    else {
      throw LTXError.invalid(
        "Invalid latent tensors, allocation bound or missing ancestral noise provider.")
    }
    try Task.checkCancellation()
    // Four buffers per stream: current sample, velocity, noise, next sample.
    let buffers: [[MTLBuffer]] = try counts.map { count in
      try (0..<4).map { _ in
        guard let buffer = device.makeBuffer(length: count * 4, options: .storageModeShared) else {
          throw LTXError.invalid("Cannot allocate bounded sampling buffers.")
        }
        return buffer
      }
    }
    func upload(_ values: [Float], _ buffer: MTLBuffer) {
      values.withUnsafeBytes {
        buffer.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count)
      }
    }
    func download(_ buffer: MTLBuffer, _ count: Int) -> [Float] {
      Array(
        UnsafeBufferPointer(
          start: buffer.contents().assumingMemoryBound(to: Float.self), count: count))
    }
    var state = AVLatents(video: video, audio: audio)
    for index in schedule.steps.indices {
      try Task.checkCancellation()
      let step = schedule.steps[index]
      if step.ancestral {
        for (stream, modality) in [Modality.video, .audio].enumerated() {
          let values = try noise!(index, modality, counts[stream])
          guard values.count == counts[stream], values.allSatisfy(\.isFinite) else {
            throw LTXError.invalid("Invalid ancestral noise for \(modality.rawValue).")
          }
          upload(values, buffers[stream][2])
          try Task.checkCancellation()
        }
      }
      // Reference wrapper uses BF16 sigma for velocity -> x0 conversion.
      let predictionSigma = schedule.predictionSigmas[index]
      let velocity = try autoreleasepool { try predict(state, predictionSigma) }
      try Task.checkCancellation()
      guard velocity.video.count == video.count, velocity.audio.count == audio.count,
        velocity.video.allSatisfy(\.isFinite), velocity.audio.allSatisfy(\.isFinite)
      else {
        throw LTXError.invalid("Invalid denoiser velocity outputs.")
      }
      guard let commands = queue.makeCommandBuffer() else {
        throw LTXError.invalid("Cannot create sampling commands.")
      }
      for (stream, pair) in [(state.video, velocity.video), (state.audio, velocity.audio)]
        .enumerated()
      {
        let b = buffers[stream]
        upload(pair.0, b[0])
        upload(pair.1, b[1])
        try sampler.encode(
          commandBuffer: commands, sample: b[0], denoised: b[1],
          noise: step.ancestral ? b[2] : nil, output: b[3], elementCount: counts[stream],
          channels: 1,
          step: step, velocitySigma: predictionSigma)
      }
      commands.commit()
      commands.waitUntilCompleted()
      guard commands.status == .completed else {
        throw LTXError.invalid("Metal sampling failed: \(String(describing: commands.error))")
      }
      try Task.checkCancellation()
      state = AVLatents(
        video: download(buffers[0][3], video.count), audio: download(buffers[1][3], audio.count))
      guard state.video.allSatisfy(\.isFinite), state.audio.allSatisfy(\.isFinite) else {
        throw LTXError.invalid("Sampling produced nonfinite latents.")
      }
      let event = Progress(
        completedSteps: index + 1, totalSteps: schedule.steps.count,
        sigma: schedule.sigmas[index], nextSigma: schedule.sigmas[index + 1])
      try progress(event)
      try preview?(state, event)
      try Task.checkCancellation()
    }
    return state
  }
}
