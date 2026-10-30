import Foundation
import InferenceContracts
import MLX

/// Fixed publisher grids retain their Double scalar arithmetic until each
/// operation's Float32 boundary. They are independent of ordinary H3 schedules.
public enum H3RegenerateSchedule: String, Codable, Sendable {
  case publisherThreeStep, publisherTwoStep
  public var sigmas: [Double] {
    switch self {
    case .publisherThreeStep: return NativeCapabilityDescriptor.h3Regenerate.sampling.sigmas
    case .publisherTwoStep:
      let start=0.7, raw=start/(6-5*start), midpoint=raw/2
      return [start,6*midpoint/(1+5*midpoint),0]
    }
  }
  public var modelTimes: [Float] { sigmas.dropLast().map { Float(1-$0) } }
  public var audioSigmas: [Double] { Array(repeating:0,count:sigmas.count) }
  public var evaluations: Int { sigmas.count-1 }
  public var contract: NativeSamplingContract {
    .init(sigmas:sigmas,audioSigmas:audioSigmas,prediction:.dataWardVelocity,
      timeBasis:.oneMinusSigma,updateConvention:.addDataWardVelocity,audioClock:.fixedZero)
  }
  public func initialize(source:MLXArray,noise:MLXArray) throws -> MLXArray {
    guard source.ndim == 3,source.shape == noise.shape,source.shape[0] == 1,
      source.shape[2] == 96,(1...H3RegenerateAdmission.maximumPackedRows).contains(source.shape[1]),
      source.dtype == .float32,noise.dtype == .float32,
      all(isFinite(source)).item(Bool.self),all(isFinite(noise)).item(Bool.self) else {
      throw H3CheckpointError.invalid("Regeneration requires finite normalized video rows and explicit matched Float32 noise.")
    }
    return Float(1-sigmas[0])*source+Float(sigmas[0])*noise
  }
  public func advance(sample:MLXArray,velocity:MLXArray,index:Int) throws -> MLXArray {
    guard (0..<evaluations).contains(index),sample.shape == velocity.shape,
      sample.dtype == .float32,velocity.dtype.isFloatingPoint,
      all(isFinite(velocity)).item(Bool.self) else {
      throw H3CheckpointError.invalid("Invalid regenerate velocity, precision or evaluation index.")
    }
    return sample+Float(sigmas[index]-sigmas[index+1])*velocity.asType(.float32)
  }
}
