import Foundation

struct PreparedH3Tensor { let shape:[Int];let values:[Float16] }
struct PreparedH3Weights { let index:Int;let tensors:[PreparedH3Tensor] }

// This queue owns at most one CPU block; only the inference thread touches Metal.
final class WeightPrefetch {
  private let queue=DispatchQueue(label:"wee-todd.h3.weight-prepare",qos:.userInitiated)
  private let group=DispatchGroup()
  private var result:Result<(PreparedH3Weights,Double),Error>?
  private(set) var pending=false
  private(set) var preparationSeconds:Double=0
  func submit(index:Int,block:H3Block,checkpoint:SafeTensorReader,adapter:SafeTensorReader) throws {
    guard !pending else { throw ProbeError.invalid("Weight prefetch queue already occupied") }
    pending=true;group.enter()
    queue.async {
      let began=Date()
      self.result=Result { (try block.prepareReusable(index:index,checkpoint:checkpoint,adapter:adapter),Date().timeIntervalSince(began)) }
      self.group.leave()
    }
  }
  func take() throws -> PreparedH3Weights {
    guard pending else { throw ProbeError.invalid("No prefetched weights") }
    group.wait();pending=false
    guard let saved=result else { throw ProbeError.invalid("Missing prefetch result") }
    result=nil
    let (weights,seconds)=try saved.get();preparationSeconds += seconds
    return weights
  }
  func drain() { group.wait();result=nil;pending=false }
}
