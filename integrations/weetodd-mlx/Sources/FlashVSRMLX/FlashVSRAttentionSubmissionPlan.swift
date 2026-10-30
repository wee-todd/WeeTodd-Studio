import Foundation

/// Keeps original rank2 matmuls; bounds independent windows sharing one eval.
struct FlashVSRAttentionSubmissionPlan {
  let groups:[[Int]]
  // F32 Q/state/output, K/V plus gather copies, scores/probabilities, online
  // state and indices. Inputs and accumulated final output are separate.
  static let maximumScratchBytes=8*(3*128*128*4+4*8*128*128*4+2*128*8*128*4+8*128*4+9*4)

  init(selectedCounts:[Int],maximumWindows:Int=8)throws {
    guard !selectedCounts.isEmpty,selectedCounts.allSatisfy({$0>0}),(1...8).contains(maximumWindows) else {
      throw FlashVSRError.invalid("Invalid FlashVSR attention submission plan.")
    }
    if maximumWindows == 1 {
      groups=selectedCounts.indices.map{[$0]}
      return
    }
    var buckets=[Int:[Int]]()
    for (index,count) in selectedCounts.enumerated() { buckets[count,default:[]].append(index) }
    groups=buckets.keys.sorted().flatMap { count in
      let windows=buckets[count]!
      return stride(from:0,to:windows.count,by:maximumWindows).map {
        Array(windows[$0..<min($0+maximumWindows,windows.count)])
      }
    }
  }
}
