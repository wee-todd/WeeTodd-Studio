import Foundation

/// Optional display updates cannot fail or alter a final-generation contract.
public struct NativeTAEPreviewPolicy {
  public private(set) var emittedCount = 0
  public private(set) var disabled = false
  private var lastStep: Int?
  private var lastUptime: Double?
  public init() {}
  public mutating func attempt<Frame>(completed: Int, total: Int, uptime: Double,
    render: () throws -> Frame, publish: (Frame) throws -> Void,
    onFailure: (String) throws -> Void) throws {
    guard !disabled, total > 0, (1...total).contains(completed), uptime.isFinite, uptime >= 0,
      lastStep == nil || completed >= lastStep! + 2,
      lastUptime == nil || uptime - lastUptime! >= 2 else { return }
    do {
      let frame = try render()
      try Task.checkCancellation()
      try publish(frame)
      lastStep = completed; lastUptime = uptime; emittedCount += 1
    } catch is CancellationError { throw CancellationError() }
    catch {
      disabled = true
      try onFailure("Live TAE previews stopped: \(error). Final decoding will use the selected full VAE.")
    }
  }
}
