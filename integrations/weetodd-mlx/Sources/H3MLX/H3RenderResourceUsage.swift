import Darwin

/// Process-level physical footprint plus MLX allocator peak for one H3 worker
/// run. External FFmpeg memory is outside this process measurement.
public struct H3RenderResourceUsage: Sendable {
  public let peakMLXBytes: Int
  public let currentPhysicalBytes: UInt64
  public let peakPhysicalBytes: UInt64

  public static func capture(peakMLXBytes: Int) throws -> Self {
    guard peakMLXBytes >= 0 else {
      throw H3CheckpointError.invalid("H3 MLX peak allocation cannot be negative.")
    }
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(
      MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
    let status = withUnsafeMutablePointer(to: &info) { pointer in
      pointer.withMemoryRebound(to: integer_t.self,
        capacity: Int(count)) {
        task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
      }
    }
    guard status == KERN_SUCCESS,
      info.ledger_phys_footprint_peak >= 0 else {
      throw H3CheckpointError.invalid("Cannot measure H3 worker physical footprint.")
    }
    return Self(peakMLXBytes: peakMLXBytes,
      currentPhysicalBytes: info.phys_footprint,
      peakPhysicalBytes: UInt64(info.ledger_phys_footprint_peak))
  }
}
