public enum HostError: Error, Equatable {
  case invalidMemoryEstimate
  case insufficientMemory(required: UInt64, available: UInt64)
}
