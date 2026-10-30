import Foundation

public enum ContractError: Error, Equatable, LocalizedError {
  case invalid(String)
  public var errorDescription: String? { if case .invalid(let message) = self { return message }; return nil }
}
