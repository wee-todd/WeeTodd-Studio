import Combine
import Foundation
import StudioCore

/// Catalog discovery has its own lifetime; opening an editor never competes with rendering.
@MainActor final class DrawThingsCatalogDiscovery: ObservableObject {
  @Published private(set) var catalogs: [String: [String: Any]] = [:]
  @Published private(set) var errors: [String: String] = [:]
  @Published private(set) var loading = Set<String>()
  private var connections: [String: DrawThingsConnection] = [:]
  private var requests: [String: (id: UUID, task: Task<Void, Never>)] = [:]

  func invalidate(_ id: String) {
    requests.removeValue(forKey: id)?.task.cancel()
    connections.removeValue(forKey: id)
    catalogs.removeValue(forKey: id); errors.removeValue(forKey: id); loading.remove(id)
  }

  func load(_ connection: DrawThingsConnection, force: Bool = false,
            fetch: @escaping @MainActor () async throws -> [String: Any]) async {
    let key = connection.id
    if let previous = connections[key], previous != connection { invalidate(key) }
    if let request = requests[key] { await request.task.value; return }
    // A failed automatic attempt is retried explicitly, not in a view-update loop.
    if !force, catalogs[key] != nil || errors[key] != nil { return }
    connections[key] = connection
    let token = UUID()
    loading.insert(key); errors.removeValue(forKey: key)
    let task = Task { @MainActor [weak self] in
      guard let self else { return }
      do {
        let result = try await fetch()
        guard self.requests[key]?.id == token else { return }
        guard result["models"] is [[String: Any]], result["capabilities"] is [String: Any] else {
          throw StudioError.invalid("Draw Things returned an incomplete model catalog. Retry discovery.")
        }
        self.catalogs[key] = result
        self.errors.removeValue(forKey: key)
      } catch {
        guard self.requests[key]?.id == token else { return }
        // Keep the last successful list visible; generation still requires fresh preflight.
        self.errors[key] = error.localizedDescription
      }
      guard self.requests[key]?.id == token else { return }
      self.loading.remove(key); self.requests.removeValue(forKey: key)
    }
    requests[key] = (token, task)
    await task.value
  }
}
