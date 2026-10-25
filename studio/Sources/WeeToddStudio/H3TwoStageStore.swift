import Foundation
import StudioCore

@MainActor extension StudioStore {
  func loadH3TwoStageModels() {
    let file=dataDirectory.appendingPathComponent("h3-two-stage-models.json")
    guard FileManager.default.fileExists(atPath:file.path) else {return}
    do {h3TwoStageModels=try JSONDecoder().decode(H3TwoStageModelLinks.self,from:Data(contentsOf:file))}
    catch {self.error="Could not read H3 two-stage model links: \(error.localizedDescription)"}
  }
  func saveH3TwoStageModels(_ links:H3TwoStageModelLinks) {
    do {
      try JSONEncoder().encode(links).write(to:dataDirectory.appendingPathComponent("h3-two-stage-models.json"),options:.atomic)
      h3TwoStageModels=links
    } catch {self.error=error.localizedDescription}
  }
}
