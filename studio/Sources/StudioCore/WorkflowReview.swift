import Foundation

public struct WorkflowCharacter: Codable, Equatable, Identifiable {
  public var id: String
  public var description: String
}
public struct WorkflowStoryOutline: Codable {
  public var characters: [WorkflowCharacter]
  public var beats: [String]
}
public struct WorkflowClipDraft: Codable, Equatable, Identifiable {
  public var id: String
  public var startFrame: Int
  public var frameCount: Int
  public var action: String
  public var startState: String
  public var endState: String
  public var location: String
  public var characters: [String]
  public var continuity: String
}
public struct WorkflowClipPlan: Codable {
  public var fps: Int
  public var totalFrames: Int
  public var characters: [WorkflowCharacter]
  public var clips: [WorkflowClipDraft]

  public mutating func replace(_ clip: WorkflowClipDraft) throws {
    guard let index = clips.firstIndex(where: { $0.id == clip.id }),
          clips[index].startFrame == clip.startFrame,
          clips[index].frameCount == clip.frameCount else {
      throw StudioError.invalid("Change movie inputs to alter clip timing.")
    }
    clips[index] = clip
  }
  public func outputs() throws -> [String: JSONValue] {
    ["clips": try JSONDecoder().decode(JSONValue.self, from: JSONEncoder().encode(self))]
  }
}
