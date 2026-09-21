import Foundation

public extension CharacterPanelRole {
  var label: String {
    switch self { case .front: return "Front"; case .side: return "Side"; case .back: return "Back"; case .closeUp: return "Facial close-up" }
  }
  var orientationInstruction: String {
    switch self {
    case .front: return "straight-on front view"
    case .side: return "exact 90-degree side profile, without turning toward camera"
    case .back: return "exact 180-degree rear view, showing the back of the head, no frontal face"
    case .closeUp: return "eye-level frontal head close-up"
    }
  }
}

public struct CharacterPanelPromptContext: Codable, Equatable {
  public var version = 1
  public var role: CharacterPanelRole
  public var definition: CharacterSheetDefinition
  public var preservesInputHead = false
  public var replacesHead: Bool
  public var detailLoRAID: String
  public var headLoRAID: String?
  public var prompt: String {
    let referenceOwnsHead = replacesHead || preservesInputHead
    let scopedDefinition = role == .closeUp ? closeUpDefinition : referenceOwnsHead ? bodyDefinition : definition
    let compiled = CharacterSheetCompiler.compile(scopedDefinition)
    let sections = compiled.sections.filter { section in
      // The reference head supplies identity when swapping. No four-panel/collage instructions leak here.
      let included = role == .closeUp
        ? (referenceOwnsHead ? [9, 11] : [2, 4, 5, 7, 8, 9, 11])
        : (referenceOwnsHead ? [3, 6, 7, 8, 9, 11] : Array(2...9) + [11])
      return included.contains(section.index)
    }.map(\.text).filter { !$0.isEmpty }.joined(separator: ". ")
    if role == .closeUp {
      let start = replacesHead
        ? "head_swap: replace the head with the reference head. high quality. Image 1 is the target facial close-up; Image 2 supplies the reference head. Match the reference head identity and hair to the target head rotation, expression and lighting."
        : "high quality. Refine the facial close-up in Image 1, preserving the subject's identity."
      return start + " Preserve the exact tight face/head crop and subject scale of Image 1, including the head position, eye-level frontal view and plain solid white background. Do not zoom out or reveal torso, legs or feet. Preserve any existing clothing only where it is already visible at the crop edge. Improve visible facial and hair surface detail without adding marks, accessories or changing the design. "
        + sections + photographicTextureInstruction + ". Output one facial close-up only, no collage, no additional views or panel labels."
    }
    let start = replacesHead
      ? "head_swap: replace the head with the reference head. high quality. Image 1 is the target character panel; Image 2 supplies the reference head. Match the reference head identity and hair to the target head rotation and blend the head/body junction."
      : "high quality. Refine the single character image in Image 1, preserving the subject's identity."
    return start + " Preserve the \(role.orientationInstruction), pose, silhouette, body anatomy, outfit construction, local colors and plain solid white background. Improve visible surface detail without adding marks, accessories or changing the design. "
      + sections + photographicTextureInstruction + ". Output one image of this view only, no collage, no additional views or panel labels."
  }

  private var photographicTextureInstruction: String {
    guard ["photograph", "cinematicPhotograph"].contains(definition.settings.stylePresetID) else { return "" }
    // Texture preservation applies even when a replacement head owns appearance and
    // canonical face fields are deliberately omitted from the refinement prompt.
    return ". Preserve the reference images' visible surface texture and tonal variation at natural scale; no beauty retouching, airbrushing or waxy smoothing. Retain visible pores, fine lines and skin color variation where present, without inventing or exaggerating them"
  }

  /// Keep authored detail only when it belongs to the visible head. Unlocated
  /// records and full outfits can otherwise make an edit model expand the crop.
  private var closeUpDefinition: CharacterSheetDefinition {
    var result = definition
    let headWords: Set<String> = ["head", "face", "facial", "scalp", "forehead", "temple", "temples",
      "eye", "eyes", "eyebrow", "eyebrows", "eyelid", "eyelids", "ear", "ears", "nose", "muzzle",
      "beak", "mouth", "lip", "lips", "chin", "cheek", "cheeks", "jaw", "beard", "mustache",
      "moustache", "hair", "neck", "throat"]
    func isHeadLocation(_ value: String) -> Bool {
      !Set(value.lowercased().split { !$0.isLetter }.map(String.init)).isDisjoint(with: headWords)
    }
    func text(_ record: CharacterRepeatableRecord, _ key: String) -> String {
      record.fields[key]?.state == .value ? record.fields[key]?.displayString ?? "" : ""
    }
    result.appearance.garments = []
    result.appearance.accessories.removeAll { !isHeadLocation(text($0, "placement")) }
    result.appearance.features.removeAll { !isHeadLocation(text($0, "placement")) }
    let retainedTargets = Set(result.appearance.accessories.map(\.id))
    result.appearance.surfaces.removeAll { record in
      let target = text(record, "target")
      let targetID: UUID?
      if case .choice(let id, _)? = record.fields["target"]?.value { targetID = UUID(uuidString: id) ?? UUID(uuidString: target) }
      else { targetID = UUID(uuidString: target) }
      if let targetID { return !retainedTargets.contains(targetID) }
      return !isHeadLocation(target)
    }
    if replacesHead || preservesInputHead {
      result.appearance.fields = result.appearance.fields.filter { key, _ in
        guard let section = CharacterFieldCatalog.shared.field(for: key)?.section else { return true }
        return section != 4 && section != 5
      }
    }
    return result
  }

  /// Scope the prompt only; the accepted definition and its provenance stay intact.
  /// Unlocated marks/accessories must not compete with the authoritative input head.
  private var bodyDefinition: CharacterSheetDefinition {
    var result = definition
    func text(_ record: CharacterRepeatableRecord, _ key: String) -> String {
      guard record.fields[key]?.state == .value else { return "" }
      return record.fields[key]?.displayString ?? ""
    }
    func words(_ text: String) -> Set<String> {
      Set(text.lowercased().split { !$0.isLetter }.map(String.init))
    }
    let headWords: Set<String> = ["head", "face", "facial", "scalp", "forehead", "temple", "temples",
      "eye", "eyes", "eyebrow", "eyebrows", "eyelid", "eyelids", "ear", "ears", "nose", "muzzle",
      "beak", "mouth", "lip", "lips", "chin", "cheek", "cheeks", "jaw", "beard", "mustache",
      "moustache", "helmet", "hat", "cap", "hood", "crown", "tiara", "mask", "glasses", "eyewear",
      "earring", "earrings", "hairpin", "wig", "neck", "throat"]
    let bodyWords: Set<String> = ["body", "torso", "chest", "back", "shoulder", "shoulders", "arm", "arms",
      "upperarm", "forearm", "forearms", "elbow", "elbows", "wrist", "wrists", "hand", "hands",
      "finger", "fingers", "waist", "abdomen", "stomach", "hip", "hips", "pelvis", "leg", "legs",
      "thigh", "thighs", "knee", "knees", "shin", "shins", "ankle", "ankles", "foot", "feet",
      "toe", "toes", "paw", "paws", "hoof", "hooves", "tail", "wing", "wings", "fin", "fins"]
    func locatedOnBody(_ location: String) -> Bool {
      let tokens = words(location)
      return tokens.isDisjoint(with: headWords) && !tokens.isDisjoint(with: bodyWords)
    }
    func isHeadSensitive(_ record: CharacterRepeatableRecord) -> Bool {
      let tokens = words([text(record, "type"), text(record, "bodyRegion"), text(record, "placement")].joined(separator: " "))
      return !tokens.isDisjoint(with: headWords)
    }
    result.appearance.garments.removeAll { isHeadSensitive($0) }
    result.appearance.accessories.removeAll { isHeadSensitive($0) || !locatedOnBody(text($0, "placement")) }
    result.appearance.features.removeAll { isHeadSensitive($0) || !locatedOnBody(text($0, "placement")) }
    let retainedTargets = Set((result.appearance.garments + result.appearance.accessories).map(\.id))
    result.appearance.surfaces.removeAll { record in
      let target = text(record, "target")
      let targetID: UUID?
      if case .choice(let id, _)? = record.fields["target"]?.value { targetID = UUID(uuidString: id) ?? UUID(uuidString: target) }
      else { targetID = UUID(uuidString: target) }
      if let targetID { return !retainedTargets.contains(targetID) }
      return !locatedOnBody(target)
    }
    // Preset applicability must not reintroduce the canonical head's hair or skin.
    result.appearance.fields = result.appearance.fields.filter { key, _ in
      guard let section = CharacterFieldCatalog.shared.field(for: key)?.section else { return true }
      return section != 4 && section != 5
    }
    return result
  }
}

public enum CharacterPanelRecipe {
  public static func headOnlyDraft(from source: DrawThingsImageDraft) throws -> DrawThingsImageDraft {
    guard var context = source.characterPanel, context.replacesHead,
      let headID = context.headLoRAID,
      let head = source.loras.first(where: { $0.modelID == headID && $0.isEnabled && $0.weight > 0 }) else {
      throw StudioError.invalid("The head-only pass requires its enabled BFS adapter and reference head.")
    }
    var result = source
    context.detailLoRAID = headID
    result.characterPanel = context; result.loras = [head]; result.prompt = context.prompt
    if let issue = result.managedCharacterPromptIssue { throw StudioError.invalid(issue) }
    return result
  }

  public static func makeDraft(role: CharacterPanelRole, definition: CharacterSheetDefinition,
    settings: CharacterRefinementSettings, profileID: String, panelPath: String, headPath: String?,
    width: Int, height: Int, documentID: UUID, seed: Int,
    preservesInputHead: Bool = false) throws -> DrawThingsImageDraft {
    let compiled = CharacterSheetCompiler.compile(definition)
    guard compiled.canGenerate else {
      throw StudioError.invalid(compiled.diagnostics.first?.message ?? "Resolve the required character fields.")
    }
    guard !settings.modelID.isEmpty, !settings.detailLoRAID.isEmpty else {
      throw StudioError.invalid("Choose FLUX.2 klein 9B and its compatible HichResolution9B LoRA.")
    }
    guard width > 0, height > 0, width <= 4096, height <= 4096, width % 64 == 0, height % 64 == 0 else {
      throw StudioError.invalid("Prepare the panel at its pass resolution with white padding to 64-pixel dimensions.")
    }
    guard (0...Int(UInt32.max)).contains(seed), (1...100).contains(settings.steps),
      settings.guidance.isFinite, (0...30).contains(settings.guidance) else {
      throw StudioError.invalid("Use a 32-bit seed, 1–100 steps and finite guidance from 0–30.")
    }
    if settings.replaceFaces && (settings.headLoRAID.isEmpty || headPath?.isEmpty != false) {
      throw StudioError.invalid("Replace faces requires the prepared reference head and compatible BFS rank-64 LoRA.")
    }
    var draft = DrawThingsImageDraft(destination: .init(scope: .global, projectID: documentID))
    draft.profileID = profileID; draft.modelID = settings.modelID; draft.name = role.label + " refinement"
    draft.width = width; draft.height = height; draft.steps = settings.steps; draft.guidance = settings.guidance; draft.seed = seed
    draft.moodboard = [ImageWorkspaceInput(path: panelPath)]
    draft.loras = [DrawThingsLoRA(modelID: settings.detailLoRAID, weight: settings.detailStrength)]
    if settings.replaceFaces, let headPath {
      draft.moodboard.append(ImageWorkspaceInput(path: headPath))
      draft.loras.append(DrawThingsLoRA(modelID: settings.headLoRAID, weight: settings.headStrength))
    }
    draft.characterPanel = CharacterPanelPromptContext(role: role, definition: definition,
      preservesInputHead: preservesInputHead, replacesHead: settings.replaceFaces, detailLoRAID: settings.detailLoRAID,
      headLoRAID: settings.replaceFaces ? settings.headLoRAID : nil)
    draft.prompt = draft.characterPanel!.prompt
    return draft
  }
}

public extension DrawThingsImageDraft {
  var managedCharacterPromptIssue: String? {
    if let context = characterPanel {
      let compiled = CharacterSheetCompiler.compile(context.definition)
      guard compiled.canGenerate else { return compiled.diagnostics.first?.message ?? "Resolve the required character fields." }
      guard context.version == 1, prompt == context.prompt else { return "This panel's managed prompt changed. Rebuild it from the character fields." }
      guard executionProvider == .drawThings, canvas == nil,
        moodboard.filter({ $0.enabled && $0.strength > 0 }).count == (context.replacesHead ? 2 : 1) else {
        return "Panel refinement requires the ordered target image and, when enabled, reference head."
      }
      let ids = [context.detailLoRAID] + (context.headLoRAID.map { [$0] } ?? [])
      guard ids.allSatisfy({ id in loras.contains { $0.modelID == id && $0.isEnabled && $0.weight > 0 } }) else {
        return "Enable the required panel refinement LoRAs."
      }
    }
    if referenceSheet?.template == .characterSheet {
      guard let definition = referenceSheet?.characterDefinition else {
        return "Map and review this legacy character description in Character Director before generating a new sheet."
      }
      let compiled = CharacterSheetCompiler.compile(definition)
      guard compiled.canGenerate else { return compiled.diagnostics.first?.message ?? "Resolve the required character fields." }
      guard prompt == compiled.prompt else { return "This character-sheet prompt changed. Rebuild it from the character fields." }
      guard width == 1920, height == 1088 else { return "Initial character sheets must be 1920 × 1088." }
    }
    return nil
  }
  var hasManagedCharacterPrompt: Bool {
    characterPanel != nil || referenceSheet?.template == .characterSheet || referenceSheet?.characterDefinition != nil
  }
}
