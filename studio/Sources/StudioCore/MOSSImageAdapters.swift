import Foundation

extension DrawThingsImageDraft {
  public mutating func applyMOSSLoRAs(_ members: [LoRAMember], mode: LoRAGroupApplicationMode = .add) throws {
    guard executionProvider == .nativeMOSS else { throw StudioError.invalid("Choose Local MOSS first.") }
    for member in members { try member.validateForMOSS() }
    let current = mode == .replace ? [] : mossImage?.loras ?? []
    let next = current + members
    guard next.count <= 8, Set(next.map(\.fileKey)).count == next.count else { throw StudioError.invalid("A MOSS stack supports up to eight distinct adapters. Remove duplicates or choose Replace.") }
    var candidate = self
    if candidate.mossImage == nil { candidate.mossImage = .init() }
    candidate.mossImage?.loras = next
    try candidate.reconcileMOSSFastRecipe(); self = candidate
  }
  public mutating func setMOSSAdapterEnabled(id: UUID, enabled: Bool) throws {
    var candidate = self
    guard let index = candidate.mossImage?.loras?.firstIndex(where: { $0.id == id }) else { return }
    candidate.mossImage?.loras?[index].enabled = enabled
    try candidate.reconcileMOSSFastRecipe(); self = candidate
  }
  public mutating func reconcileMOSSFastRecipe() throws {
    let fast = (mossImage?.loras ?? []).filter { $0.isEnabled && $0.strength != 0 && $0.asset.loraProfile == "moss_sft_4step_v2" }
    if !fast.isEmpty {
      guard fast.count == 1, mossImage?.variant == "sft", fast[0].strength == 1 else { throw StudioError.invalid("The four-step adapter requires SFT and strength 1.") }
      if mossImage?.stockRecipe == nil {
        let saved = MOSSRecipeSnapshot(draft: self); mossImage?.stockRecipe = saved
        steps = 4; guidance = 1; shift = 5
        mossImage?.sampler = "euler"; mossImage?.schedule = "release"
      }
    } else if let stock = mossImage?.stockRecipe {
      stock.apply(to: &self); mossImage?.stockRecipe = nil
    }
  }
}
