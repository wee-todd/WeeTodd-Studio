import Foundation
extension GenerationSelection {
  mutating func mapH3CreativePaths(_ transform:(String)->String) {
    if let source=h3MotionFidelity?.sourceVideo {h3MotionFidelity?.sourceVideo=transform(source)}
    if let model=h3Joint?.refinement?.learnedUpscalerPath {h3Joint?.refinement?.learnedUpscalerPath=transform(model)}
    if let model=h3Joint?.twoStage?.draftTurboPath {h3Joint?.twoStage?.draftTurboPath=transform(model)}
    if let model=h3Joint?.twoStage?.learnedUpscalerPath {h3Joint?.twoStage?.learnedUpscalerPath=transform(model)}
    if let manifest=h3Joint?.refinement?.source.manifest { h3Joint?.refinement?.source.manifest=transform(manifest) }
  }
}
