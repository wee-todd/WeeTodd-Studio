import Foundation
import ImageLayout

enum MOSSLayoutDetailGeometry {
  static func face(in subject: NormalizedLayoutRect, candidates: [NormalizedLayoutRect], preferred: NormalizedLayoutRect? = nil) -> NormalizedLayoutRect? {
    candidates.filter { face in
      let x = face.x + face.width/2, y = face.y + face.height/2
      return x >= subject.x && x <= subject.x+subject.width && y >= subject.y && y <= subject.y+subject.height
    }.max { a,b in
      func score(_ face: NormalizedLayoutRect) -> Double {
        guard let preferred else { return face.width*face.height }
        return -pow((face.x+face.width/2)-(preferred.x+preferred.width/2),2)-pow((face.y+face.height/2)-(preferred.y+preferred.height/2),2)
      }
      return score(a) < score(b)
    }
  }
  static func padded(_ b: NormalizedLayoutRect, fraction: Double) -> NormalizedLayoutRect {
    let x = max(0, b.x-b.width*fraction), y = max(0, b.y-b.height*fraction)
    return .init(x: x, y: y, width: min(1, b.x+b.width*(1+fraction))-x, height: min(1, b.y+b.height*(1+fraction))-y)
  }
}
