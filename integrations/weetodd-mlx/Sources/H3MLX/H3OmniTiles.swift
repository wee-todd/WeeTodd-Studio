import Foundation

/// Packed spans retain their semantic role; image grids are never sparse videos.
enum H3OmniSegment: Sendable, Equatable {
  enum DenseRole: Sendable { case text, image, referenceAudio, targetAudio }
  case dense(role: DenseRole, rows: Int)
  case referenceVideo(grid: [Int])
}

/// Immutable CPU metadata for the trained 4×4×8, p2_multi_region layout.
/// Dense spans move first, each reference video keeps its own region, and the
/// target is last. Each span and each spatial tile has independent padding.
struct H3OmniVSAPlan: Sendable {
  static let tileShape = [4,4,8]
  static let tileRows = 128
  static let maximumWorkspaceBytes = 4 * 1024 * 1024 * 1024
  let prefixSegments: [H3OmniSegment]
  let targetVideoGrid: [Int]
  let rows: Int
  let denseTiles: Int
  let denseRows: [Int32]
  let videoRegions: [Range<Int>]
  let indices: [Int32]
  let rowSlots: [Int32]
  let denseOrdinals: [Int32]
  let sizes: [Int]
  var selectedTilesPerQuery: Int { denseTiles + videoRegions.reduce(0) { $0 + Self.keepCount($1.count) } }
  static func keepCount(_ count: Int) -> Int { max(1,(count+9)/10) }

  init(prefixSegments: [H3OmniSegment], targetVideoGrid: [Int]) throws {
    func gridRows(_ grid:[Int]) throws -> Int {
      guard grid.count == 3, grid.allSatisfy({(1...512).contains($0)}),grid.reduce(1,*) <= 40_000 else {
        throw H3CheckpointError.invalid("Invalid Omni VSA video grid.")
      }
      return grid.reduce(1,*)
    }
    var dense: [[Int]] = [], videos: [(Int,[Int])] = [], offset=0
    for segment in prefixSegments {
      let count:Int
      switch segment {
      case let .dense(_,rows):
        guard (1...40_000).contains(rows) else { throw H3CheckpointError.invalid("Invalid Omni dense span.") }
        count=rows
        for start in stride(from:0,to:rows,by:128) {
          dense.append(Array((offset+start)..<(offset+min(start+128,rows))))
        }
      case let .referenceVideo(grid):
        count=try gridRows(grid); videos.append((offset,grid))
      }
      guard offset <= 40_000-count else { throw H3CheckpointError.invalid("Omni VSA exceeds 40000 actual rows.") }
      offset += count
    }
    let targetRows=try gridRows(targetVideoGrid)
    guard offset <= 40_000-targetRows else { throw H3CheckpointError.invalid("Omni VSA exceeds 40000 actual rows.") }
    videos.append((offset,targetVideoGrid));rows=offset+targetRows
    self.prefixSegments=prefixSegments;self.targetVideoGrid=targetVideoGrid
    denseTiles=dense.count; denseRows=dense.flatMap{$0}.map(Int32.init)
    var tiles=dense,regions:[Range<Int>]=[]
    for (start,grid) in videos {
      let first=tiles.count,t=grid[0],h=grid[1],w=grid[2]
      for ti in stride(from:0,to:t,by:4) {
        for hi in stride(from:0,to:h,by:4) {
          for wi in stride(from:0,to:w,by:8) {
            var members:[Int]=[]
            for tt in ti..<min(ti+4,t) {
              for hh in hi..<min(hi+4,h) {
                for ww in wi..<min(wi+8,w) { members.append(start+(tt*h+hh)*w+ww) }
              }
            }
            tiles.append(members)
          }
        }
      }
      regions.append(first..<tiles.count)
    }
    videoRegions=regions;sizes=tiles.map(\.count)
    var gather:[Int32]=[],inverse=[Int32](repeating:0,count:rows)
    for (tile,members) in tiles.enumerated() {
      for (slot,row) in members.enumerated() { inverse[row]=Int32(tile*128+slot) }
      gather += members.map(Int32.init)+[Int32](repeating:0,count:128-members.count)
    }
    indices=gather;rowSlots=inverse
    var ordinals=[Int32](repeating:-1,count:rows)
    for (ordinal,row) in denseRows.enumerated() { ordinals[Int(row)]=Int32(ordinal) }
    denseOrdinals=ordinals
    // The real model has 56 heads: reject pathological padding/segment counts
    // before weighted execution, not after allocating quadratic tile scores.
    _ = try workspaceBytes(heads:56)
  }

  /// Conservative live-array bound, excluding caller-owned Q/K/V/gate and the
  /// MLX allocator cache. Includes scores/sort scratch, summaries, routes,
  /// dense/sparse outputs, final output and up to three feature-view copies.
  func workspaceBytes(heads:Int) throws -> Int {
    guard (1...56).contains(heads) else { throw H3CheckpointError.invalid("Invalid Omni head count.") }
    let tiles=sizes.count,sparse=tiles-denseTiles
    let scores=4*heads*tiles*tiles*4
    let summaries=4*heads*tiles*128*4
    let routes=4*heads*sparse*selectedTilesPerQuery
    let activations=2*heads*128*(sparse*128+denseRows.count*2+rows*4)
    let bytes=scores+summaries+routes+activations+4*(indices.count+rows*2+tiles)
    guard bytes <= Self.maximumWorkspaceBytes else {
      throw H3CheckpointError.invalid("Omni VSA workspace exceeds its 4 GiB bound.")
    }
    return bytes
  }
}
