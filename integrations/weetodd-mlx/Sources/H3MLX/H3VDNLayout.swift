import Foundation

/// The target video is a dense frame-major interval; prompt and audio rows
/// stay global. This plan never allocates a sequence-squared attention mask.
struct H3VDNLayout {
  struct AttentionGroup { let query:Range<Int>;let keys:[Range<Int>] }
  let sequence:Int
  let videoStart:Int
  let frames:Int
  let height:Int
  let width:Int
  let textStart:Int
  let textLength:Int
  let tokensPerFrame:Int
  let videoEnd:Int

  init(packed:H3PackedLayout) throws {
    guard packed.conditionVideoRows == 0,packed.positions.count == packed.tags.count,
      packed.audioStart > 0,packed.videoStart < packed.positions.count,
      packed.tags.prefix(packed.audioStart).allSatisfy({ $0 == 1 }) else {
      throw H3CheckpointError.invalid("Experimental VDN currently requires plain T2VA packing without endpoint or visual-reference rows.")
    }
    let positions=Array(packed.positions[packed.videoStart...])
    guard positions.allSatisfy({ $0.x.isFinite && $0.y.isFinite && $0.z.isFinite }) else {
      throw H3CheckpointError.invalid("VDN target positions must be finite.")
    }
    let times=Set(positions.map(\.x)).sorted(),heights=Set(positions.map(\.y)).sorted(),widths=Set(positions.map(\.z)).sorted()
    try self.init(sequence:packed.tags.count,videoStart:packed.videoStart,frames:times.count,
      height:heights.count,width:widths.count,textStart:0,textLength:packed.audioStart)
    var index=0
    for time in times { for height in heights { for width in widths {
      guard index < positions.count,positions[index] == SIMD3(time,height,width) else {
        throw H3CheckpointError.invalid("VDN target rows require one dense frame-major grid.")
      }
      index += 1
    } } }
    guard index == positions.count else { throw H3CheckpointError.invalid("VDN target grid has extra rows.") }
  }

  init(sequence:Int,videoStart:Int,frames:Int,height:Int,width:Int,
    textStart:Int,textLength:Int) throws {
    guard (1...64_000).contains(sequence),(1...107).contains(frames),height > 0,width > 0,
      videoStart >= 0,textStart >= 0,textLength > 0 else {
      throw H3CheckpointError.invalid("Invalid VDN packed grid.")
    }
    let (perFrame,a)=height.multipliedReportingOverflow(by:width)
    let (rows,b)=perFrame.multipliedReportingOverflow(by:frames)
    let (end,c)=videoStart.addingReportingOverflow(rows)
    let (textEnd,d)=textStart.addingReportingOverflow(textLength)
    guard !a,!b,!c,!d,end <= sequence,textEnd <= sequence,
      textEnd <= videoStart || textStart >= end else {
      throw H3CheckpointError.invalid("VDN grid exceeds the packed sequence or overlaps prompt rows.")
    }
    self.sequence=sequence;self.videoStart=videoStart;self.frames=frames
    self.height=height;self.width=width;self.textStart=textStart;self.textLength=textLength
    tokensPerFrame=perFrame;videoEnd=end
  }

  func window(_ frame:Int) -> (lower:Int,upper:Int) {
    (((frame/5)-1)*5,((frame/5)+2)*5-1)
  }

  var attentionGroups:[AttentionGroup] {
    var groups:[AttentionGroup]=[]
    if videoStart > 0 { groups.append(.init(query:0..<videoStart,keys:[0..<sequence])) }
    var frame=0
    while frame < frames {
      let bounds=window(frame);var stop=frame+1
      if frame != 0 && frame != frames-1 {
        while stop < frames-1 && stop/5 == frame/5 { stop += 1 }
      }
      let query=(videoStart+frame*tokensPerFrame)..<(videoStart+stop*tokensPerFrame)
      if frame == 0 || frame == frames-1 {
        groups.append(.init(query:query,keys:[0..<sequence]))
      } else {
        let lower=max(bounds.lower,0),upper=min(bounds.upper,frames-1)
        var keys:[Range<Int>]=[]
        if videoStart > 0 { keys.append(0..<videoStart) }
        if videoEnd < sequence { keys.append(videoEnd..<sequence) }
        keys.append((videoStart+lower*tokensPerFrame)..<(videoStart+(upper+1)*tokensPerFrame))
        for anchor in [0,frames-1] where !(lower...upper).contains(anchor) {
          keys.append((videoStart+anchor*tokensPerFrame)..<(videoStart+(anchor+1)*tokensPerFrame))
        }
        groups.append(.init(query:query,keys:keys))
      }
      frame=stop
    }
    if videoEnd < sequence { groups.append(.init(query:videoEnd..<sequence,keys:[0..<sequence])) }
    return groups
  }
}
