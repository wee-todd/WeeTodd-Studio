import Foundation
import MLX

/// Segment-pure 64-row tiles: text/audio are linear; video uses 4×4×4 cubes.
/// CPU metadata only. Each original row has one slot; boundary padding is masked.
struct H3FastTiles: Sendable {
  let prefixTiles: Int
  let prefixRows: Int
  let rows: Int
  let indices: [Int32]
  let sizes: [Int]
  let rowSlots: [Int32]

  init(prefixSegments:[Int],videoGrid:[Int]) throws {
    guard !prefixSegments.isEmpty, prefixSegments.allSatisfy({ (1...40_000).contains($0) }),
      videoGrid.count == 3,videoGrid.allSatisfy({ (1...512).contains($0) }),
      prefixSegments.reduce(0,+)+videoGrid.reduce(1,*) <= 40_000 else {
      throw H3CheckpointError.invalid("Invalid FastH3 unmasked tile geometry.")
    }
    var tiles:[[Int]] = [],start = 0
    for count in prefixSegments {
      for offset in stride(from:0,to:count,by:64) {
        tiles.append(Array((start+offset)..<(start+min(offset+64,count))))
      }
      start += count
    }
    prefixTiles = tiles.count;prefixRows = start
    let time = videoGrid[0],height = videoGrid[1],width = videoGrid[2]
    for t in stride(from:0,to:time,by:4) {
      for h in stride(from:0,to:height,by:4) {
        for w in stride(from:0,to:width,by:4) {
          var cube:[Int] = []
          for ti in t..<min(t+4,time) {
            for hi in h..<min(h+4,height) {
              for wi in w..<min(w+4,width) { cube.append(start+(ti*height+hi)*width+wi) }
            }
          }
          tiles.append(cube)
        }
      }
    }
    rows = start+time*height*width
    sizes = tiles.map(\.count)
    var slots = [Int32](repeating:0,count:rows),gather:[Int32] = []
    for (tile,entries) in tiles.enumerated() {
      for (slot,row) in entries.enumerated() { slots[row] = Int32(tile*64+slot) }
      gather += entries.map(Int32.init)+[Int32](repeating:0,count:64-entries.count)
    }
    indices = gather;rowSlots = slots
  }
}

/// Trained FastH3 VSA: tile-pooled compression plus dense-prefix / sparse-video
/// attention. Per-head video gathers are bounded to eight query tiles; no global
/// token×token mask or dense weight conversion is constructed.
enum H3FastAttention {
  static func selectedTileCount(videoTiles:Int,sparsity:Double) -> Int {
    max(1,Int(ceil(Double(1-sparsity)*Double(videoTiles))))
  }
  static func evaluate(query:MLXArray,key:MLXArray,value:MLXArray,gate:MLXArray,
    tiles:H3FastTiles,sparsity:Double = 0.9,minimumSparseRows:Int = 4096) throws -> MLXArray {
    guard query.ndim == 4,query.shape == key.shape,key.shape == value.shape,
      query.shape == gate.shape,query.shape[0] == 1,query.shape[2] == tiles.rows,
      query.dtype == key.dtype,key.dtype == value.dtype,gate.dtype == value.dtype,
      query.dtype.isFloatingPoint,(0..<1).contains(sparsity),minimumSparseRows > 0 else {
      throw H3CheckpointError.invalid("Invalid FastH3 VSA tensors or sparsity.")
    }
    try Task.checkCancellation()
    let heads = query.shape[1],width = query.shape[3],count = tiles.sizes.count
    let scale = 1 / Float(width).squareRoot()
    let valid = tiles.sizes.flatMap { size in (0..<64).map { Float($0 < size ? 1 : 0) } }
    let mask = MLXArray(valid,[1,1,count*64,1])
    let indices = MLXArray(tiles.indices)
    func packed(_ input:MLXArray) -> MLXArray { take(input,indices,axis:2)*mask.asType(input.dtype) }
    let tq = packed(query),tk = packed(key),tv = packed(value)
    let sizes = MLXArray(tiles.sizes.map(Float.init),[1,1,count,1])
    func pooled(_ input:MLXArray) -> MLXArray {
      input.asType(.float32).reshaped([1,heads,count,64,width]).sum(axis:3)/sizes
    }
    let pq = pooled(tq),pk = pooled(tk),pv = pooled(tv)
    let scores = matmul(pq,pk.swappedAxes(-1,-2))*scale
    let compression = matmul(softmax(scores,axis:-1),pv)
    let rowTiles = MLXArray(tiles.rowSlots.map { $0/64 })
    let expanded = take(compression,rowTiles,axis:2).asType(value.dtype)
    eval([scores,expanded])
    let output:MLXArray
    if tiles.rows < minimumSparseRows {
      output = MLXFast.scaledDotProductAttention(queries:query,keys:key,values:value,scale:scale,mask:nil)
    } else {
      let prefix = tiles.prefixTiles,video = count-prefix
      let keep = selectedTileCount(videoTiles:video,sparsity:sparsity)
      let tiledKeys = tk.reshaped([heads*count*64,width])
      let tiledValues = tv.reshaped([heads*count*64,width])
      let headOffsets = MLXArray((0..<heads).map { Int32($0*count*64) },[1,heads,1,1,1])
      let localRows = MLXArray((0..<64).map(Int32.init),[1,1,1,1,64])
      let tiledValidity = MLXArray(valid)
      var groups:[MLXArray] = []
      for start in stride(from:0,to:video,by:8) {
        try Task.checkCancellation()
        let end = min(start+8,video),group = end-start
        let selected = argSort(-scores[.ellipsis,(prefix+start)..<(prefix+end),prefix..<count],axis:-1)[.ellipsis,0..<keep]+Int32(prefix)
        let allPrefix = broadcast(MLXArray((0..<prefix).map(Int32.init),[1,1,1,prefix]),to:[1,heads,group,prefix])
        let selectedTiles = concatenated([allPrefix,selected],axis:-1)
        let selectedRows = selectedTiles.expandedDimensions(axis:-1)*Int32(64)+localRows
        let gathered = (selectedRows+headOffsets).reshaped([-1])
        let keyRows = (prefix+keep)*64
        let keys = take(tiledKeys,gathered,axis:0).reshaped([heads*group,1,keyRows,width])
        let values = take(tiledValues,gathered,axis:0).reshaped([heads*group,1,keyRows,width])
        let validKeys = take(tiledValidity,selectedRows.reshaped([-1]),axis:0).reshaped([heads*group,1,1,keyRows])
        let attentionMask = which(validKeys .> 0,MLXArray(Float(0)),MLXArray(-Float.infinity)).asType(query.dtype)
        let queries = tq[.ellipsis,((prefix+start)*64)..<((prefix+end)*64),0..<width].reshaped([heads*group,1,64,width])
        let attended = MLXFast.scaledDotProductAttention(queries:queries,keys:keys,values:values,scale:scale,mask:attentionMask)
          .reshaped([1,heads,group*64,width])
        eval(attended);groups.append(attended)
      }
      let videoOutput = take(concatenated(groups,axis:2),MLXArray(tiles.rowSlots.dropFirst(tiles.prefixRows).map { $0-Int32(prefix*64) }),axis:2)
      let prefixOutput = MLXFast.scaledDotProductAttention(queries:query[.ellipsis,0..<tiles.prefixRows,0..<width],keys:key,values:value,scale:scale,mask:nil)
      output = concatenated([prefixOutput,videoOutput],axis:2)
    }
    let result = output+expanded*gate
    eval(result)
    try Task.checkCancellation()
    return result
  }
}
