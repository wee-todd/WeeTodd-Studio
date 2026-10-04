import Foundation
import CoreFoundation
import CryptoKit
import Darwin

/// Explicit user intent; nil on existing clips retains manual timing.
public struct LTX25AutomaticDurationSettings: Codable, Equatable {
  public var experimentalEnabled: Bool
  public var minimumSeconds: Double
  public var maximumSeconds: Double
  public init(experimentalEnabled: Bool = false, minimumSeconds: Double = 1, maximumSeconds: Double = 20) {
    self.experimentalEnabled = experimentalEnabled
    self.minimumSeconds = minimumSeconds; self.maximumSeconds = maximumSeconds
  }
}

/// Weight-free Studio admission. The shared worker owns trained duration prediction.
public enum NativeLTXAutomaticDuration {
  static let headShapes: [String: [Int]] = [
    "video_input_proj.weight": [256,4096], "video_input_proj.bias": [256], "video_modality_emb": [256],
    "audio_input_proj.weight": [256,2048], "audio_input_proj.bias": [256], "audio_modality_emb": [256],
    "attention_pooler.query_tokens": [1,256], "attention_pooler.cross_attn.in_proj_weight": [768,256],
    "attention_pooler.cross_attn.in_proj_bias": [768], "attention_pooler.cross_attn.out_proj.weight": [256,256],
    "attention_pooler.cross_attn.out_proj.bias": [256], "mlp_hidden.weight": [256,256],
    "mlp_hidden.bias": [256], "mlp_out.weight": [1,256], "mlp_out.bias": [1]]
  public static func maximumFrames(minimumSeconds: Double, maximumSeconds: Double, fps: Double) throws -> Int {
    guard minimumSeconds.isFinite, maximumSeconds.isFinite, (0.25...30).contains(minimumSeconds),
      maximumSeconds >= minimumSeconds, maximumSeconds <= 30, fps.isFinite, (1...120).contains(fps) else {
      throw StudioError.invalid("Automatic duration requires finite ordered bounds within 0.25–30 seconds.")
    }
    let minimum = Int((minimumSeconds*fps).rounded(.toNearestOrEven))
    let maximum = Int((maximumSeconds*fps).rounded(.toNearestOrEven))
    let firstGrid = ((max(0,minimum-1)+7)/8)*8+1
    let lastGrid = ((max(0,maximum-1))/8)*8+1
    guard firstGrid <= lastGrid, lastGrid <= maximum else {
      throw StudioError.invalid("Automatic duration bounds contain no valid 8k+1 frame count.")
    }
    return lastGrid
  }
  /// Reads a regular header only, with explicit tensor lengths and complete payload bounds.
  public static func validateHead(at url: URL) throws -> String {
    guard url.isFileURL, url.path.utf8.count <= 4096, !url.path.utf8.contains(0) else {
      throw StudioError.invalid("Choose a local duration-head checkpoint.")
    }
    let source = url.resolvingSymlinksInPath().standardizedFileURL
    let fd = Darwin.open(source.path, O_RDONLY | O_CLOEXEC | O_NONBLOCK)
    guard fd >= 0 else { throw StudioError.invalid("Relink the LTX 2.5 duration head.") }
    let file = FileHandle(fileDescriptor:fd,closeOnDealloc:true); defer { try? file.close() }
    var before = stat()
    guard fstat(fd,&before) == 0, before.st_mode & S_IFMT == S_IFREG,
      (9...16*1024*1024).contains(before.st_size),
      let prefix = try file.read(upToCount:8), prefix.count == 8 else {
      throw StudioError.invalid("Duration head must be a regular file under 16 MiB.")
    }
    let length = prefix.enumerated().reduce(UInt64(0)) { $0 | UInt64($1.element) << (8*$1.offset) }
    guard length > 0, length <= 1024*1024, length + 8 <= UInt64(before.st_size),
      let bytes = try file.read(upToCount:Int(length)), bytes.count == Int(length),
      let header = try JSONSerialization.jsonObject(with:bytes) as? [String:Any],
      let metadata = header["__metadata__"] as? [String:String], ["2.5","2.5.0"].contains(metadata["model_version"] ?? ""),
      let configString = metadata["config"], configString.utf8.count <= 256*1024,
      let configData = configString.data(using:.utf8),
      let config = try JSONSerialization.jsonObject(with:configData) as? [String:Any],
      let transformer = config["transformer"] as? [String:Any], let head = config["duration_head"] as? [String:Any] else {
      throw StudioError.invalid("Choose a compatible official LTX 2.5 BF16 duration head.")
    }
    func number(_ raw: Any?) -> Int? {
      guard let value = raw as? NSNumber, CFGetTypeID(value) != CFBooleanGetTypeID(),
        value.doubleValue.isFinite, value.doubleValue >= 0, value.doubleValue <= 16*1024*1024,
        value.doubleValue.rounded() == value.doubleValue else { return nil }
      return value.intValue
    }
    func dimension(_ values: [String:Any], _ key: String, _ expected: Int) -> Bool {
      values[key] == nil || number(values[key]) == expected
    }
    guard dimension(transformer,"cross_attention_dim",4096), dimension(transformer,"audio_cross_attention_dim",2048),
      dimension(head,"pooler_hidden_dim",256), dimension(head,"num_queries",1),
      dimension(head,"num_pooler_heads",4), dimension(head,"mlp_hidden",256),
      Set(header.keys.filter { $0 != "__metadata__" }) == Set(headShapes.keys.map { "duration_head."+$0 }) else {
      throw StudioError.invalid("Duration-head architecture is incompatible.")
    }
    var spans: [(Int,Int)] = []
    for (name,shape) in headShapes {
      guard let record = header["duration_head."+name] as? [String:Any], record["dtype"] as? String == "BF16",
        let dimensions = record["shape"] as? [Any], dimensions.compactMap(number) == shape,
        dimensions.count == shape.count, let offsets = record["data_offsets"] as? [Any], offsets.count == 2,
        let start = number(offsets[0]), let end = number(offsets[1]), end >= start,
        end-start == shape.reduce(1,*)*2 else { throw StudioError.invalid("Duration-head tensor layout is incompatible.") }
      spans.append((start,end))
    }
    var end = 0
    for span in spans.sorted(by: { $0.0 < $1.0 }) {
      guard span.0 == end else { throw StudioError.invalid("Duration-head payload is overlapping or incomplete.") }
      end = span.1
    }
    var after = stat(), current = stat()
    guard UInt64(end)+length+8 == UInt64(before.st_size), fstat(fd,&after) == 0, lstat(source.path,&current) == 0,
      after.st_size == before.st_size, after.st_mtimespec.tv_sec == before.st_mtimespec.tv_sec,
      after.st_mtimespec.tv_nsec == before.st_mtimespec.tv_nsec, after.st_ctimespec.tv_sec == before.st_ctimespec.tv_sec,
      after.st_ctimespec.tv_nsec == before.st_ctimespec.tv_nsec, current.st_dev == before.st_dev,
      current.st_ino == before.st_ino else { throw StudioError.invalid("Duration head changed during header admission.") }
    return SHA256.hash(data:bytes).map { String(format:"%02x",$0) }.joined()
  }
}
