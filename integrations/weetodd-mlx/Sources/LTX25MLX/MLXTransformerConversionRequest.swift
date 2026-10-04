import Foundation
import CoreFoundation
import LTX25Engine

/// Header-only admission for native transformer conversion. The worker owns
/// scheduling and cancellation; this transport cannot choose inference tasks.
public struct MLXTransformerConversionRequest {
  public let source:URL
  public let outputDirectory:URL
  public let sourceIdentity:MLXTransformerPageConverter.SourceIdentity?

  public init(data:Data,outputDirectory:URL,requiresSourceIdentity:Bool) throws {
    guard data.count<=64*1024,
      let object=try JSONSerialization.jsonObject(with:data) as? [String:Any],
      Set(object.keys).isSubset(of:["version","engine","task","source_path","output_directory","source_identity"]),
      let version=object["version"] as? NSNumber,CFGetTypeID(version) != CFBooleanGetTypeID(),version==1,
      object["engine"] as? String=="ltx25",object["task"] as? String=="transformer-page-conversion",
      let input=object["source_path"] as? String,let output=object["output_directory"] as? String else {
      throw LTXError.invalid("Expected a version-1 native LTX transformer conversion request.")
    }
    source=try Self.localPath(input)
    self.outputDirectory=try Self.localPath(output)
    guard outputDirectory.isFileURL,
      self.outputDirectory.path==(try MLXTransformerPageConverter.canonicalLocalURL(outputDirectory)).path else {
      throw LTXError.invalid("Transformer conversion output must match the worker output directory.")
    }
    if let captured=object["source_identity"] {
      guard let fields=captured as? [String:Any],Set(fields.keys)==Self.identityKeys else {
        throw LTXError.invalid("Transformer source identity has unknown or missing fields.")
      }
      let identity=try JSONDecoder().decode(MLXTransformerPageConverter.SourceIdentity.self,
        from:JSONSerialization.data(withJSONObject:fields))
      try Self.validate(identity)
      sourceIdentity=identity
    } else {
      guard !requiresSourceIdentity else {
        throw LTXError.invalid("Conversion requires the source identity captured by preflight.")
      }
      sourceIdentity=nil
    }
  }

  /// Reads and validates source headers, then checks the captured identity.
  /// The returned plan is the only input accepted by the payload converter.
  public func preflight() throws -> MLXTransformerPageConverter.Plan {
    let plan=try MLXTransformerPageConverter.preflight(source:source)
    if let sourceIdentity,sourceIdentity != plan.sourceIdentity {
      throw LTXError.invalid("Transformer source identity differs from the captured preflight.")
    }
    try plan.checkUnchanged()
    return plan
  }

  private static let identityKeys:Set<String>=["device","inode","bytes","modifiedSeconds","modifiedNanos",
    "changedSeconds","changedNanos","headerSHA256"]
  private static func validate(_ identity:MLXTransformerPageConverter.SourceIdentity) throws {
    guard identity.inode>0,identity.bytes>0,identity.bytes<=UInt64(Int64.max),
      (0..<1_000_000_000).contains(identity.modifiedNanos),(0..<1_000_000_000).contains(identity.changedNanos),
      identity.headerSHA256.utf8.count==64,
      identity.headerSHA256.utf8.allSatisfy({ (48...57).contains($0)||(97...102).contains($0) }) else {
      throw LTXError.invalid("Malformed captured transformer source identity.")
    }
  }
  private static func localPath(_ path:String) throws -> URL {
    guard path.hasPrefix("/"),!path.utf8.contains(0),path.utf8.count<=4096,
      !path.isEmpty else { throw LTXError.invalid("Transformer conversion paths must be bounded absolute local paths.") }
    let url=URL(fileURLWithPath:path)
    guard !["/",".",".."].contains(url.lastPathComponent) else {
      throw LTXError.invalid("Transformer conversion requires named source and output paths.")
    }
    return try MLXTransformerPageConverter.canonicalLocalURL(url)
  }
}
