import Darwin
import Foundation

/// Select paths once, before checksum admission. A hash mismatch must not cause
/// another lookup. Canonical URLs allow compatible installed symlinks in place.
public struct NativeFlashVSRAssets:Sendable {
  public enum Policy:Sendable { case native,legacyVpipe }
  public let transformer:URL,projection:URL,context:URL,decoder:URL
  public var files:[URL] { [transformer,projection,context,decoder] }
  public static func resolve(root:URL,variant:NativeFlashVSRRecipe.Variant,
    decoder:NativeFlashVSRRecipe.Decoder,policy:Policy = .native)throws->Self {
    guard root.isFileURL,!root.path.utf8.contains(0) else { throw ContractError.invalid("FlashVSR assets require a local model directory.") }
    let folder=root.appendingPathComponent(variant.rawValue)
    func canonicalRegular(_ url:URL)->URL? {
      let canonical=url.standardizedFileURL.resolvingSymlinksInPath()
      return (try? canonical.resourceValues(forKeys:[.isRegularFileKey]).isRegularFile) == true ? canonical:nil
    }
    func require(_ url:URL)throws->URL {
      guard let selected=canonicalRegular(url) else { throw ContractError.invalid("Selected FlashVSR component is missing or not a regular file: \(url.path)") }
      return selected
    }
    func shared(_ name:String)throws->URL {
      let original=root.appendingPathComponent(name),local=folder.appendingPathComponent(name)
      if policy == .legacyVpipe {
        // Preserve vpipe's established regular-file variant-first search.
        return try canonicalRegular(local) ?? require(original)
      }
      var info=stat()
      if lstat(original.path,&info) == 0 { return try require(original) }
      guard errno == ENOENT else { throw ContractError.invalid("Cannot inspect the selected FlashVSR component: \(original.path)") }
      return try require(local)
    }
    return try .init(transformer:require(folder.appendingPathComponent("dit.safetensors")),
      projection:require(folder.appendingPathComponent("lq_proj_in.safetensors")),context:shared("posi_prompt.pth"),
      decoder:shared(decoder == .tiny ? "TCDecoder.ckpt":"Wan2.1_VAE.pth"))
  }
}
