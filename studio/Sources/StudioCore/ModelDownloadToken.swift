import Darwin
import Foundation
import Security

/// Credentials for explicit model downloads. Secrets stay out of Studio's persisted settings.
public enum ModelDownloadToken {
  private static let service = "org.weetodd.studio.model-download"
  private static let account = "huggingface"

  private static var query: [String: Any] {
    [
      kSecClass as String: kSecClassGenericPassword,
      kSecAttrService as String: service, kSecAttrAccount as String: account,
    ]
  }

  public static func normalized(_ token: String) throws -> String {
    let value = token.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !value.isEmpty,
      !value.unicodeScalars.contains(where: {
        CharacterSet.whitespacesAndNewlines.contains($0)
          || CharacterSet.controlCharacters.contains($0)
      })
    else { throw StudioError.invalid("Enter a single Hugging Face read token.") }
    return value
  }

  public static func environment(_ original: [String: String], savedToken: String?) throws
    -> [String: String]
  {
    guard let savedToken else { return original }
    var result = original
    result["HF_TOKEN"] = try normalized(savedToken)
    return result
  }

  /// Reuse the existing Studio or Hugging Face login without copying credentials.
  /// Call off the UI executor because Keychain can require OS interaction.
  public static func resolveExisting(environment:[String:String]=ProcessInfo.processInfo.environment) throws -> String? {
    try resolveExisting(environment:environment,home:FileManager.default.homeDirectoryForCurrentUser,keychainReader:read)
  }

  static func resolveExisting(environment:[String:String],home:URL,keychainReader:() throws -> String?) throws -> String? {
    if let saved=try keychainReader() { return try normalized(saved) }
    if let token=environment["HF_TOKEN"] { return try normalized(token) }
    for key in ["HF_TOKEN_PATH","HF_HOME","XDG_CACHE_HOME"] {
      if let value=environment[key],!value.hasPrefix("/") || value.utf8.contains(0) {
        throw StudioError.invalid("Configured Hugging Face credential locations must be absolute local paths.")
      }
    }
    let path:String
    if let explicit=environment["HF_TOKEN_PATH"] { path=explicit }
    else if let root=environment["HF_HOME"] { path=URL(fileURLWithPath:root).appendingPathComponent("token").path }
    else if let cache=environment["XDG_CACHE_HOME"] { path=URL(fileURLWithPath:cache).appendingPathComponent("huggingface/token").path }
    else { path=home.appendingPathComponent(".cache/huggingface/token").path }
    guard path.hasPrefix("/"),!path.utf8.contains(0),path.utf8.count<=4096 else {
      throw StudioError.invalid("The configured Hugging Face token path must be an absolute local file.")
    }
    let descriptor=Darwin.open(path,O_RDONLY|O_CLOEXEC|O_NONBLOCK|O_NOFOLLOW)
    if descriptor<0,errno==ENOENT { return nil }
    guard descriptor>=0 else {throw StudioError.invalid("Cannot read the configured Hugging Face credential file.")}
    let handle=FileHandle(fileDescriptor:descriptor,closeOnDealloc:true);defer{try?handle.close()}
    var before=stat(),after=stat()
    guard fstat(descriptor,&before)==0,before.st_mode&S_IFMT==S_IFREG,(1...16384).contains(before.st_size) else {
      throw StudioError.invalid("The configured Hugging Face credential must be a nonempty regular file of at most16KiB.")
    }
    let data=try handle.read(upToCount:16385) ?? Data()
    guard data.count==before.st_size,fstat(descriptor,&after)==0,before.st_size==after.st_size,
      before.st_mtimespec.tv_sec==after.st_mtimespec.tv_sec,before.st_mtimespec.tv_nsec==after.st_mtimespec.tv_nsec,
      before.st_ctimespec.tv_sec==after.st_ctimespec.tv_sec,before.st_ctimespec.tv_nsec==after.st_ctimespec.tv_nsec,
      let token=String(data:data,encoding:.utf8) else {
      throw StudioError.invalid("The configured Hugging Face credential changed while reading or is not UTF-8.")
    }
    return try normalized(token)
  }

  public static func isConfigured() throws -> Bool {
    var request = query
    request[kSecMatchLimit as String] = kSecMatchLimitOne
    request[kSecReturnAttributes as String] = true
    let status = SecItemCopyMatching(request as CFDictionary, nil)
    if status == errSecItemNotFound { return false }
    try check(status)
    return true
  }

  public static func read() throws -> String? {
    var request = query
    request[kSecMatchLimit as String] = kSecMatchLimitOne
    request[kSecReturnData as String] = true
    var result: CFTypeRef?
    let status = SecItemCopyMatching(request as CFDictionary, &result)
    if status == errSecItemNotFound { return nil }
    try check(status)
    guard let data = result as? Data, let token = String(data: data, encoding: .utf8) else {
      throw StudioError.invalid(
        "The saved model token could not be read. Remove it and save a new token.")
    }
    return try normalized(token)
  }

  public static func save(_ token: String) throws {
    let bytes = Data(try normalized(token).utf8)
    let status = SecItemUpdate(
      query as CFDictionary, [kSecValueData as String: bytes] as CFDictionary)
    if status == errSecItemNotFound {
      var item = query
      item[kSecValueData as String] = bytes
      item[kSecAttrLabel as String] = "WeeTodd Studio model downloads"
      item[kSecAttrAccessible as String] = kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly
      try check(SecItemAdd(item as CFDictionary, nil))
    } else {
      try check(status)
    }
  }

  public static func remove() throws {
    let status = SecItemDelete(query as CFDictionary)
    if status != errSecItemNotFound { try check(status) }
  }

  private static func check(_ status: OSStatus) throws {
    guard status == errSecSuccess else {
      throw StudioError.invalid(
        "macOS Keychain could not complete the model-token operation (\(status)). Unlock your Keychain or try again."
      )
    }
  }
}
