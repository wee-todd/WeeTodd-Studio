import CryptoKit
import Darwin
import Foundation

public struct NativeModelDownloadFile: Codable {
  public let repo: String
  public let revision: String
  public let filename: String
  public let size: Int64
  public let sha256: String
  public let target: String
  public let provider: String?

  var url: URL {
    let host = provider == "github" ? "https://raw.githubusercontent.com" : "https://huggingface.co"
    let separator = provider == "github" ? "/" : "/resolve/"
    return URL(string: host + "/" + repo + separator + revision + "/" + filename)!
  }
}

public struct NativeModelDownloadPackage: Codable {
  public var descriptor: ModelSetupDownload
  public let kind: String
  public let files: [NativeModelDownloadFile]
}

/// Downloads pinned packages whose released files are compatible with Swift.
/// Network bytes and checksum reads are streamed; no Python, conversion, or weights are loaded.
public enum NativeModelDownloads {
  public typealias Progress = @Sendable (String, Double) -> Void
  typealias Transfer = @Sendable (NativeModelDownloadFile, URL, String?, @escaping Progress) async throws -> Void
  public static var bundledCatalog: URL? {
    Bundle.main.resourceURL?.appendingPathComponent("RendererSource/src/wee_todd_mlx/model_download_catalog.json")
  }

  private static func relative(_ name: String) -> Bool {
    !name.isEmpty && !name.hasPrefix("/") && !name.contains("\\") && !name.utf8.contains(0)
      && name.split(separator: "/", omittingEmptySubsequences: false).allSatisfy { !$0.isEmpty && $0 != "." && $0 != ".." }
  }
  private static func hexadecimal(_ text: String, count: Int) -> Bool {
    text.utf8.count == count && text.utf8.allSatisfy { (48...57).contains($0) || (97...102).contains($0) }
  }
  public static func catalog(at url: URL) throws -> [NativeModelDownloadPackage] {
    let bytes = try bounded(url, limit: 8 * 1024 * 1024)
    let kinds: Set<String> = ["h3-qwen", "ltx25", "h3-video-vae", "h3-audio-vae", "h3-dt-tokenizer",
      "h3-direct-transformer", "h3-native-support", "h3-native-control", "ltx25-adapter"]
    var packages = try JSONDecoder().decode([NativeModelDownloadPackage].self, from: bytes)
      .filter { kinds.contains($0.kind) }
    // Direct Swift checkpoints have a different admission contract from the
    // optional Python packages. Keep their sources in a separate pinned catalog.
    if url.lastPathComponent == "model_download_catalog.json" {
      let native = url.deletingLastPathComponent().appendingPathComponent("native_model_download_catalog.json")
      if FileManager.default.fileExists(atPath: native.path) {
        packages += try JSONDecoder().decode([NativeModelDownloadPackage].self,
          from: bounded(native, limit: 8 * 1024 * 1024))
      }
    }
    guard packages.count <= 64, Set(packages.map { $0.descriptor.id }).count == packages.count else {
      throw StudioError.invalid("Invalid native model download catalog.")
    }
    for package in packages {
      guard relative(package.descriptor.id), !package.descriptor.id.contains("/"),
        kinds.contains(package.kind), !package.files.isEmpty, package.files.count <= 512,
        Set(package.files.map(\.target)).count == package.files.count else {
        throw StudioError.invalid("Invalid native model package.")
      }
      var total: Int64 = 0
      for file in package.files {
        guard relative(file.filename), relative(file.target), hexadecimal(file.revision, count: 40),
          hexadecimal(file.sha256, count: 64), file.size > 0, file.size <= 256 * 1_073_741_824,
          [nil, "huggingface", "github"].contains(file.provider),
          file.repo.range(of: #"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$"#, options: .regularExpression) != nil,
          let url = URL(string: file.url.absoluteString), url.scheme == "https",
          ["huggingface.co", "raw.githubusercontent.com"].contains(url.host ?? "") else {
          throw StudioError.invalid("Invalid pinned native model source.")
        }
        total += file.size
      }
      guard total <= 512 * 1_073_741_824, total == package.descriptor.downloadBytes else {
        throw StudioError.invalid("Native model package size differs from its catalog.")
      }
    }
    for index in packages.indices where packages[index].kind == "h3-qwen" {
      // The pinned vision-capable Qwen package supplies both setup fields.
      if packages[index].descriptor.components?.contains("text_encoder") == true,
        packages[index].descriptor.components?.contains("vision_encoder") == false {
        packages[index].descriptor.components?.append("vision_encoder")
      }
    }
    for index in packages.indices where packages[index].kind == "h3-video-vae" {
      packages[index].descriptor.description = packages[index].descriptor.description.replacingOccurrences(
        of: "Audio VAE is included in the separate task support files.",
        with: "Choose the separate folded-weight audio VAE download for Swift inference.")
    }
    return packages
  }

  private static func bounded(_ url: URL, limit: Int) throws -> Data {
    let descriptor = Darwin.open(url.path, O_RDONLY | O_CLOEXEC | O_NONBLOCK | O_NOFOLLOW)
    guard descriptor >= 0 else { throw StudioError.invalid("The model catalog is unavailable.") }
    let handle = FileHandle(fileDescriptor: descriptor, closeOnDealloc: true)
    defer { try? handle.close() }
    var status = stat()
    guard fstat(descriptor, &status) == 0, status.st_mode & S_IFMT == S_IFREG,
      status.st_size > 0, status.st_size <= limit else {
      throw StudioError.invalid("Invalid bounded model catalog.")
    }
    let data = try handle.read(upToCount: limit + 1) ?? Data()
    guard data.count == status.st_size else { throw StudioError.invalid("Model catalog changed while reading.") }
    return data
  }

  static func verified(_ url: URL, file: NativeModelDownloadFile) throws -> Bool {
    let canonical = url.resolvingSymlinksInPath()
    let descriptor = Darwin.open(canonical.path, O_RDONLY | O_CLOEXEC | O_NONBLOCK | O_NOFOLLOW)
    guard descriptor >= 0 else { return false }
    let handle = FileHandle(fileDescriptor: descriptor, closeOnDealloc: true)
    defer { try? handle.close() }
    var before = stat(), after = stat()
    guard fstat(descriptor, &before) == 0, before.st_mode & S_IFMT == S_IFREG,
      before.st_size == file.size else { return false }
    var hash = SHA256()
    while let bytes = try handle.read(upToCount: 1024 * 1024), !bytes.isEmpty {
      try Task.checkCancellation()
      hash.update(data: bytes)
    }
    guard fstat(descriptor, &after) == 0, before.st_size == after.st_size,
      before.st_mtimespec.tv_sec == after.st_mtimespec.tv_sec,
      before.st_mtimespec.tv_nsec == after.st_mtimespec.tv_nsec,
      before.st_ctimespec.tv_sec == after.st_ctimespec.tv_sec,
      before.st_ctimespec.tv_nsec == after.st_ctimespec.tv_nsec else { return false }
    return hash.finalize().map { String(format: "%02x", $0) }.joined() == file.sha256
  }

  private static func existingFiles(roots: [String], sizes: Set<Int64>) throws -> [Int64: [URL]] {
    var result: [Int64: [URL]] = [:], seen: Set<String> = []
    var pending = roots.map { URL(fileURLWithPath: $0) }, count = 0
    while let raw = pending.popLast() {
      try Task.checkCancellation()
      count += 1
      guard count <= 20_000 else { break }
      let url = raw.standardizedFileURL.resolvingSymlinksInPath()
      guard seen.insert(url.path).inserted else { continue }
      let values = try? url.resourceValues(forKeys: [.isDirectoryKey, .isRegularFileKey, .fileSizeKey])
      if values?.isDirectory == true {
        let children = (try? FileManager.default.contentsOfDirectory(at: url,
          includingPropertiesForKeys: nil, options: [.skipsHiddenFiles])) ?? []
        pending += children.filter { ![".git", ".cache"].contains($0.lastPathComponent) }
      } else if values?.isRegularFile == true, let size = values?.fileSize, sizes.contains(Int64(size)) {
        result[Int64(size), default: []].append(url)
      }
    }
    return result
  }

  public static func prepare(id: String, catalog: URL, destination: URL,
    existingRoots: [String], token: String? = nil, progress: @escaping Progress) async throws -> URL {
    try await prepare(id: id, catalog: catalog, destination: destination, existingRoots: existingRoots,
      token: token, progress: progress, transfer: { file, partial, token, progress in
        try await NativeModelHTTPTransfer(file: file, partial: partial, token: token, progress: progress).run()
      })
  }

  static func prepare(id: String, catalog: URL, destination: URL,
    existingRoots: [String], token: String?, progress: @escaping Progress,
    transfer: @escaping Transfer) async throws -> URL {
    guard let package = try self.catalog(at: catalog).first(where: { $0.descriptor.id == id }) else {
      throw StudioError.invalid("Choose a pinned native download package.")
    }
    if let token { _ = try ModelDownloadToken.normalized(token) }
    let fm = FileManager.default, root = destination.standardizedFileURL.resolvingSymlinksInPath()
    guard root.path.hasPrefix("/"), !root.path.utf8.contains(0) else { throw StudioError.invalid("Choose a model library.") }
    try fm.createDirectory(at: root, withIntermediateDirectories: true)
    let final = root.appendingPathComponent(id), lock = root.appendingPathComponent("." + id + ".lock")
    guard !fm.fileExists(atPath: final.path) else { throw StudioError.invalid("Model folder already exists. Scan it or choose another library.") }
    let descriptor = Darwin.open(lock.path, O_CREAT | O_EXCL | O_WRONLY | O_CLOEXEC, 0o600)
    guard descriptor >= 0 else { throw StudioError.invalid("Another setup owns this destination. Check its .lock file before retrying.") }
    Darwin.close(descriptor)
    defer { try? fm.removeItem(at: lock) }
    let cache = root.appendingPathComponent(".weetodd-downloads/" + id)
    try fm.createDirectory(at: cache, withIntermediateDirectories: true)
    let candidates = try existingFiles(roots: existingRoots, sizes: Set(package.files.map(\.size)))
    let staging = root.appendingPathComponent(".prepare-" + UUID().uuidString)
    try fm.createDirectory(at: staging, withIntermediateDirectories: false)
    defer { try? fm.removeItem(at: staging) }
    var done: Int64 = 0
    var verifiedSources: [String: URL] = [:]
    for file in package.files {
      try Task.checkCancellation()
      let cached = cache.appendingPathComponent(file.target)
      let partial = cached.appendingPathExtension("partial")
      try fm.createDirectory(at: cached.deletingLastPathComponent(), withIntermediateDirectories: true)
      progress("Verifying " + file.target, 0.95 * Double(done) / Double(package.descriptor.downloadBytes))
      let identity = "\(file.size):\(file.sha256)"
      var source = verifiedSources[identity]
      if source == nil, try verified(cached, file: file) { source = cached }
      if source == nil {
        for candidate in candidates[file.size, default: []] {
          if try verified(candidate, file: file) { source = candidate; break }
        }
      }
      if source == nil {
        let used = (try? fm.attributesOfItem(atPath: partial.path)[.size] as? NSNumber)?.int64Value ?? 0
        let remaining = file.size - max(0, min(used, file.size))
        let space = try fm.attributesOfFileSystem(forPath: root.path)[.systemFreeSize] as? NSNumber
        guard let space, space.int64Value >= remaining + 256 * 1024 * 1024 else {
          throw StudioError.invalid("The model library needs more free disk space.")
        }
        let base = done, total = package.descriptor.downloadBytes
        try await transfer(file, partial, token, { message, fraction in
          progress(message, 0.95 * (Double(base) + Double(file.size) * fraction) / Double(total))
        })
        guard try verified(partial, file: file) else {
          // A complete corrupt download cannot be resumed; incomplete network bytes remain reusable.
          if (try? fm.attributesOfItem(atPath: partial.path)[.size] as? NSNumber)?.int64Value == file.size {
            try? fm.removeItem(at: partial)
          }
          throw StudioError.invalid("Downloaded model checksum differs from its pinned source: " + file.target)
        }
        if fm.fileExists(atPath: cached.path) { try fm.removeItem(at: cached) }
        try fm.moveItem(at: partial, to: cached); source = cached
      }
      verifiedSources[identity] = source
      let output = staging.appendingPathComponent(file.target)
      try fm.createDirectory(at: output.deletingLastPathComponent(), withIntermediateDirectories: true)
      do { try fm.linkItem(at: source!, to: output) }
      catch {
        guard (error as NSError).domain == NSPOSIXErrorDomain && (error as NSError).code == Int(EXDEV)
          || (error as NSError).underlyingPOSIXCode == Int(EXDEV) else { throw error }
        try fm.createSymbolicLink(at: output, withDestinationURL: source!)
      }
      done += file.size
    }
    try Task.checkCancellation()
    let provenance: [String: Any] = ["format": "weetodd-model-setup-v1", "id": id,
      "converter": "preconverted", "nativeRuntime": "swift", "include_vision": package.kind == "h3-qwen",
      "sources": try JSONSerialization.jsonObject(with: JSONEncoder().encode(package.files))]
    try JSONSerialization.data(withJSONObject: provenance, options: [.prettyPrinted, .sortedKeys])
      .write(to: staging.appendingPathComponent("setup_provenance.json"), options: .withoutOverwriting)
    guard renamex_np(staging.path, final.path, UInt32(RENAME_EXCL)) == 0 else {
      throw StudioError.invalid("Model destination appeared during setup; it was preserved.")
    }
    progress("Verified model package installed. Scan its components to create a recipe.", 1)
    return final
  }
}

private extension NSError {
  var underlyingPOSIXCode: Int? { (userInfo[NSUnderlyingErrorKey] as? NSError).flatMap {
    $0.domain == NSPOSIXErrorDomain ? $0.code : nil
  } }
}

/// URLSession writes incremental data directly to the persistent partial file.
/// Range admission and final SHA verification prevent mixing mismatched transfers.
final class NativeModelHTTPTransfer: NSObject, URLSessionDataDelegate, @unchecked Sendable {
  let file: NativeModelDownloadFile, partial: URL, token: String?, progress: NativeModelDownloads.Progress
  let configuration: URLSessionConfiguration
  private let lock = NSLock()
  private var task: URLSessionDataTask?, session: URLSession?, handle: FileHandle?
  private var continuation: CheckedContinuation<Void, Error>?
  private var failure: Error?, cancelled = false, received: Int64 = 0
  private var lastProgress = Date.distantPast
  init(file: NativeModelDownloadFile, partial: URL, token: String?,
    configuration: URLSessionConfiguration = .ephemeral, progress: @escaping NativeModelDownloads.Progress) {
    self.file = file; self.partial = partial; self.token = token; self.progress = progress
    self.configuration = configuration
  }
  func run() async throws {
    try Task.checkCancellation()
    try await withTaskCancellationHandler {
      try await withCheckedThrowingContinuation { continuation in
        do {
          let descriptor = Darwin.open(partial.path, O_CREAT | O_RDWR | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK, 0o600)
          guard descriptor >= 0 else { throw StudioError.invalid("Cannot open the partial model download.") }
          let handle = FileHandle(fileDescriptor: descriptor, closeOnDealloc: true)
          var status = stat()
          guard fstat(descriptor, &status) == 0, status.st_mode & S_IFMT == S_IFREG else {
            try? handle.close(); throw StudioError.invalid("Invalid partial model download.")
          }
          received = status.st_size
          if received > file.size { try handle.truncate(atOffset: 0); received = 0 }
          try handle.seek(toOffset: UInt64(received)); self.handle = handle
          if received == file.size { try handle.close(); continuation.resume(); return }
          var request = URLRequest(url: file.url)
          request.timeoutInterval = 120
          request.setValue("identity", forHTTPHeaderField: "Accept-Encoding")
          if received > 0 { request.setValue("bytes=\(received)-", forHTTPHeaderField: "Range") }
          if let token, file.url.host == "huggingface.co" { request.setValue("Bearer " + token, forHTTPHeaderField: "Authorization") }
          configuration.urlCache = nil; configuration.httpCookieStorage = nil
          let queue = OperationQueue(); queue.maxConcurrentOperationCount = 1
          let session = URLSession(configuration: configuration, delegate: self, delegateQueue: queue)
          let task = session.dataTask(with: request)
          lock.lock(); self.continuation = continuation; self.session = session; self.task = task
          let wasCancelled = cancelled; lock.unlock()
          task.resume(); if wasCancelled { task.cancel() }
        } catch { continuation.resume(throwing: error) }
      }
    } onCancel: {
      self.lock.lock(); self.cancelled = true; let task = self.task; self.lock.unlock()
      task?.cancel()
    }
  }
  func urlSession(_ session: URLSession, task: URLSessionTask,
    willPerformHTTPRedirection response: HTTPURLResponse, newRequest request: URLRequest,
    completionHandler: @escaping (URLRequest?) -> Void) {
    guard request.url?.scheme == "https" else { completionHandler(nil); return }
    var redirect = request
    if request.url?.host != "huggingface.co" { redirect.setValue(nil, forHTTPHeaderField: "Authorization") }
    completionHandler(redirect)
  }
  func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive response: URLResponse,
    completionHandler: @escaping (URLSession.ResponseDisposition) -> Void) {
    do {
      guard let http = response as? HTTPURLResponse else { throw StudioError.invalid("Invalid model download response.") }
      if http.statusCode == 200 {
        guard response.expectedContentLength < 0 || response.expectedContentLength == file.size else {
          throw StudioError.invalid("Model response size differs from its pinned source.")
        }
        try handle?.truncate(atOffset: 0); try handle?.seek(toOffset: 0); received = 0
      } else if http.statusCode == 206 {
        guard http.value(forHTTPHeaderField: "Content-Range") == "bytes \(received)-\(file.size - 1)/\(file.size)" else {
          throw StudioError.invalid("Model server returned a mismatched resume range.")
        }
      } else if [401, 403].contains(http.statusCode) {
        throw StudioError.invalid("Model access was denied. Review source access and the saved Hugging Face read token.")
      } else { throw StudioError.invalid("Model server returned HTTP \(http.statusCode). Retry to resume.") }
      guard response.expectedContentLength < 0 || response.expectedContentLength == file.size - received else {
        throw StudioError.invalid("Model response size differs from its pinned source.")
      }
      completionHandler(.allow)
    } catch { failure = error; completionHandler(.cancel) }
  }
  func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive data: Data) {
    guard failure == nil else { return }
    do {
      guard Int64(data.count) <= file.size - received else { throw StudioError.invalid("Model response exceeds its pinned size.") }
      try handle?.write(contentsOf: data); received += Int64(data.count)
      let now = Date()
      if received == file.size || now.timeIntervalSince(lastProgress) >= 0.1 {
        lastProgress = now
        progress("Downloading " + file.target, Double(received) / Double(file.size))
      }
    } catch { failure = error; dataTask.cancel() }
  }
  func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
    try? handle?.synchronize(); try? handle?.close(); handle = nil
    lock.lock(); let continuation = self.continuation; self.continuation = nil
    let cancelled = self.cancelled; lock.unlock()
    session.finishTasksAndInvalidate()
    if let failure { continuation?.resume(throwing: failure) }
    else if cancelled { continuation?.resume(throwing: CancellationError()) }
    else if error != nil { continuation?.resume(throwing: StudioError.invalid("Model transfer interrupted. Retry to resume its verified partial file.")) }
    else { continuation?.resume() }
  }
}
