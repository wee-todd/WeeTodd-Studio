import AppKit
import StudioCore
import SwiftUI

struct ProductionLibraryView: View {
  @EnvironmentObject var store: StudioStore
  @Environment(\.dismiss) private var dismiss
  @State private var query = ""
  @State private var packages: [ProductionLibraryPackage] = []
  @State private var selection: UUID?
  @State private var message = ""
  @State private var error: String?
  @State private var library: ProductionLibrary?

  var body: some View {
    VStack(alignment: .leading, spacing: 12) {
      HStack { Text("Production Library").font(.title2); Spacer(); Button("Done") { dismiss() } }
      Text("Global definitions · Import a pinned version into this movie, then edit locally. Media files stay in their current locations.")
        .font(.caption).foregroundStyle(.secondary)
      HStack {
        TextField("Search names, types and tags", text: $query).textFieldStyle(.roundedBorder)
        Button("Search") { refresh() }.keyboardShortcut(.return)
        Menu("Publish from this movie") {
          ForEach(store.planning.subjects.sorted { $0.name < $1.name }) { subject in
            Button("\(subject.name) · \(subject.kind.label)") { perform {
              let package = try catalog().publish(rootID: subject.id, planning: store.planning, assets: store.allAssets)
              selection = package.id; message = "Published \(subject.name), version \(package.version)."; refresh()
            } }
          }
        }.disabled(store.planning.subjects.isEmpty)
      }
      HSplitView {
        List(packages, selection: $selection) { package in
          VStack(alignment: .leading) {
            Text(package.root?.name ?? "Object").font(.headline)
            Text("\(package.root?.kind.label ?? "") · v\(package.version) · \(package.subjects.count) objects")
              .font(.caption).foregroundStyle(.secondary)
          }.tag(package.id)
        }.frame(minWidth: 260)
        if let selected = packages.first(where: { $0.id == selection }) {
          ScrollView {
            VStack(alignment: .leading, spacing: 12) {
              Text(selected.root?.name ?? "Package").font(.title2)
              HStack {
                Button("Use in movie") { perform {
                  var movie = store.project; try movie.importLibraryPackage(selected)
                  store.change { $0 = movie }; message = "Added pinned version \(selected.version) to this movie’s production objects."
                } }
                Button("Export package…") { export(selected) }
              }
              if !selected.missingMediaPaths.isEmpty {
                Text("\(selected.missingMediaPaths.count) linked image files are unavailable. Relink them in the movie before generation.")
                  .foregroundStyle(.orange).font(.caption)
              }
              ForEach(PlanningSubjectKind.reviewOrder) { kind in
                let objects = selected.subjects.filter { $0.kind == kind }
                if !objects.isEmpty {
                  DisclosureGroup("\(kind.groupTitle) (\(objects.count))") {
                    ForEach(objects) { object in
                      VStack(alignment: .leading, spacing: 4) {
                        Text(object.name).font(.headline)
                        Text(object.details).textSelection(.enabled)
                        Text((object.tags ?? []).joined(separator: ", ")).font(.caption).foregroundStyle(.secondary)
                        Text(packageApproved(object, in: selected) ? "Description approved" : "Draft · review in movie")
                          .font(.caption).foregroundStyle(.secondary)
                      }.frame(maxWidth: .infinity, alignment: .leading).padding(.vertical, 6)
                    }
                  }
                }
              }
            }.padding(14)
          }.frame(minWidth: 440)
        } else { Text("Choose a library package, or publish a movie object with its linked dependencies.").foregroundStyle(.secondary).padding(30).frame(maxWidth: .infinity) }
      }
      HStack {
        Button("Import package…") { importPackage() }
        Text(message).font(.caption).foregroundStyle(.secondary)
      }
      if let error { Text(error).foregroundStyle(.red).textSelection(.enabled) }
    }.padding(20).frame(width: 920, height: 680).onAppear { refresh() }
  }
  private func packageApproved(_ object: PlanningSubject, in package: ProductionLibraryPackage) -> Bool {
    var plan = ProjectPlanning(); plan.subjects = package.subjects
    return plan.isSubjectApproved(object.id)
  }
  private func catalog() throws -> ProductionLibrary {
    if let library { return library }
    let opened = try ProductionLibrary(url: StudioStore.supportDirectory.appendingPathComponent("ProductionLibrary/catalog.sqlite"))
    library = opened; return opened
  }
  private func refresh() { perform { packages = try catalog().search(query) } }
  private func perform(_ work: () throws -> Void) { do { try work(); error = nil } catch { self.error = error.localizedDescription } }
  private func export(_ package: ProductionLibraryPackage) {
    let panel = NSSavePanel(); panel.allowedContentTypes = [.json]; panel.nameFieldStringValue = "production-library.json"
    if panel.runModal() == .OK, let url = panel.url { perform {
      let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
      try encoder.encode(package).write(to: url, options: .atomic)
      message = "Exported metadata and linked paths. Use Collect Media on the movie for portable image files."
    } }
  }
  private func importPackage() {
    let panel = NSOpenPanel(); panel.allowedContentTypes = [.json]
    if panel.runModal() == .OK, let url = panel.url { perform {
      let size = try url.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? 0
      guard size <= 2_000_000 else { throw StudioError.invalid("Library packages are limited to 2 MB of metadata.") }
      let package = try JSONDecoder().decode(ProductionLibraryPackage.self, from: Data(contentsOf: url))
      var movie = store.project; try movie.importLibraryPackage(package)
      store.change { $0 = movie }; message = "Imported pinned package into this movie."
    } }
  }
}
