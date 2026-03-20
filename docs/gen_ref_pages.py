"""Auto-generate API reference pages from the versatil package."""

from pathlib import Path

import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()

src = Path("src")
package = src / "versatil"

for path in sorted(package.rglob("*.py")):
    module_path = path.relative_to(src)
    doc_path = path.relative_to(src).with_suffix(".md")
    full_doc_path = Path("reference", doc_path)
    parts = tuple(module_path.with_suffix("").parts)
    if any(part.startswith("_") for part in parts):
        continue
    if "endpoints" in parts:
        continue
    nav[parts] = doc_path.as_posix()
    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        identifier = ".".join(parts)
        fd.write(f"::: {identifier}")
    mkdocs_gen_files.set_edit_path(full_doc_path, path.as_posix())

with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
