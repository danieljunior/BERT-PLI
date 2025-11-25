#!/usr/bin/env python3
"""Generate MkDocs Markdown pages for each module in dfa_lib_python.

This script creates `docs/` pages with `mkdocstrings` import blocks like:

    ::: dfa_lib_python.attribute

so MkDocs + mkdocstrings can render API docs.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PKG = ROOT / 'dfa_lib_python'
DOCS = ROOT / 'docs'

def module_name_from_path(p: Path):
    return f"dfa_lib_python.{p.stem}"

def write_index(modules):
    lines = ["# dfa_lib_python Documentation\n", "This site was generated automatically.\n", "## Modules\n"]
    for name, file in modules:
        lines.append(f"- [{name}]({file})\n")
    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / 'index.md').write_text('\n'.join(lines), encoding='utf-8')

def write_package_page(package_doc_exists: bool):
    DOCS.mkdir(parents=True, exist_ok=True)
    content = ["# Package dfa_lib_python\n"]
    content.append("::: dfa_lib_python\n")
    (DOCS / 'package.md').write_text('\n'.join(content), encoding='utf-8')

def write_module_page(module: str, filename: str):
    DOCS.mkdir(parents=True, exist_ok=True)
    content = [f"# {module}\n", f"::: {module}\n"]
    (DOCS / filename).write_text('\n'.join(content), encoding='utf-8')

def main():
    modules = []
    for p in sorted(PKG.glob('*.py')):
        if p.name == '__init__.py':
            continue
        mod = module_name_from_path(p)
        filename = f"{p.stem}.md"
        write_module_page(mod, filename)
        modules.append((mod, filename))

    # package page from __init__
    write_package_page((PKG / '__init__.py').exists())
    write_index(modules)
    print(f"MkDocs pages generated in: {DOCS}")

if __name__ == '__main__':
    main()
