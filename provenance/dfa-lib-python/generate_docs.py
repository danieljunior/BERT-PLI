#!/usr/bin/env python3
"""Gerador simples de documentação estática para `dfa_lib_python`.

Gera HTML básico em `docs_site/` contendo docstrings de módulos, classes e funções.
Não modifica o código existente.
"""
import ast
import os
import html
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PKG_DIR = ROOT / 'dfa_lib_python'
OUT_DIR = ROOT / 'docs_site'

def extract_from_file(path: Path):
    src = path.read_text(encoding='utf-8')
    mod = ast.parse(src)
    module_doc = ast.get_docstring(mod) or ''
    classes = []
    functions = []
    for node in mod.body:
        if isinstance(node, ast.ClassDef):
            classes.append((node.name, ast.get_docstring(node) or ''))
        elif isinstance(node, ast.FunctionDef):
            functions.append((node.name, ast.get_docstring(node) or ''))
    return module_doc, classes, functions

def write_module_page(module_name: str, relpath: str, module_doc, classes, functions):
    title = f"Module {module_name}"
    html_parts = [f"<html><head><meta charset=\"utf-8\"><title>{html.escape(title)}</title></head><body>"]
    html_parts.append(f"<h1>{html.escape(module_name)}</h1>")
    if module_doc:
        html_parts.append(f"<div class='moddoc'><pre>{html.escape(module_doc)}</pre></div>")
    if classes:
        html_parts.append("<h2>Classes</h2>")
        for name, doc in classes:
            html_parts.append(f"<h3>{html.escape(name)}</h3>")
            html_parts.append(f"<pre>{html.escape(doc)}</pre>")
    if functions:
        html_parts.append("<h2>Functions</h2>")
        for name, doc in functions:
            html_parts.append(f"<h3>{html.escape(name)}</h3>")
            html_parts.append(f"<pre>{html.escape(doc)}</pre>")
    html_parts.append("<p><a href=\"index.html\">Voltar ao índice</a></p>")
    html_parts.append("</body></html>")
    out_path = OUT_DIR / relpath
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text('\n'.join(html_parts), encoding='utf-8')

def build_index(modules):
    parts = ["<html><head><meta charset='utf-8'><title>dfa_lib_python — Docs</title></head><body>"]
    parts.append("<h1>dfa_lib_python — Documentation</h1>")
    parts.append("<p>Generated simple docs for modules in <code>dfa_lib_python/</code>.</p>")
    parts.append("<ul>")
    for mod, rel in modules:
        parts.append(f"<li><a href=\"{html.escape(rel)}\">{html.escape(mod)}</a></li>")
    parts.append("</ul>")
    parts.append("</body></html>")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / 'index.html').write_text('\n'.join(parts), encoding='utf-8')

def find_py_modules(pkg_dir: Path):
    modules = []
    for p in sorted(pkg_dir.glob('*.py')):
        if p.name == '__init__.py':
            mod = pkg_dir.name
            rel = 'package.html'
        else:
            stem = p.stem
            mod = f"{pkg_dir.name}.{stem}"
            rel = f"{stem}.html"
        modules.append((p, mod, rel))
    return modules

def extract_package_init(pkg_dir: Path):
    init = pkg_dir / '__init__.py'
    if init.exists():
        return extract_from_file(init)
    return ('', [], [])

def main():
    modules = find_py_modules(PKG_DIR)
    catalog = []
    # package-level page from __init__
    pkg_doc, pkg_classes, pkg_funcs = extract_package_init(PKG_DIR)
    write_module_page(PKG_DIR.name, 'package.html', pkg_doc, pkg_classes, pkg_funcs)
    catalog.append((PKG_DIR.name, 'package.html'))

    for path, mod_name, rel in modules:
        if path.name == '__init__.py':
            continue
        module_doc, classes, functions = extract_from_file(path)
        write_module_page(mod_name, rel, module_doc, classes, functions)
        catalog.append((mod_name, rel))

    build_index(catalog)
    print(f"Documentation generated in: {OUT_DIR}")

if __name__ == '__main__':
    main()
