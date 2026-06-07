#!/usr/bin/env python3
"""
validate-posts.py
Proverava sve Jekyll postove u _posts/ na česte greške:
- Nepotpun frontmatter (nedostaje title ili layout)
- Pokvarene reference na lokalne slike (/images/...)
- Broj postova sa published: false
- (Opciono) osnovne stvari oko imena fajlova

Pokretanje:
    python3 scripts/validate-posts.py

Izlaz: lista problema + sažetak.
Možeš ga dodati u CI kasnije.
"""

import os
import glob
import re
import sys
from typing import List, Tuple

POSTS_GLOB = "_posts/*.[mM][dD]*"


def parse_frontmatter(content: str):
    """Vrati (frontmatter_dict ili None, error_str ili None)."""
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n", content, re.DOTALL)
    if not m:
        return None, "no frontmatter block"
    try:
        import yaml
        fm = yaml.safe_load(m.group(1)) or {}
        return fm, None
    except Exception as e:
        return None, f"YAML parse error: {e}"


def find_local_image_refs(content: str) -> List[str]:
    refs = []
    for mm in re.finditer(r"!\[[^\]]*\]\(([^)]+)\)", content):
        ref = mm.group(1).strip()
        if ref.startswith(("/images/", "images/")):
            refs.append(ref.lstrip("/"))
    return refs


def main():
    posts = sorted(glob.glob(POSTS_GLOB))
    if not posts:
        print("Nema postova u _posts/")
        return 1

    issues: List[Tuple[str, str]] = []
    published_false: List[str] = []
    total_images_checked = 0

    for path in posts:
        name = os.path.basename(path)
        with open(path, encoding="utf-8", errors="ignore") as f:
            content = f.read()

        fm, err = parse_frontmatter(content)
        if err:
            issues.append((name, err))
            continue

        if not fm.get("title"):
            issues.append((name, "missing title"))
        if not fm.get("layout"):
            issues.append((name, "missing layout"))

        if fm.get("published") is False:
            published_false.append(name)

        for ref in find_local_image_refs(content):
            total_images_checked += 1
            if not os.path.exists(ref):
                issues.append((name, f"missing image: /{ref}"))

    # Izveštaj
    print(f"Provereno postova: {len(posts)}")
    print(f"Provereno lokalnih slika: {total_images_checked}")
    print(f"Ukupno problema: {len(issues)}")
    print(f"Postovi sa published: false: {len(published_false)}")
    print()

    if issues:
        print("=== PROBLEMI ===")
        for name, msg in issues:
            print(f"  {name}: {msg}")
        print()

    if published_false:
        print("=== published: false (skriveni/draftovi) ===")
        for n in published_false:
            print(f"  {n}")
        print()

    if not issues:
        print("✅ Nema pronađenih problema sa frontmatter-om ili slikama.")
    else:
        print("⚠️  Postoje problemi koje treba ispraviti (vidi iznad).")

    # Exit code za CI
    return 1 if issues else 0


if __name__ == "__main__":
    sys.exit(main())
