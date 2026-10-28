"""Shrink a .docx's page margins in place.

macOS `textutil` writes an empty <w:sectPr/>, so Word falls back to 1in margins
and our one-pager spills onto a second page. This drops it to 0.75in top/bottom
and 0.8in sides, which is enough.

    textutil -convert docx docs/checkpoint_onepager.html -output out.docx
    .venv/bin/python scripts/set_docx_margins.py out.docx
"""

import shutil
import sys
import zipfile

MARGINS = (
    "<w:sectPr>"
    '<w:pgSz w:w="12240" w:h="15840"/>'  # US Letter
    '<w:pgMar w:top="1080" w:right="1170" w:bottom="1080" w:left="1170" '
    'w:header="720" w:footer="720" w:gutter="0"/>'
    "</w:sectPr>"
)


def set_margins(path):
    with zipfile.ZipFile(path) as src:
        items = {name: src.read(name) for name in src.namelist()}

    doc = items["word/document.xml"].decode()
    if "w:pgMar" in doc:
        return False
    if "<w:sectPr></w:sectPr>" not in doc:
        raise SystemExit(f"{path}: no empty sectPr to replace; was this made by textutil?")
    items["word/document.xml"] = doc.replace("<w:sectPr></w:sectPr>", MARGINS).encode()

    tmp = f"{path}.tmp"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as out:
        for name, data in items.items():
            out.writestr(name, data)
    shutil.move(tmp, path)
    return True


def demo():
    """Self-check: round-trip a minimal textutil-shaped docx."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "t.docx"
        with zipfile.ZipFile(path, "w") as z:
            z.writestr("word/document.xml", "<w:document><w:body><w:sectPr></w:sectPr></w:body></w:document>")
            z.writestr("[Content_Types].xml", "<Types/>")
        assert set_margins(path) is True
        doc = zipfile.ZipFile(path).read("word/document.xml").decode()
        assert 'w:top="1080"' in doc, doc
        assert zipfile.ZipFile(path).read("[Content_Types].xml") == b"<Types/>"
        assert set_margins(path) is False  # idempotent
    print("ok")


if __name__ == "__main__":
    if sys.argv[1:2] == ["--demo"]:
        demo()
    elif len(sys.argv) == 2:
        print("margins set" if set_margins(sys.argv[1]) else "margins already set")
    else:
        raise SystemExit(__doc__)
