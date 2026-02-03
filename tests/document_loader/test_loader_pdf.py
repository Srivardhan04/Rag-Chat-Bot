import os
from pathlib import Path

from document_loader.loader import DirectoryLoader
from entities.document import Document


class DummyPage:
    def __init__(self, text: str):
        self._text = text

    def extract_text(self):
        return self._text


class DummyReader:
    def __init__(self, path):
        # ignore path, it's just a dummy
        self.pages = [DummyPage("Page one text"), DummyPage("Page two text")]


def test_directory_loader_handles_pdf(monkeypatch, tmp_path):
    # create a dummy pdf file to satisfy is_file checks
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%EOF")

    # monkeypatch PdfReader to return dummy pages
    # Patch PyPDF2.PdfReader so the local import in loader.load_file resolves to DummyReader
    monkeypatch.setattr("PyPDF2.PdfReader", DummyReader, raising=False)

    loader = DirectoryLoader(path=tmp_path, glob="*.pdf", recursive=False, show_progress=False)
    docs = loader.load()

    assert isinstance(docs, list)
    # Our DummyReader has 2 pages with text
    assert len(docs) == 2
    assert all(isinstance(d, Document) for d in docs)
    assert docs[0].metadata.get("document_type") == "pdf"
    assert docs[0].metadata.get("page") == 1
    assert "Page one text" in docs[0].page_content
