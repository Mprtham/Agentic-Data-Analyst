"""Streamlit utility functions."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path


def download_dir_as_zip(directory: Path) -> bytes:
    """Create a ZIP file from a directory and return as bytes."""
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for file_path in directory.rglob("*"):
            if file_path.is_file():
                arcname = file_path.relative_to(directory.parent)
                zip_file.write(file_path, arcname=arcname)

    zip_buffer.seek(0)
    return zip_buffer.getvalue()
