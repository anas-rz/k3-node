import os
import os.path as osp
import ssl
import sys
import urllib.request
from typing import Optional


def download_url(
    url: str,
    folder: str,
    log: bool = True,
    filename: Optional[str] = None,
) -> str:
    """Downloads the content of a URL to a specific folder."""
    if filename is None:
        filename = url.rpartition("/")[2]
        filename = filename if filename[0] == "?" else filename.split("?")[0]

    path = osp.join(folder, filename)
    if osp.exists(path):
        return path

    if log and "PYTEST_CURRENT_TEST" not in os.environ:
        print(f"Downloading {url}", file=sys.stderr)

    os.makedirs(folder, exist_ok=True)
    context = ssl._create_unverified_context()
    with urllib.request.urlopen(url, context=context) as response:
        with open(path, "wb") as f:
            while True:
                chunk = response.read(10 * 1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)

    return path


def download_google_url(
    id: str,
    folder: str,
    filename: str,
    log: bool = True,
) -> str:
    """Downloads the content of a Google Drive ID to a specific folder."""
    url = f"https://drive.usercontent.google.com/download?id={id}&confirm=t"
    return download_url(url, folder, log, filename)

