import glob
import gzip
import os
import os.path as osp
import shutil
import ssl
import sys
import tarfile
import urllib.request
import zipfile
from typing import List, Optional


def exists(path: str) -> bool:
    return osp.exists(path)


def isdir(path: str) -> bool:
    return osp.isdir(path)


def isfile(path: str) -> bool:
    return osp.isfile(path)


def is_url(url: str) -> bool:
    return url.startswith("http://") or url.startswith("https://")


def download_url(
    url: str,
    folder: str,
    filename: Optional[str] = None,
    log: bool = True,
) -> str:
    r"""Downloads the content of an URL to a specific folder."""
    if filename is None:
        filename = url.rpartition("/")[2].split("?")[0]

    os.makedirs(folder, exist_ok=True)
    out_path = osp.join(folder, filename)

    if osp.exists(out_path):
        return out_path

    if log and "PYTEST_CURRENT_TEST" not in os.environ:
        print(f"Downloading {url}", file=sys.stderr)

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
    )
    try:
        context = ssl.create_default_context()
        with urllib.request.urlopen(req, context=context) as response, open(out_path, "wb") as out_file:
            shutil.copyfileobj(response, out_file)
    except Exception:
        # Fallback to unverified context if SSL verification fails
        context = ssl._create_unverified_context()
        with urllib.request.urlopen(req, context=context) as response, open(out_path, "wb") as out_file:
            shutil.copyfileobj(response, out_file)

    return out_path


def extract_archive(path: str, dst: str):
    r"""Extracts an archive (zip, tar.gz, tgz, gz) to dst directory."""
    os.makedirs(dst, exist_ok=True)
    if path.endswith(".zip"):
        with zipfile.ZipFile(path, "r") as zf:
            zf.extractall(dst)
    elif path.endswith(".tar.gz") or path.endswith(".tgz"):
        with tarfile.open(path, "r:gz") as tf:
            tf.extractall(dst)
    elif path.endswith(".tar"):
        with tarfile.open(path, "r:") as tf:
            tf.extractall(dst)
    elif path.endswith(".gz"):
        out_name = osp.splitext(osp.basename(path))[0]
        out_file = osp.join(dst, out_name)
        with gzip.open(path, "rb") as f_in, open(out_file, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)


def cp(src: str, dst: str, extract: bool = False, log: bool = True):
    if is_url(src):
        # Determine destination folder and filename
        if dst.endswith("/") or osp.isdir(dst) or not osp.splitext(dst)[1]:
            folder = dst
            filename = src.rpartition("/")[2].split("?")[0]
        else:
            folder = osp.dirname(dst)
            filename = osp.basename(dst)

        local_path = download_url(src, folder, filename=filename, log=log)
        if extract:
            extract_archive(local_path, folder)
    else:
        if osp.isdir(src):
            shutil.copytree(src, dst)
        else:
            os.makedirs(osp.dirname(dst) if osp.splitext(dst)[1] else dst, exist_ok=True)
            shutil.copy(src, dst)
            local_path = osp.join(dst, osp.basename(src)) if osp.isdir(dst) else dst
            if extract:
                extract_archive(local_path, osp.dirname(local_path))


def rm(path: str):
    if osp.isdir(path):
        shutil.rmtree(path)
    elif osp.exists(path):
        os.remove(path)


def glob_files(pattern: str) -> List[str]:
    return sorted(glob.glob(pattern))
