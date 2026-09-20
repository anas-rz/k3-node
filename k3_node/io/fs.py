import glob
import os
import os.path as osp
import shutil
from typing import List


def exists(path: str) -> bool:
    return osp.exists(path)


def isdir(path: str) -> bool:
    return osp.isdir(path)


def isfile(path: str) -> bool:
    return osp.isfile(path)


def cp(src: str, dst: str):
    if osp.isdir(src):
        shutil.copytree(src, dst)
    else:
        shutil.copy(src, dst)


def rm(path: str):
    if osp.isdir(path):
        shutil.rmtree(path)
    elif osp.exists(path):
        os.remove(path)


def glob_files(pattern: str) -> List[str]:
    return sorted(glob.glob(pattern))

