import gzip
import shutil
import os
from memory_storage import MEM, COMP

def compress_file(src_path):
    fname = os.path.basename(src_path)
    dst_path = os.path.join(COMP, fname + ".gz")
    with open(src_path, "rb") as f_in:
        with gzip.open(dst_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    os.remove(src_path)
    return dst_path

def compress_old(keep=50):
    files = [f for f in os.listdir(MEM) if f.endswith(".json")]
    files.sort(reverse=True)
    if len(files) > keep:
        to_compress = files[keep:]
        for fname in to_compress:
            src = os.path.join(MEM, fname)
            compress_file(src)