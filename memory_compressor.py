import gzip
import shutil
import os

COMPRESSED_DIR = "oracle_memory/compressed"

def compress_memory(file_path):

    if not os.path.exists(COMPRESSED_DIR):
        os.makedirs(COMPRESSED_DIR)

    filename = os.path.basename(file_path)

    compressed_file = os.path.join(COMPRESSED_DIR, filename + ".gz")

    with open(file_path, 'rb') as f_in:
        with gzip.open(compressed_file, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    return compressed_file