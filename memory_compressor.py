import gzip
import shutil
import os

def compress(path):

    out=path+".gz"

    with open(path,"rb") as f_in:

        with gzip.open(out,"wb") as f_out:

            shutil.copyfileobj(f_in,f_out)

    return out