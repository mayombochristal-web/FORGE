#!/usr/bin/env python3
from utils import download_corpus
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='shakespeare')
    args = parser.parse_args()
    path = download_corpus(args.name)
    print(f"Corpus téléchargé : {path}")