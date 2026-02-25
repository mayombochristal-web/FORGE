#!/usr/bin/env python3
import argparse
from utils import download_corpus

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='shakespeare',
                        choices=['shakespeare', 'wikitext', 'tiny_shakespeare'])
    args = parser.parse_args()
    path = download_corpus(args.name)
    print(f"Corpus téléchargé : {path}")