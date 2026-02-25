#!/usr/bin/env python3
"""
Télécharge des corpus texte publics pour entraîner le modèle TTU.
"""

import os
import requests
import argparse

def download_file(url, dest_path):
    print(f"Téléchargement de {url}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(dest_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"Sauvegardé dans {dest_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='shakespeare',
                        choices=['shakespeare', 'wikitext', 'tiny_shakespeare'])
    args = parser.parse_args()

    os.makedirs('data', exist_ok=True)

    urls = {
        'shakespeare': 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt',
        'tiny_shakespeare': 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt',
        'wikitext': 'https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/train.txt'
    }

    url = urls.get(args.name, urls['shakespeare'])
    dest = os.path.join('data', f"{args.name}.txt")
    download_file(url, dest)

if __name__ == '__main__':
    main()