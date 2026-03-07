import subprocess
import os

def push_memory():
    """Ajoute, commit et push le dossier oracle_memory sur GitHub"""
    try:
        subprocess.run(["git", "add", "oracle_memory/"], check=True)
        subprocess.run(["git", "commit", "-m", "ORACLE memory update"], check=True)
        subprocess.run(["git", "push"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Git error: {e}")
        raise