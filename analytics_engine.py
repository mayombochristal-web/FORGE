from github import Github
import os

def backup_memory(token,repo_name,folder):

    g=Github(token)

    repo=g.get_repo(repo_name)

    for root,dirs,files in os.walk(folder):

        for f in files:

            path=os.path.join(root,f)

            with open(path,"rb") as file:

                content=file.read()

            try:

                repo.create_file(
                f"{folder}/{f}",
                "oracle memory backup",
                content,
                branch="main"
                )

            except:
                pass