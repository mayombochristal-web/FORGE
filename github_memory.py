import subprocess


def push_memory():

    subprocess.run(["git","add","oracle_memory/"])

    subprocess.run(["git","commit","-m","oracle memory update"])

    subprocess.run(["git","push"])