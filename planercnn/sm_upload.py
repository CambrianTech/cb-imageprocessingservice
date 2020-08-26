import sagemaker
import tarfile
import os
from glob import glob

"""
model.tar.gz/
|- model.pth

sourcedir.tar.gz/
|- script.py
|- requirements.txt
"""


def main():
    """
    with tarfile.open("model.tar.gz", mode="w:gz") as archive:
        os.chdir("planercnn")
        archive.add("checkpoint", recursive=True)
        archive.add("anchors", recursive=True)
    """

    #os.chdir("..")

    with tarfile.open("sourcedir.tar.gz", mode="w:gz") as archive:
        os.chdir("planercnn")
        archive.add("requirements.txt")
        for script_path in glob("*.py"):
            archive.add(script_path)
        archive.add("models", recursive=True)
        archive.add("nms", recursive=True)
        archive.add("roialign", recursive=True)
        archive.add("datasets", recursive=True)

    os.chdir("..")

    sagemaker_session = sagemaker.Session()
    """
    sagemaker_session.upload_data(
        path="model.tar.gz", key_prefix="planercnn"
    )
    """
    sagemaker_session.upload_data(
        path="sourcedir.tar.gz", key_prefix="planercnn"
    )


if __name__ == "__main__":
    main()
