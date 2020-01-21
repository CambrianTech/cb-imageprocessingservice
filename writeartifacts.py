import argparse
import os

imagedefs_text = """[
    {
        "name": "ImageProcessingContainer",
        "imageUri": "%s"
    }
]
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out_path")
    parser.add_argument("repo_uri")
    args = parser.parse_args()
    print("Writing to", args.out_path, "image repo", args.repo_uri)
    with open(args.out_path, "w") as out_file:
        out_file.write(imagedefs_text % args.repo_uri)


if __name__ == "__main__":
    main()
