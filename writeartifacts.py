import argparse
import os

imagedefs_text = """[
    {
        "name": "ImageProcessingContainer",
        "imageUri": "%s"
    },
    {
        "name": "PlanesContainer",
        "imageUri": "%s"
    }
]
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out_path")
    parser.add_argument("main_repo_uri")
    parser.add_argument("planes_repo_uri")
    args = parser.parse_args()

    print("Writing to", args.out_path, "main image repo", args.main_repo_uri, "planes image repo", args.planes_repo_uri)
    with open(args.out_path, "w") as out_file:
        out_file.write(imagedefs_text % (args.main_repo_uri, args.planes_repo_uri))


if __name__ == "__main__":
    main()
