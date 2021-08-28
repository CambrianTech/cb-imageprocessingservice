import os
import json
from pathlib import Path
from pipeline.output import PipelineOutput
import cv2

class PipelineFileOutput(PipelineOutput):
    def __init__(self, base_path, api_level):
        super().__init__(base_path, api_level)

    def save_image(self, image, filename, url, quality=90):
        path = Path(os.path.join(self.base_path, self.unique_id, url))
        if len(path.parents) > 0:
            path.parents[0].mkdir(parents=True, exist_ok=True)

        if len(image.shape) == 3:
            cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        else:
            cv2.imwrite(str(path), image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])

    def save_file(self, data, filename, url):
        path = Path(os.path.join(self.base_path, self.unique_id, url))
        if len(path.parents) > 0:
            path.parents[0].mkdir(parents=True, exist_ok=True)

        with open(str(path), "wb") as out_file:
            out_file.write(bytearray(data))

    def save_data(self, data, filename, url):
        path = Path(os.path.join(self.base_path, self.unique_id, url))
        if len(path.parents) > 0:
            path.parents[0].mkdir(parents=True, exist_ok=True)

        with open(str(path), "w", encoding="utf-8") as out_file:
            json.dump(data, out_file, indent=4)
