import cv2
import os
import json
from pathlib import Path
from .output import PipelineOutput

class PipelineFileOutput(PipelineOutput):

    def save_image(self, image, filename, url, quality=90):
        path = Path(os.path.join(self.config.dest_path, self.unique_id, url))
        if len(path.parents) > 0:
            path.parents[0].mkdir(parents=True, exist_ok=True)

        print("Writing image %s with shape" % filename, image.shape)

        if len(image.shape) == 2:
            _image = image
        elif image.shape[2] == 3:
            _image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        elif image.shape[2] == 4:
            _image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
            
        cv2.imwrite(str(path), _image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])

    def save_file(self, data, filename, url):
        path = Path(os.path.join(self.config.dest_path, self.unique_id, url))
        if len(path.parents) > 0:
            path.parents[0].mkdir(parents=True, exist_ok=True)

        with open(str(path), "wb") as out_file:
            out_file.write(bytearray(data))

    def save_data(self, data, filename, url):
        path = Path(os.path.join(self.config.dest_path, self.unique_id, url))
        if len(path.parents) > 0:
            path.parents[0].mkdir(parents=True, exist_ok=True)

        with open(str(path), "w", encoding="utf-8") as out_file:
            json.dump(data, out_file, indent=4)

