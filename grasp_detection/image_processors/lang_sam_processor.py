from typing import List, Type, Tuple

from PiL import Image
import numpy as np

from image_processor import ImageProcessor
from lang_sam import LangSAM

class LangSAMProcessor(ImageProcessor):
    def __init__(self):
        self.model = LangSAM()

    def detect_obj(
        self,
        image: Type[Image.Image],
        text: str = None,
        bbox: List[int] = None,
        save_file: str = None,
        visualize_box: bool = False
    ) -> Tuple[np.ndarray, List[int]]:

        masks, boxes, phrases, logits = self.model.predict(image, text)
        seg_mask = np.array(masks[0])
        bbox = np.array(boxes[0], dtype=int)

        if visualize_box:
            self.draw_bounding_box(image, bbox, save_file)

        return masks, bbox
