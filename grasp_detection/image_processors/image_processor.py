from typing import List, Type, Any
from abc import ABC, abstractmethod
import copy

from PIL import Image, ImageDraw
import numpy as np

class ImageProcessor(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def detect_obj(
        self,
        image: Type[Image.Image],
        text: str = None,
        bbox: List[int] = None
    ) -> Any:
        pass

    def draw_bounding_box(
        self,
        image: Type[Image.Image],
        bbox: List[int],
        save_file: str = None
    ) -> None:
        new_image = copy.deepcopy(image)
        img_drw = ImageDraw.Draw(new_image)
        img_drw.rectangle([(bbox[0], bbox[1]), (bbox[2], bbox[3])], outline="green")

        if save_file is not None:
            new_image.save(save_file)
    
    def draw_bounding_boxes(
        self,
        image: Type[Image.Image],
        bboxes: List[int],
        scores: list[int],
        max_box_ind: int = -1,
        save_file: str = None
    ) -> None:
        if (max_box_ind != -1):
            max_score = np.max(scores.detach().numpy())
            print(f"max_score: {max_score}")
            max_ind = np.argmax(scores.detach().numpy())
        max_box = bboxes.detach().numpy()[max_ind].astype(int)

        new_image = copy.deepcopy(image)
        img_drw = ImageDraw.Draw(new_image)
        img_drw.rectangle([(max_box[0], max_box[1]), (max_box[2], max_box[3])], outline="green")
        img_drw.text((max_box[0], max_box[1]), str(round(max_score.item(), 3)), fill="green")

        for box, score, label in zip(bboxes, scores):
            box = [int(i) for i in box.tolist()]
            # print(f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}")
            if (score == max_score):
                img_drw.rectangle([(box[0], box[1]), (box[2], box[3])], outline="red")
                img_drw.text((box[0], box[1]), str(round(max_score.item(), 3)), fill="red")
            else:
                img_drw.rectangle([(box[0], box[1]), (box[2], box[3])], outline="white")
        new_image.save(save_file)

    def show_mask_on_image(
        image: Type[Image.Image],
        mask: np.ndarray,
        save_file: str = None,
    ) -> None:
        pass
