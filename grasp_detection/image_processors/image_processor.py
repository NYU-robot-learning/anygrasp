from typing import List, Type
from abc import ABC, abstractmethod

from PIL import Image

class ImageProcessor(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def detect_obj():
        pass

    def draw_bounding_box(
        self,
        image: Type[Image.Image],
        bbox: List[int],
        save_file: str = None
    ) -> None:
        pass
    
    def draw_bounding_boxes(
        self,
        image: Type[Image.Image],
        bboxes: List[int],
        scores: list[int],
        labels: List[int],
        save_file: str = None
    ) -> None:
        pass
