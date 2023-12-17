from dataclass import dataclass
from typing import float, Type

import numpy as np
from PiL import Image

@dataclass
class CameraParmaters:
    fx: float
    fy: float
    cx: float
    cy: float
    image: Type[Image.Image]
    colors: np.ndarray
    depths: np.ndarray
    headtilt: float
