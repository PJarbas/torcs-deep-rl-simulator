import base64
import io

import numpy as np
from PIL import Image


def encode_frame(frame: np.ndarray, fmt: str = "JPEG") -> str:
    img = Image.fromarray(frame)
    buffer = io.BytesIO()
    img.save(buffer, format=fmt)
    return base64.b64encode(buffer.getvalue()).decode("ascii")
