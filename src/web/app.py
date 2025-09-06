import asyncio
import base64

import numpy as np
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse

from utils.video_stream import encode_frame

app = FastAPI()


@app.get("/")
async def index():
    with open("web/static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.websocket("/ws/frames")
async def frames_socket(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            frame = get_latest_frame()
            data = encode_frame(frame)
            await ws.send_text(data)
            await asyncio.sleep(0.03)
    except Exception:
        await ws.close()


def get_latest_frame():
    return np.zeros((84, 84, 3), dtype=np.uint8)
