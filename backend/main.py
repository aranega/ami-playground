import numpy as np
import socketio

from backend.services.websocket_service import get_images, get_aligned_images

sio = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins='*')
app = socketio.ASGIApp(sio)

sessions = {}


class TransformationSession(object):
    def __init__(self):
        self.transformations = {
            'rotation': {},
            'scale': {},
            'position': {},
        }
        self.pipeline = []
        self.initial_matrix = np.identity(4)

    def append(self, transformation, axis, value):
        self.transformations[transformation][axis] = value
        self.pipeline.append((transformation, {axis: value}))


# Define a Socket.IO event handler for when clients connect
@sio.on('connect')
async def connect(sid, *args, **kwargs):
    print('Client connected:', sid)


# Define a Socket.IO event handler for when clients disconnect
@sio.on('disconnect')
async def disconnect(sid, *args, **kwargs):
    print('Client disconnected:', sid)
    try:
        del sessions[sid]
    except KeyError:
        print("Session unknown for", sid)


# Define a Socket.IO event handler for when clients send a "start" event
@sio.on('start')
async def start(sid, *args, **kwargs):
    print('Starting image stream for client:', sid)
    sessions[sid] = TransformationSession()
    await sio.emit('images', get_images(), room=sid)


@sio.on('transform')
async def transform(sid, tform, axis, amount):
    print('Applying transform for client', sid)
    session = sessions[sid]
    await sio.emit('images', get_aligned_images(tform, axis, amount, session), room=sid)
    print('Finished applying transform for client', sid)


if __name__ == '__main__':
    import logging
    import sys

    logging.basicConfig(level=logging.DEBUG,
                        stream=sys.stdout)

    import uvicorn

    uvicorn.run("main:app", host='0.0.0.0', port=8000, reload=True)