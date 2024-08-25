import tornado.ioloop
import tornado.web
import tornado.gen
import base64
from faster_whisper import WhisperModel
import io
import numpy as np
import torch

# Load Whisper model
model = WhisperModel("large-v2", device="cuda", compute_type="float16")

class TranscribeHandler(tornado.web.RequestHandler):
    async def post(self):
        try:
            # Get the audio buffer from the request body
            audio_data = self.request.body
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            segments, _ = model.transcribe(audio_np)
            transcription = ''.join([segment.text for segment in segments])
            self.write({"transcription": transcription})

        except Exception as e:
            self.set_status(500)
            self.write({"error": str(e)})

def make_app():
    return tornado.web.Application([
        (r"/transcribe", TranscribeHandler),
    ])

if __name__ == "__main__":
    app = make_app()
    app.listen(8889)  # Run the server on port 8888
    tornado.ioloop.IOLoop.current().start()
