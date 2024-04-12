import argparse
import os
import numpy as np
import speech_recognition as sr
import whisper
import torch

from datetime import datetime, timedelta
from queue import Queue
from time import sleep
from sys import platform

audio_model = whisper.load_model("medium.en")

def transcribe_audio(audio_data):
    global audio_model
    audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
    result = audio_model.transcribe(audio_np, fp16=torch.cuda.is_available())
    text = result['text'].strip()


