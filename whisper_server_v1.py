import tornado.ioloop
import tornado.web
import tornado.gen
import base64
from faster_whisper import WhisperModel
import io
import numpy as np
import torch
from pydub import AudioSegment
import os
import librosa
from transformers import AutoTokenizer, AutoModelForTokenClassification
from jiwer import wer

cuda_available = torch.cuda.is_available()

if cuda_available:
    try:
        import ctypes
        ctypes.CDLL("libcudnn_ops_infer.so.8")
        device = "cuda"
        compute_type = "float16"
    except OSError:
        print("CUDA libraries not found. falling back to CPU.")
        device = "cpu"
        compute_type = "int8"
else:
    device = "cpu"
    compute_type = "int8"

# Initialize the model
model = WhisperModel("large-v3", device=device, compute_type=compute_type)

# Initialize Hugging Face NER model and tokenizer
ner_model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
ner_tokenizer = AutoTokenizer.from_pretrained(ner_model_name)
ner_model = AutoModelForTokenClassification.from_pretrained(ner_model_name)
ner_model = ner_model.to("cuda" if cuda_available else "cpu")

# Define a default temperature value
default_temperature = 0.2

def transcribe_audio(audio_file_path, temperature=default_temperature):
    # Load audio using librosa for better audio processing
    audio_array, sample_rate = librosa.load(audio_file_path, sr=16000)
    
    # Normalize audio
    audio_array = librosa.util.normalize(audio_array)
    
    # Apply noise reduction
    audio_array = librosa.effects.preemphasis(audio_array)
    
    # Transcribe with beam search and language set to English (closest to Singlish)
    segments, info = model.transcribe(audio_array, 
                                      temperature=temperature,
                                      beam_size=5,
                                      language='en',
                                      vad_filter=True,
                                      word_timestamps=True)
    return segments, info

def detect_entities(text):
    inputs = ner_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = inputs.to(ner_model.device)
    
    with torch.no_grad():
        outputs = ner_model(**inputs)
    
    predictions = torch.argmax(outputs.logits, dim=2)
    tokens = ner_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    
    entities = []
    for token, prediction in zip(tokens, predictions[0]):
        if prediction != 0:  # 0 is usually the 'O' (Outside) label
            entity = {
                'word': token,  # Use the token directly instead of converting
                'entity': ner_model.config.id2label[prediction.item()]
            }
            entities.append(entity)
    
    return entities

def calculate_wer(reference, hypothesis):
    return wer(reference, hypothesis)

def transcribe(audio_file_path, temperature=default_temperature, reference_text=None):
    segments, info = transcribe_audio(audio_file_path, temperature)

    transcription = ""
    for segment in segments:
        transcription += segment.text.strip() + " "
    
    transcription = transcription.strip()
    entities = detect_entities(transcription)
    
    word_error_rate = None
    if reference_text:
        word_error_rate = calculate_wer(reference_text, transcription)
    
    return transcription, info, entities, word_error_rate

if __name__ == "__main__":
    audio_file_path = "test_audio.wav"
    temperature = 0.2
    reference_text = "This is a sample reference text for WER calculation."
    transcription_result, info, entities, word_error_rate = transcribe(audio_file_path, temperature, reference_text)
    print(f"Transcription result (temperature={temperature}):")
    print(transcription_result)
    print(f"\nDetected language: {info.language}")
    print(f"Language probability: {info.language_probability}")
    print("\nDetected entities:")
    for entity in entities:
        print(f"{entity['word']} - {entity['entity']}")
    if word_error_rate is not None:
        print(f"\nWord Error Rate: {word_error_rate:.4f}")
