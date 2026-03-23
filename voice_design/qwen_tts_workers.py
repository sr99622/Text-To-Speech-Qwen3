import json
import os
import traceback
from datetime import datetime

import soundfile as sf
import torch

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

from qwen_tts import Qwen3TTSModel


MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"


def safe_write_text_file(path: str, text: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


class ModelLoadWorker(QObject):
    finished = pyqtSignal(object, list)
    error = pyqtSignal(str)
    status = pyqtSignal(str)

    @pyqtSlot()
    def run(self):
        try:
            self.status.emit("Loading model...")

            model = Qwen3TTSModel.from_pretrained(
                MODEL_ID,
                device_map="cuda:0",
                dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
            )

            self.status.emit("Retrieving supported languages...")
            languages = model.get_supported_languages()

            if languages is None:
                languages = []
            elif not isinstance(languages, (list, tuple)):
                languages = list(languages)

            languages = [str(lang) for lang in languages]
            self.finished.emit(model, languages)

        except Exception:
            self.error.emit(traceback.format_exc())


class GenerateWorker(QObject):
    finished = pyqtSignal(list)
    error = pyqtSignal(str)
    status = pyqtSignal(str)

    def __init__(self, model, text, language, instruct, batch_size, output_dir):
        super().__init__()
        self.model = model
        self.text = text
        self.language = language
        self.instruct = instruct
        self.batch_size = batch_size
        self.output_dir = output_dir

    @pyqtSlot()
    def run(self):
        try:
            self.status.emit(f"Generating {self.batch_size} trial(s)...")

            texts = [self.text] * self.batch_size
            languages = [self.language] * self.batch_size
            instructs = [self.instruct] * self.batch_size

            wavs, sr = self.model.generate_voice_design(
                text=texts,
                language=languages,
                instruct=instructs,
            )

            os.makedirs(self.output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            batch_dir_name = f"batch_{timestamp}"
            batch_dir = os.path.join(self.output_dir, batch_dir_name)
            os.makedirs(batch_dir, exist_ok=True)

            safe_write_text_file(os.path.join(batch_dir, "text.txt"), self.text)
            safe_write_text_file(os.path.join(batch_dir, "instruct.txt"), self.instruct)

            results = []

            for i, wav in enumerate(wavs, start=1):
                filename = os.path.join(
                    batch_dir,
                    f"voice_design_{timestamp}_{i:03d}.wav",
                )

                sf.write(filename, wav, sr)
                duration_sec = len(wav) / float(sr)

                results.append(
                    {
                        "trial": i,
                        "path": filename,
                        "filename": os.path.basename(filename),
                        "duration_sec": duration_sec,
                        "sample_rate": sr,
                        "batch_dir": batch_dir,
                        "batch_name": batch_dir_name,
                    }
                )

                self.status.emit(
                    f"Saved {os.path.basename(filename)} ({duration_sec:.2f} sec)"
                )

            manifest = {
                "batch_name": batch_dir_name,
                "batch_dir": batch_dir,
                "created_at": timestamp,
                "language": self.language,
                "batch_size": self.batch_size,
                "text_file": "text.txt",
                "instruct_file": "instruct.txt",
                "files": results,
            }

            with open(os.path.join(batch_dir, "manifest.json"), "w", encoding="utf-8") as f:
                json.dump(manifest, f, ensure_ascii=False, indent=2)

            self.status.emit(f"Batch saved to {batch_dir}")
            self.finished.emit(results)

        except Exception:
            self.error.emit(traceback.format_exc())
