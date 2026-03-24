import os
import traceback
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import soundfile as sf


@dataclass
class ModelConfig:
    model_name: str
    device_map: str = "cuda:0"
    attn_implementation: str = "flash_attention_2"
    dtype_name: str = "bfloat16"


@dataclass
class BatchItem:
    trial: int
    text: str
    language: str
    output_path: str


class QwenTTSBackend:
    """
    Persistent backend that owns the model and cached voice clone prompt.
    Heavy work should be called from worker threads, not the GUI thread.
    """

    def __init__(self):
        self.model = None
        self.model_config: Optional[ModelConfig] = None

        self.cached_prompt = None
        self.cached_ref_audio: Optional[str] = None
        self.cached_ref_text: Optional[str] = None
        self.cached_x_vector_only_mode: Optional[bool] = None

    def is_model_loaded(self) -> bool:
        return self.model is not None

    def unload_model(self):
        self.model = None
        self.model_config = None
        self.invalidate_prompt_cache()

    def invalidate_prompt_cache(self):
        self.cached_prompt = None
        self.cached_ref_audio = None
        self.cached_ref_text = None
        self.cached_x_vector_only_mode = None

    def load_model(self, config: ModelConfig):
        import torch
        from qwen_tts import Qwen3TTSModel

        dtype = getattr(torch, config.dtype_name)

        self.model = Qwen3TTSModel.from_pretrained(
            config.model_name,
            device_map=config.device_map,
            dtype=dtype,
            attn_implementation=config.attn_implementation,
        )
        self.model_config = config
        self.invalidate_prompt_cache()

    def ensure_prompt(
        self,
        ref_audio: str,
        ref_text: str,
        x_vector_only_mode: bool = False,
    ):
        if self.model is None:
            raise RuntimeError("Model is not loaded.")

        if not os.path.isfile(ref_audio):
            raise FileNotFoundError(f"Reference audio not found: {ref_audio}")

        if not ref_text.strip():
            raise ValueError("Reference text is empty.")

        cache_hit = (
            self.cached_prompt is not None
            and self.cached_ref_audio == ref_audio
            and self.cached_ref_text == ref_text
            and self.cached_x_vector_only_mode == x_vector_only_mode
        )
        if cache_hit:
            return self.cached_prompt

        prompt_items = self.model.create_voice_clone_prompt(
            ref_audio=ref_audio,
            ref_text=ref_text,
            x_vector_only_mode=x_vector_only_mode,
        )

        self.cached_prompt = prompt_items
        self.cached_ref_audio = ref_audio
        self.cached_ref_text = ref_text
        self.cached_x_vector_only_mode = x_vector_only_mode
        return prompt_items

    def generate_voice_clone_batch(
        self,
        batch_items: List[BatchItem],
        ref_audio: str,
        ref_text: str,
        generation_kwargs: Dict,
        x_vector_only_mode: bool = False,
    ) -> List[Tuple[int, str, float]]:
        if self.model is None:
            raise RuntimeError("Model is not loaded.")

        if not batch_items:
            raise ValueError("No batch items to generate.")

        prompt_items = self.ensure_prompt(
            ref_audio=ref_audio,
            ref_text=ref_text,
            x_vector_only_mode=x_vector_only_mode,
        )

        texts = [item.text for item in batch_items]
        languages = [item.language for item in batch_items]

        wavs, sr = self.model.generate_voice_clone(
            text=texts,
            language=languages,
            voice_clone_prompt=prompt_items,
            **generation_kwargs,
        )

        results = []
        for item, wav in zip(batch_items, wavs):
            os.makedirs(os.path.dirname(item.output_path), exist_ok=True)
            sf.write(item.output_path, wav, sr)

            duration_sec = 0.0
            try:
                duration_sec = float(len(wav)) / float(sr)
            except Exception:
                pass

            results.append((item.trial, item.output_path, duration_sec))

        return results


from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot


class ModelLoadWorker(QObject):
    finished = pyqtSignal()
    error = pyqtSignal(str)
    log = pyqtSignal(str)
    model_loaded = pyqtSignal(str)

    def __init__(self, backend: QwenTTSBackend, config: ModelConfig):
        super().__init__()
        self.backend = backend
        self.config = config

    @pyqtSlot()
    def run(self):
        try:
            self.log.emit(f"Loading model: {self.config.model_name}")
            self.log.emit(f"Device: {self.config.device_map}")
            self.log.emit(f"Attention: {self.config.attn_implementation}")
            self.log.emit(f"DType: {self.config.dtype_name}")
            self.backend.load_model(self.config)
            self.model_loaded.emit(self.config.model_name)
        except Exception as exc:
            self.error.emit(f"{exc}\n\n{traceback.format_exc()}")
        finally:
            self.finished.emit()


class PromptBuildWorker(QObject):
    finished = pyqtSignal()
    error = pyqtSignal(str)
    log = pyqtSignal(str)
    prompt_ready = pyqtSignal()

    def __init__(
        self,
        backend: QwenTTSBackend,
        ref_audio: str,
        ref_text: str,
        x_vector_only_mode: bool = False,
    ):
        super().__init__()
        self.backend = backend
        self.ref_audio = ref_audio
        self.ref_text = ref_text
        self.x_vector_only_mode = x_vector_only_mode

    @pyqtSlot()
    def run(self):
        try:
            self.log.emit("Building voice clone prompt...")
            self.backend.ensure_prompt(
                ref_audio=self.ref_audio,
                ref_text=self.ref_text,
                x_vector_only_mode=self.x_vector_only_mode,
            )
            self.prompt_ready.emit()
        except Exception as exc:
            self.error.emit(f"{exc}\n\n{traceback.format_exc()}")
        finally:
            self.finished.emit()


class BatchGenerateWorker(QObject):
    finished = pyqtSignal()
    error = pyqtSignal(str)
    log = pyqtSignal(str)
    batch_complete = pyqtSignal(list)

    def __init__(
        self,
        backend: QwenTTSBackend,
        batch_items: List[BatchItem],
        ref_audio: str,
        ref_text: str,
        generation_kwargs: Dict,
        x_vector_only_mode: bool = False,
    ):
        super().__init__()
        self.backend = backend
        self.batch_items = batch_items
        self.ref_audio = ref_audio
        self.ref_text = ref_text
        self.generation_kwargs = generation_kwargs
        self.x_vector_only_mode = x_vector_only_mode

    @pyqtSlot()
    def run(self):
        try:
            self.log.emit(f"Generating batch with {len(self.batch_items)} item(s)...")
            results = self.backend.generate_voice_clone_batch(
                batch_items=self.batch_items,
                ref_audio=self.ref_audio,
                ref_text=self.ref_text,
                generation_kwargs=self.generation_kwargs,
                x_vector_only_mode=self.x_vector_only_mode,
            )
            self.batch_complete.emit(results)
        except Exception as exc:
            self.error.emit(f"{exc}\n\n{traceback.format_exc()}")
        finally:
            self.finished.emit()
