import os
import sys
from datetime import datetime
from typing import Any, Dict, List

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QUrl
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QScrollArea,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from qwen_tts_workers import (
    BatchGenerateWorker,
    BatchItem,
    ModelConfig,
    ModelLoadWorker,
    PromptBuildWorker,
    QwenTTSBackend,
)


PARAM_HELP = {
    "do_sample": (
        "A boolean parameter that determines whether the model uses probabilistic "
        "sampling or deterministic decoding when generating audio tokens.\n\n"
        "Key Details:\n"
        "• Definition: Enables stochastic sampling instead of greedy decoding.\n"
        "• Typical Default: True for TTS use cases.\n"
        "• Range: True or False.\n"
        "• Usage: Must be enabled for top_k, top_p, and temperature to have any effect.\n"
        "• Effect:\n"
        "  - False -> deterministic, repeatable, often more robotic speech\n"
        "  - True -> more natural, varied, expressive speech"
    ),
    "top_k": (
        "A sampling parameter that limits token selection to the top K most probable candidates.\n\n"
        "Key Details:\n"
        "• Definition: Only the K highest-probability tokens are considered at each step.\n"
        "• Typical Default: 50.\n"
        "• Range: Integer >= 1.\n"
        "• Usage: Works with top_p and temperature to control randomness.\n"
        "• Effect:\n"
        "  - Lower values (10-30) -> more stable, less variation\n"
        "  - Higher values (100+) -> more diversity, possible instability"
    ),
    "top_p": (
        "A nucleus sampling parameter that selects tokens whose cumulative probability mass is <= p.\n\n"
        "Key Details:\n"
        "• Definition: Dynamically selects a subset of tokens whose total probability <= top_p.\n"
        "• Typical Default: 0.9 to 1.0.\n"
        "• Range: 0.0 to 1.0.\n"
        "• Usage: Often preferred over top_k for smoother control.\n"
        "• Effect:\n"
        "  - Lower values (0.8-0.9) -> safer, more consistent speech\n"
        "  - Higher values (0.95-1.0) -> more expressive, but greater risk of artifacts"
    ),
    "temperature": (
        "Controls how sharply probabilities are distributed during sampling.\n\n"
        "Key Details:\n"
        "• Definition: Scales logits before sampling.\n"
        "• Typical Default: 0.7 to 1.0.\n"
        "• Range: > 0.0.\n"
        "• Usage: Primary control for randomness.\n"
        "• Effect:\n"
        "  - Low (0.3-0.6) -> deterministic, flatter speech\n"
        "  - Medium (0.7-1.0) -> natural balance\n"
        "  - High (1.2+) -> expressive but potentially unstable"
    ),
    "repetition_penalty": (
        "Penalizes repeated tokens to reduce looping or monotony.\n\n"
        "Key Details:\n"
        "• Definition: Reduces probability of previously generated tokens.\n"
        "• Typical Default: 1.1 to 1.2.\n"
        "• Range: >= 1.0 in normal use.\n"
        "• Usage: Helps avoid repeated phonemes or repetitive patterns.\n"
        "• Effect:\n"
        "  - 1.0 -> no penalty\n"
        "  - 1.1-1.3 -> natural repetition control\n"
        "  - Too high -> unnatural or broken speech"
    ),
    "subtalker_dosample": (
        "A sampling toggle specifically for the sub-talker component.\n\n"
        "Key Details:\n"
        "• Definition: Enables sampling for the sub-talker stage.\n"
        "• Typical Default: True.\n"
        "• Range: True or False.\n"
        "• Usage: Independent from the main do_sample setting.\n"
        "• Effect:\n"
        "  - False -> very stable, possibly flatter audio detail\n"
        "  - True -> richer, more human-like micro-variation"
    ),
    "subtalker_top_k": (
        "Top-k sampling applied to sub-talker token generation.\n\n"
        "Key Details:\n"
        "• Definition: Limits sub-talker token choices to the top K candidates.\n"
        "• Typical Default: 50.\n"
        "• Range: Integer >= 1.\n"
        "• Usage: Works with subtalker_top_p and subtalker_temperature.\n"
        "• Effect:\n"
        "  - Lower -> cleaner, more stable audio texture\n"
        "  - Higher -> more nuanced detail, but potentially noisier output"
    ),
    "subtalker_top_p": (
        "A sampling parameter used in Qwen3-TTS to control the diversity and randomness "
        "of the generated audio in the sub-talker component.\n\n"
        "Key Details:\n"
        "• Definition: Acts as a top-p (nucleus) sampling parameter for the sub-talker, "
        "restricting the cumulative probability of potential tokens.\n"
        "• Typical Default: 1.0.\n"
        "• Range: 0.0 to 1.0.\n"
        "• Usage: Often adjusted alongside subtalker_temperature and subtalker_top_k.\n"
        "• Effect:\n"
        "  - Lowering to 0.9 or 0.8 makes generation more conservative\n"
        "  - Higher values allow more diverse output"
    ),
    "subtalker_temperature": (
        "Temperature applied to sub-talker sampling.\n\n"
        "Key Details:\n"
        "• Definition: Controls randomness of acoustic token generation.\n"
        "• Typical Default: 0.9.\n"
        "• Range: > 0.0.\n"
        "• Usage: Fine-tunes expressiveness at the micro level.\n"
        "• Effect:\n"
        "  - Low -> flatter, more consistent tone\n"
        "  - Medium -> natural realism\n"
        "  - High -> expressive but potentially unstable"
    ),
    "max_new_tokens": (
        "Limits how many new codec tokens are generated.\n\n"
        "Key Details:\n"
        "• Definition: Maximum length of generated output.\n"
        "• Typical Default: task dependent.\n"
        "• Range: Integer > 0.\n"
        "• Usage: Controls duration and cost of generation.\n"
        "• Effect:\n"
        "  - Lower values -> shorter audio\n"
        "  - Higher values -> longer audio, slower generation, more memory use"
    ),
}


class HelpDialog(QDialog):
    def __init__(self, title: str, text: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(520, 380)

        layout = QVBoxLayout(self)

        body = QPlainTextEdit()
        body.setReadOnly(True)
        body.setPlainText(text)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        buttons.button(QDialogButtonBox.StandardButton.Close).clicked.connect(self.accept)

        layout.addWidget(body)
        layout.addWidget(buttons)


class ModelTuningPanel(QGroupBox):
    kwargs_changed = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__("Model Tuning Parameters", parent)
        self.widgets: Dict[str, Any] = {}
        self.current_kwargs: Dict[str, Any] = {}

        outer_layout = QVBoxLayout(self)
        outer_layout.setSpacing(12)

        panel_row = QHBoxLayout()
        panel_row.setSpacing(12)
        panel_row.addWidget(self._build_main_talker_group(), 1)
        panel_row.addWidget(self._build_subtalker_group(), 1)

        button_row = QHBoxLayout()
        button_row.addStretch()

        self.reset_btn = QPushButton("Restore Defaults")
        self.reset_btn.clicked.connect(self.reset_defaults)
        self.reset_btn.setMinimumWidth(160)
        button_row.addWidget(self.reset_btn)

        button_row.addStretch()

        outer_layout.addLayout(panel_row)
        outer_layout.addLayout(button_row)

        self.update_enabled_states()
        self.update_kwargs()

    def _build_main_talker_group(self) -> QGroupBox:
        box = QGroupBox("Main Talker")
        layout = QFormLayout(box)
        layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        layout.setFormAlignment(Qt.AlignmentFlag.AlignTop)
        layout.setSpacing(10)

        self.widgets["do_sample"] = QCheckBox()
        self.widgets["do_sample"].setChecked(True)
        self.widgets["do_sample"].toggled.connect(self.update_enabled_states)
        self.widgets["do_sample"].toggled.connect(self.update_kwargs)
        layout.addRow("do_sample", self._control_with_help(self.widgets["do_sample"], "do_sample"))

        self.widgets["top_k"] = QSpinBox()
        self.widgets["top_k"].setRange(1, 500)
        self.widgets["top_k"].setValue(50)
        self.widgets["top_k"].valueChanged.connect(self.update_kwargs)
        layout.addRow("top_k", self._control_with_help(self.widgets["top_k"], "top_k"))

        self.widgets["top_p"] = QDoubleSpinBox()
        self.widgets["top_p"].setRange(0.0, 1.0)
        self.widgets["top_p"].setSingleStep(0.01)
        self.widgets["top_p"].setDecimals(2)
        self.widgets["top_p"].setValue(1.00)
        self.widgets["top_p"].valueChanged.connect(self.update_kwargs)
        layout.addRow("top_p", self._control_with_help(self.widgets["top_p"], "top_p"))

        self.widgets["temperature"] = QDoubleSpinBox()
        self.widgets["temperature"].setRange(0.05, 2.00)
        self.widgets["temperature"].setSingleStep(0.05)
        self.widgets["temperature"].setDecimals(2)
        self.widgets["temperature"].setValue(0.90)
        self.widgets["temperature"].valueChanged.connect(self.update_kwargs)
        layout.addRow("temperature", self._control_with_help(self.widgets["temperature"], "temperature"))

        self.widgets["repetition_penalty"] = QDoubleSpinBox()
        self.widgets["repetition_penalty"].setRange(1.00, 2.00)
        self.widgets["repetition_penalty"].setSingleStep(0.01)
        self.widgets["repetition_penalty"].setDecimals(2)
        self.widgets["repetition_penalty"].setValue(1.05)
        self.widgets["repetition_penalty"].valueChanged.connect(self.update_kwargs)
        layout.addRow(
            "repetition_penalty",
            self._control_with_help(self.widgets["repetition_penalty"], "repetition_penalty"),
        )

        self.widgets["max_new_tokens"] = QSpinBox()
        self.widgets["max_new_tokens"].setRange(1, 32768)
        self.widgets["max_new_tokens"].setValue(2048)
        self.widgets["max_new_tokens"].valueChanged.connect(self.update_kwargs)
        layout.addRow(
            "max_new_tokens",
            self._control_with_help(self.widgets["max_new_tokens"], "max_new_tokens"),
        )

        return box

    def _build_subtalker_group(self) -> QGroupBox:
        box = QGroupBox("Sub-Talker")
        layout = QFormLayout(box)
        layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        layout.setFormAlignment(Qt.AlignmentFlag.AlignTop)
        layout.setSpacing(10)

        note = QLabel("Used for qwen3-tts-tokenizer-v2 style models when applicable.")
        note.setWordWrap(True)
        note.setStyleSheet("color: gray;")
        layout.addRow(note)

        self.widgets["subtalker_dosample"] = QCheckBox()
        self.widgets["subtalker_dosample"].setChecked(True)
        self.widgets["subtalker_dosample"].toggled.connect(self.update_enabled_states)
        self.widgets["subtalker_dosample"].toggled.connect(self.update_kwargs)
        layout.addRow(
            "subtalker_dosample",
            self._control_with_help(self.widgets["subtalker_dosample"], "subtalker_dosample"),
        )

        self.widgets["subtalker_top_k"] = QSpinBox()
        self.widgets["subtalker_top_k"].setRange(1, 500)
        self.widgets["subtalker_top_k"].setValue(50)
        self.widgets["subtalker_top_k"].valueChanged.connect(self.update_kwargs)
        layout.addRow(
            "subtalker_top_k",
            self._control_with_help(self.widgets["subtalker_top_k"], "subtalker_top_k"),
        )

        self.widgets["subtalker_top_p"] = QDoubleSpinBox()
        self.widgets["subtalker_top_p"].setRange(0.0, 1.0)
        self.widgets["subtalker_top_p"].setSingleStep(0.01)
        self.widgets["subtalker_top_p"].setDecimals(2)
        self.widgets["subtalker_top_p"].setValue(1.00)
        self.widgets["subtalker_top_p"].valueChanged.connect(self.update_kwargs)
        layout.addRow(
            "subtalker_top_p",
            self._control_with_help(self.widgets["subtalker_top_p"], "subtalker_top_p"),
        )

        self.widgets["subtalker_temperature"] = QDoubleSpinBox()
        self.widgets["subtalker_temperature"].setRange(0.05, 2.00)
        self.widgets["subtalker_temperature"].setSingleStep(0.05)
        self.widgets["subtalker_temperature"].setDecimals(2)
        self.widgets["subtalker_temperature"].setValue(0.90)
        self.widgets["subtalker_temperature"].valueChanged.connect(self.update_kwargs)
        layout.addRow(
            "subtalker_temperature",
            self._control_with_help(self.widgets["subtalker_temperature"], "subtalker_temperature"),
        )

        return box

    def _control_with_help(self, control: QWidget, key: str) -> QWidget:
        wrapper = QWidget()
        layout = QHBoxLayout(wrapper)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        help_btn = QPushButton("?")
        help_btn.setFixedWidth(30)
        help_btn.clicked.connect(lambda: self.show_help(key))

        layout.addWidget(control, 1)
        layout.addWidget(help_btn)

        return wrapper

    def show_help(self, key: str):
        info = PARAM_HELP.get(key)
        if not info:
            QMessageBox.information(self, "Help", f"No help text found for {key}.")
            return
        dlg = HelpDialog(key, info, self)
        dlg.exec()

    def update_enabled_states(self):
        do_sample = self.widgets["do_sample"].isChecked()
        for key in ("top_k", "top_p", "temperature"):
            self.widgets[key].setEnabled(do_sample)

        subtalker_do = self.widgets["subtalker_dosample"].isChecked()
        for key in ("subtalker_top_k", "subtalker_top_p", "subtalker_temperature"):
            self.widgets[key].setEnabled(subtalker_do)

    def update_kwargs(self):
        kwargs: Dict[str, Any] = {}

        kwargs["do_sample"] = self.widgets["do_sample"].isChecked()
        kwargs["repetition_penalty"] = self.widgets["repetition_penalty"].value()
        kwargs["max_new_tokens"] = self.widgets["max_new_tokens"].value()

        if self.widgets["do_sample"].isChecked():
            kwargs["top_k"] = self.widgets["top_k"].value()
            kwargs["top_p"] = self.widgets["top_p"].value()
            kwargs["temperature"] = self.widgets["temperature"].value()

        kwargs["subtalker_dosample"] = self.widgets["subtalker_dosample"].isChecked()

        if self.widgets["subtalker_dosample"].isChecked():
            kwargs["subtalker_top_k"] = self.widgets["subtalker_top_k"].value()
            kwargs["subtalker_top_p"] = self.widgets["subtalker_top_p"].value()
            kwargs["subtalker_temperature"] = self.widgets["subtalker_temperature"].value()

        self.current_kwargs = kwargs
        self.kwargs_changed.emit(dict(self.current_kwargs))

    def get_generation_kwargs(self) -> Dict[str, Any]:
        return dict(self.current_kwargs)

    def reset_defaults(self):
        self.widgets["do_sample"].setChecked(True)
        self.widgets["top_k"].setValue(50)
        self.widgets["top_p"].setValue(1.00)
        self.widgets["temperature"].setValue(0.90)
        self.widgets["repetition_penalty"].setValue(1.05)
        self.widgets["max_new_tokens"].setValue(2048)

        self.widgets["subtalker_dosample"].setChecked(True)
        self.widgets["subtalker_top_k"].setValue(50)
        self.widgets["subtalker_top_p"].setValue(1.00)
        self.widgets["subtalker_temperature"].setValue(0.90)

        self.update_enabled_states()
        self.update_kwargs()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Qwen3 TTS Voice Clone Trials")
        self.resize(1040, 760)

        self.backend = QwenTTSBackend()
        self.generation_kwargs: Dict[str, Any] = {}
        self.active_threads: List[QThread] = []
        self.reference_text = ""

        self._build_ui()
        self.tuning_panel.kwargs_changed.connect(self.on_kwargs_changed)
        self.on_kwargs_changed(self.tuning_panel.get_generation_kwargs())

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        root.addWidget(scroll)

        container = QWidget()
        scroll.setWidget(container)
        layout = QVBoxLayout(container)
        layout.setSpacing(12)

        self.tuning_panel = ModelTuningPanel()

        top_row = QHBoxLayout()
        top_row.setSpacing(12)
        top_row.addWidget(self._build_controls_group(), 1)
        top_row.addWidget(self.tuning_panel, 1)

        layout.addLayout(top_row)
        layout.addWidget(self._build_reference_group())
        layout.addWidget(self._build_batch_group())
        layout.addWidget(self._build_run_group())
        layout.addWidget(self._build_results_group())

    def _build_controls_group(self) -> QGroupBox:
        box = QGroupBox("Controls")
        layout = QFormLayout(box)

        self.language_combo = QComboBox()
        self.language_combo.addItems(["English", "Chinese", "Russian"])
        self.language_combo.setCurrentText("English")
        layout.addRow("Language", self.language_combo)

        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 64)
        self.batch_size_spin.setValue(4)
        layout.addRow("Batch size", self.batch_size_spin)

        self.device_combo = QComboBox()
        self.device_combo.setEditable(True)
        self.device_combo.addItems(["cuda:0", "cuda:1", "cpu"])
        self.device_combo.setCurrentText("cuda:0")
        layout.addRow("Device", self.device_combo)

        self.model_name_edit = QLineEdit("Qwen/Qwen3-TTS-12Hz-1.7B-Base")
        layout.addRow("Model", self.model_name_edit)

        self.output_dir_edit = QLineEdit(os.path.expanduser("~/projects/qwen-tts-installer/outputs"))
        browse = QPushButton("Browse...")
        browse.clicked.connect(self.browse_output_dir)

        out_row = QHBoxLayout()
        out_row.addWidget(self.output_dir_edit, 1)
        out_row.addWidget(browse)
        layout.addRow("Output directory", out_row)

        btn_row = QHBoxLayout()
        self.load_model_btn = QPushButton("Load Model")
        self.load_model_btn.clicked.connect(self.load_model)

        self.build_prompt_btn = QPushButton("Build Prompt")
        self.build_prompt_btn.clicked.connect(self.build_prompt)

        btn_row.addWidget(self.load_model_btn)
        btn_row.addWidget(self.build_prompt_btn)
        btn_row.addStretch()
        layout.addRow("", btn_row)

        self.model_status_label = QLabel("Model: not loaded")
        self.prompt_status_label = QLabel("Prompt: not built")
        status_row = QVBoxLayout()
        status_row.addWidget(self.model_status_label)
        status_row.addWidget(self.prompt_status_label)
        layout.addRow("", status_row)

        return box

    def _build_reference_group(self) -> QGroupBox:
        box = QGroupBox("Reference Voice")
        layout = QFormLayout(box)

        self.ref_audio_edit = QLineEdit()
        btn_audio = QPushButton("Browse...")
        btn_audio.clicked.connect(self.browse_ref_audio)
        row_audio = QHBoxLayout()
        row_audio.addWidget(self.ref_audio_edit, 1)
        row_audio.addWidget(btn_audio)
        layout.addRow("Reference audio", row_audio)

        self.ref_text_file_edit = QLineEdit()
        btn_text_file = QPushButton("Load Text File...")
        btn_text_file.clicked.connect(self.load_ref_text_file)
        row_text_file = QHBoxLayout()
        row_text_file.addWidget(self.ref_text_file_edit, 1)
        row_text_file.addWidget(btn_text_file)
        layout.addRow("Reference text file", row_text_file)

        return box

    def _build_batch_group(self) -> QGroupBox:
        box = QGroupBox("Voice Clone Batch Input")
        layout = QVBoxLayout(box)

        self.script_edit = QPlainTextEdit()
        self.script_edit.setPlaceholderText(
            "Enter one script per paragraph.\n\n"
            "Blank lines separate batch items."
        )
        self.script_edit.setMinimumHeight(180)
        layout.addWidget(QLabel("Script"))
        layout.addWidget(self.script_edit)

        return box

    def _build_run_group(self) -> QGroupBox:
        box = QGroupBox("")
        layout = QHBoxLayout(box)
        layout.addStretch()

        self.run_btn = QPushButton("Run Batch")
        self.run_btn.setMinimumWidth(140)
        self.run_btn.clicked.connect(self.run_batch)
        layout.addWidget(self.run_btn)

        layout.addStretch()
        return box

    def _build_results_group(self) -> QGroupBox:
        box = QGroupBox("Generated Files")
        layout = QVBoxLayout(box)

        self.generated_table = QTableWidget(0, 3)
        self.generated_table.setHorizontalHeaderLabels(["Trial", "Filename", "Duration (sec)"])
        header = self.generated_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.generated_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.generated_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(self.generated_table)

        buttons = QHBoxLayout()

        self.open_folder_btn = QPushButton("Open Batch Folder")
        self.open_folder_btn.clicked.connect(self.open_output_folder)

        self.clear_btn = QPushButton("Clear List")
        self.clear_btn.clicked.connect(lambda: self.generated_table.setRowCount(0))

        buttons.addWidget(self.open_folder_btn)
        buttons.addWidget(self.clear_btn)
        buttons.addStretch()

        layout.addLayout(buttons)

        return box

    def append_log(self, text: str):
        pass

    def on_kwargs_changed(self, kwargs: Dict[str, Any]):
        self.generation_kwargs = dict(kwargs)

    def browse_output_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Select Output Directory", self.output_dir_edit.text())
        if path:
            self.output_dir_edit.setText(path)

    def browse_ref_audio(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Reference Audio",
            "",
            "Audio Files (*.wav *.flac *.mp3 *.m4a *.ogg);;All Files (*)",
        )
        if path:
            self.ref_audio_edit.setText(path)

    def load_ref_text_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Reference Text File",
            "",
            "Text Files (*.txt);;All Files (*)",
        )
        if not path:
            return

        self.ref_text_file_edit.setText(path)
        try:
            with open(path, "r", encoding="utf-8") as f:
                self.reference_text = f.read()
            self.prompt_status_label.setText("Prompt: reference text loaded")
        except Exception as exc:
            QMessageBox.critical(self, "Error", str(exc))
            self.reference_text = ""

    def _start_worker(self, worker, thread: QThread):
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(lambda: self._cleanup_thread(thread))
        self.active_threads.append(thread)
        thread.start()

    def _cleanup_thread(self, thread: QThread):
        if thread in self.active_threads:
            self.active_threads.remove(thread)

    def load_model(self):
        config = ModelConfig(
            model_name=self.model_name_edit.text().strip(),
            device_map=self.device_combo.currentText().strip(),
            attn_implementation="flash_attention_2",
            dtype_name="bfloat16",
        )

        thread = QThread()
        worker = ModelLoadWorker(self.backend, config)

        worker.log.connect(self.append_log)
        worker.error.connect(self.show_error)
        worker.model_loaded.connect(self.on_model_loaded)

        self._start_worker(worker, thread)

    def on_model_loaded(self, model_name: str):
        self.model_status_label.setText(f"Model: loaded ({model_name})")
        self.prompt_status_label.setText("Prompt: not built")

    def build_prompt(self):
        ref_audio = self.ref_audio_edit.text().strip()
        ref_text = self.reference_text.strip()

        thread = QThread()
        worker = PromptBuildWorker(
            self.backend,
            ref_audio=ref_audio,
            ref_text=ref_text,
            x_vector_only_mode=False,
        )

        worker.log.connect(self.append_log)
        worker.error.connect(self.show_error)
        worker.prompt_ready.connect(self.on_prompt_ready)

        self._start_worker(worker, thread)

    def on_prompt_ready(self):
        self.prompt_status_label.setText("Prompt: ready")

    def _parse_scripts(self) -> List[str]:
        raw = self.script_edit.toPlainText().strip()
        if not raw:
            return []
        return [chunk.strip() for chunk in raw.split("\n\n") if chunk.strip()]

    def _build_batch_items(self) -> List[BatchItem]:
        scripts = self._parse_scripts()
        if not scripts:
            return []

        batch_size = self.batch_size_spin.value()
        language = self.language_combo.currentText()
        output_dir = self.output_dir_edit.text().strip()

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        selected = scripts[:batch_size]

        items = []
        for idx, text in enumerate(selected, start=1):
            filename = f"voice_clone_{stamp}_{idx:03d}.wav"
            output_path = os.path.join(output_dir, filename)
            items.append(
                BatchItem(
                    trial=idx,
                    text=text,
                    language=language,
                    output_path=output_path,
                )
            )
        return items

    def run_batch(self):
        batch_items = self._build_batch_items()
        if not batch_items:
            QMessageBox.warning(self, "Run Batch", "No batch items found.")
            return

        ref_audio = self.ref_audio_edit.text().strip()
        ref_text = self.reference_text.strip()

        thread = QThread()
        worker = BatchGenerateWorker(
            self.backend,
            batch_items=batch_items,
            ref_audio=ref_audio,
            ref_text=ref_text,
            generation_kwargs=self.generation_kwargs,
            x_vector_only_mode=False,
        )

        worker.log.connect(self.append_log)
        worker.error.connect(self.show_error)
        worker.batch_complete.connect(self.on_batch_complete)

        self._start_worker(worker, thread)

    def on_batch_complete(self, results: List):
        for trial, output_path, duration_sec in results:
            row = self.generated_table.rowCount()
            self.generated_table.insertRow(row)

            filename = os.path.basename(output_path)

            self.generated_table.setItem(row, 0, QTableWidgetItem(str(trial)))
            self.generated_table.setItem(row, 1, QTableWidgetItem(filename))
            self.generated_table.setItem(row, 2, QTableWidgetItem(f"{duration_sec:.2f}"))

    def open_output_folder(self):
        path = self.output_dir_edit.text().strip()
        if path:
            QDesktopServices.openUrl(QUrl.fromLocalFile(path))

    def show_error(self, text: str):
        QMessageBox.critical(self, "Error", text)


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
