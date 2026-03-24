import os
import sys
from datetime import datetime
from typing import Any, Dict, List

from PyQt6.QtCore import Qt, QUrl, pyqtSignal
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
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout.addWidget(body)
        layout.addWidget(buttons)


class ModelTuningPanel(QGroupBox):
    kwargs_changed = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__("", parent)
        self.widgets: Dict[str, Any] = {}
        self.current_kwargs: Dict[str, Any] = {}

        outer = QVBoxLayout(self)
        outer.setSpacing(8)

        row = QHBoxLayout()
        row.addWidget(self._build_main(), 1)
        row.addWidget(self._build_sub(), 1)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self.reset_btn = QPushButton("Restore Defaults")
        self.reset_btn.clicked.connect(self.reset_defaults)
        btn_row.addWidget(self.reset_btn)
        btn_row.addStretch()

        outer.addLayout(row)
        outer.addLayout(btn_row)

        self.update_enabled_states()
        self.update_kwargs()

    def _build_main(self):
        box = QGroupBox("Main Talker")
        f = QFormLayout(box)

        self.widgets["do_sample"] = QCheckBox()
        self.widgets["do_sample"].setChecked(True)
        self.widgets["do_sample"].toggled.connect(self.update_enabled_states)
        self.widgets["do_sample"].toggled.connect(self.update_kwargs)
        f.addRow("do_sample", self._wrap(self.widgets["do_sample"], "do_sample"))

        self.widgets["top_k"] = QSpinBox()
        self.widgets["top_k"].setRange(1, 500)
        self.widgets["top_k"].setValue(50)
        self.widgets["top_k"].valueChanged.connect(self.update_kwargs)
        f.addRow("top_k", self._wrap(self.widgets["top_k"], "top_k"))

        self.widgets["top_p"] = QDoubleSpinBox()
        self.widgets["top_p"].setRange(0.0, 1.0)
        self.widgets["top_p"].setDecimals(2)
        self.widgets["top_p"].setSingleStep(0.01)
        self.widgets["top_p"].setValue(1.00)
        self.widgets["top_p"].valueChanged.connect(self.update_kwargs)
        f.addRow("top_p", self._wrap(self.widgets["top_p"], "top_p"))

        self.widgets["temperature"] = QDoubleSpinBox()
        self.widgets["temperature"].setRange(0.05, 2.00)
        self.widgets["temperature"].setDecimals(2)
        self.widgets["temperature"].setSingleStep(0.05)
        self.widgets["temperature"].setValue(0.90)
        self.widgets["temperature"].valueChanged.connect(self.update_kwargs)
        f.addRow("temperature", self._wrap(self.widgets["temperature"], "temperature"))

        self.widgets["repetition_penalty"] = QDoubleSpinBox()
        self.widgets["repetition_penalty"].setRange(1.00, 2.00)
        self.widgets["repetition_penalty"].setDecimals(2)
        self.widgets["repetition_penalty"].setSingleStep(0.01)
        self.widgets["repetition_penalty"].setValue(1.05)
        self.widgets["repetition_penalty"].valueChanged.connect(self.update_kwargs)
        f.addRow(
            "repetition_penalty",
            self._wrap(self.widgets["repetition_penalty"], "repetition_penalty"),
        )

        self.widgets["max_new_tokens"] = QSpinBox()
        self.widgets["max_new_tokens"].setRange(1, 32768)
        self.widgets["max_new_tokens"].setValue(2048)
        self.widgets["max_new_tokens"].valueChanged.connect(self.update_kwargs)
        f.addRow(
            "max_new_tokens",
            self._wrap(self.widgets["max_new_tokens"], "max_new_tokens"),
        )

        return box

    def _build_sub(self):
        box = QGroupBox("Sub-Talker")
        f = QFormLayout(box)

        note = QLabel("Used for qwen3-tts-tokenizer-v2 style models when applicable.")
        note.setWordWrap(True)
        note.setStyleSheet("color: gray;")
        f.addRow(note)

        self.widgets["subtalker_dosample"] = QCheckBox()
        self.widgets["subtalker_dosample"].setChecked(True)
        self.widgets["subtalker_dosample"].toggled.connect(self.update_enabled_states)
        self.widgets["subtalker_dosample"].toggled.connect(self.update_kwargs)
        f.addRow(
            "subtalker_dosample",
            self._wrap(self.widgets["subtalker_dosample"], "subtalker_dosample"),
        )

        self.widgets["subtalker_top_k"] = QSpinBox()
        self.widgets["subtalker_top_k"].setRange(1, 500)
        self.widgets["subtalker_top_k"].setValue(50)
        self.widgets["subtalker_top_k"].valueChanged.connect(self.update_kwargs)
        f.addRow(
            "subtalker_top_k",
            self._wrap(self.widgets["subtalker_top_k"], "subtalker_top_k"),
        )

        self.widgets["subtalker_top_p"] = QDoubleSpinBox()
        self.widgets["subtalker_top_p"].setRange(0.0, 1.0)
        self.widgets["subtalker_top_p"].setDecimals(2)
        self.widgets["subtalker_top_p"].setSingleStep(0.01)
        self.widgets["subtalker_top_p"].setValue(1.00)
        self.widgets["subtalker_top_p"].valueChanged.connect(self.update_kwargs)
        f.addRow(
            "subtalker_top_p",
            self._wrap(self.widgets["subtalker_top_p"], "subtalker_top_p"),
        )

        self.widgets["subtalker_temperature"] = QDoubleSpinBox()
        self.widgets["subtalker_temperature"].setRange(0.05, 2.00)
        self.widgets["subtalker_temperature"].setDecimals(2)
        self.widgets["subtalker_temperature"].setSingleStep(0.05)
        self.widgets["subtalker_temperature"].setValue(0.90)
        self.widgets["subtalker_temperature"].valueChanged.connect(self.update_kwargs)
        f.addRow(
            "subtalker_temperature",
            self._wrap(self.widgets["subtalker_temperature"], "subtalker_temperature"),
        )

        return box

    def _wrap(self, widget, key):
        w = QWidget()
        l = QHBoxLayout(w)
        l.setContentsMargins(0, 0, 0, 0)

        btn = QPushButton("?")
        btn.setFixedWidth(28)
        btn.clicked.connect(lambda: HelpDialog(key, PARAM_HELP.get(key, ""), self).exec())

        l.addWidget(widget, 1)
        l.addWidget(btn)
        return w

    def update_enabled_states(self):
        do_sample = self.widgets["do_sample"].isChecked()
        self.widgets["top_k"].setEnabled(do_sample)
        self.widgets["top_p"].setEnabled(do_sample)
        self.widgets["temperature"].setEnabled(do_sample)

        subtalker_dosample = self.widgets["subtalker_dosample"].isChecked()
        self.widgets["subtalker_top_k"].setEnabled(subtalker_dosample)
        self.widgets["subtalker_top_p"].setEnabled(subtalker_dosample)
        self.widgets["subtalker_temperature"].setEnabled(subtalker_dosample)

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

    def get_generation_kwargs(self):
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
        self.resize(1040, 900)

        self.backend = QwenTTSBackend()
        self.reference_text = ""
        self.generation_kwargs = {}

        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        top = QHBoxLayout()
        self.tuning_panel = ModelTuningPanel()
        self.tuning_panel.kwargs_changed.connect(self.on_kwargs_changed)
        self.on_kwargs_changed(self.tuning_panel.get_generation_kwargs())

        top.addWidget(self._build_controls(), 1)
        top.addWidget(self.tuning_panel, 1)

        root.addLayout(top)
        root.addWidget(self._build_batch())
        root.addWidget(self._build_results())

    def _build_controls(self):
        box = QGroupBox("")
        f = QFormLayout(box)

        self.language = QComboBox()
        self.language.addItems(["English", "Chinese", "Russian"])
        f.addRow("Language", self.language)

        self.batch = QSpinBox()
        self.batch.setValue(4)
        f.addRow("Batch size", self.batch)

        self.device = QComboBox()
        self.device.addItems(["cuda:0", "cuda:1", "cpu"])
        f.addRow("Device", self.device)

        self.model = QLineEdit("Qwen/Qwen3-TTS-12Hz-1.7B-Base")
        f.addRow("Model", self.model)

        self.out = QLineEdit(os.path.expanduser("~/outputs"))
        out_btn = QPushButton("Browse...")
        out_btn.clicked.connect(self.browse_output)
        row = QHBoxLayout()
        row.addWidget(self.out)
        row.addWidget(out_btn)
        f.addRow("Output Dir", row)

        self.ref_audio = QLineEdit()
        btn = QPushButton("Browse...")
        btn.clicked.connect(self.browse_audio)
        row = QHBoxLayout()
        row.addWidget(self.ref_audio)
        row.addWidget(btn)
        f.addRow("Ref Audio", row)

        self.ref_text = QLineEdit()
        btn = QPushButton("Browse...")
        btn.clicked.connect(self.load_text)
        row = QHBoxLayout()
        row.addWidget(self.ref_text)
        row.addWidget(btn)
        f.addRow("Ref Text File", row)

        btn_row = QHBoxLayout()

        self.load_model_btn = QPushButton("Load Model")
        self.load_model_btn.clicked.connect(self.load_model)

        self.build_prompt_btn = QPushButton("Build Prompt")
        self.build_prompt_btn.clicked.connect(self.build_prompt)

        self.model_status_label = QLabel("not loaded")
        self.prompt_status_label = QLabel("not built")

        btn_row.addWidget(self.load_model_btn)
        btn_row.addWidget(self.model_status_label)
        btn_row.addWidget(self.build_prompt_btn)
        btn_row.addWidget(self.prompt_status_label)

        f.addRow("", btn_row)

        return box

    def _build_batch(self):
        box = QGroupBox("Voice Clone Batch Input")
        v = QVBoxLayout(box)

        self.script = QPlainTextEdit()
        self.script.setPlaceholderText(
            "Enter one script per paragraph.\n\nBlank lines separate batch items."
        )
        v.addWidget(self.script)

        btn_row = QHBoxLayout()
        btn_row.addStretch()

        self.run_btn = QPushButton("Run Batch")
        self.run_btn.clicked.connect(self.run_batch)

        btn_row.addWidget(self.run_btn)
        btn_row.addStretch()

        v.addLayout(btn_row)
        return box

    def _build_results(self):
        box = QGroupBox("Generated Files")
        v = QVBoxLayout(box)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Trial", "Filename", "Duration (sec)"])

        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)

        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)

        v.addWidget(self.table)

        row = QHBoxLayout()
        btn = QPushButton("Open Batch Folder")
        btn.clicked.connect(self.open_folder)

        clr = QPushButton("Clear List")
        clr.clicked.connect(lambda: self.table.setRowCount(0))

        row.addWidget(btn)
        row.addWidget(clr)
        row.addStretch()

        v.addLayout(row)
        return box

    def on_kwargs_changed(self, kwargs: Dict[str, Any]):
        self.generation_kwargs = dict(kwargs)

    def browse_output(self):
        d = QFileDialog.getExistingDirectory(self)
        if d:
            self.out.setText(d)

    def browse_audio(self):
        f, _ = QFileDialog.getOpenFileName(self)
        if f:
            self.ref_audio.setText(f)

    def load_text(self):
        f, _ = QFileDialog.getOpenFileName(self)
        if f:
            self.ref_text.setText(f)
            with open(f, "r", encoding="utf-8") as fh:
                self.reference_text = fh.read()

    def load_model(self):
        self.model_status_label.setText("loading...")
        self.backend.load_model(ModelConfig(self.model.text(), self.device.currentText()))
        self.model_status_label.setText("loaded")

    def build_prompt(self):
        self.prompt_status_label.setText("building...")
        self.backend.ensure_prompt(self.ref_audio.text(), self.reference_text)
        self.prompt_status_label.setText("built")

    def run_batch(self):
        scripts = [s.strip() for s in self.script.toPlainText().split("\n\n") if s.strip()]
        out = self.out.text()

        for i, text in enumerate(scripts[: self.batch.value()], 1):
            path = os.path.join(out, f"voice_clone_{i:03d}.wav")

            results = self.backend.generate_voice_clone_batch(
                [BatchItem(i, text, self.language.currentText(), path)],
                self.ref_audio.text(),
                self.reference_text,
                self.tuning_panel.get_generation_kwargs(),
            )

            for trial, fname, dur in results:
                r = self.table.rowCount()
                self.table.insertRow(r)
                self.table.setItem(r, 0, QTableWidgetItem(str(trial)))
                self.table.setItem(r, 1, QTableWidgetItem(os.path.basename(fname)))
                self.table.setItem(r, 2, QTableWidgetItem(f"{dur:.2f}"))

    def open_folder(self):
        QDesktopServices.openUrl(QUrl.fromLocalFile(self.out.text()))


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
