# (imports unchanged)
import os
import sys
from datetime import datetime
from typing import Any, Dict, List

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QUrl
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import *

from qwen_tts_workers import (
    BatchGenerateWorker,
    BatchItem,
    ModelConfig,
    ModelLoadWorker,
    PromptBuildWorker,
    QwenTTSBackend,
)

# -----------------------------
# Help text (unchanged)
# -----------------------------
PARAM_HELP = {
    # (same as before — omitted here for brevity)
}

# -----------------------------
# Help dialog (unchanged)
# -----------------------------
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

        layout.addWidget(body)
        layout.addWidget(buttons)


# -----------------------------
# Model tuning panel (unchanged)
# -----------------------------
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

        def add(name, widget):
            f.addRow(name, self._wrap(widget, name))
            self.widgets[name] = widget

        add("do_sample", QCheckBox())
        self.widgets["do_sample"].setChecked(True)

        add("top_k", QSpinBox())
        self.widgets["top_k"].setRange(1, 500)
        self.widgets["top_k"].setValue(50)

        add("top_p", QDoubleSpinBox())
        self.widgets["top_p"].setRange(0, 1)
        self.widgets["top_p"].setValue(1.0)

        add("temperature", QDoubleSpinBox())
        self.widgets["temperature"].setRange(0.05, 2)
        self.widgets["temperature"].setValue(0.9)

        add("repetition_penalty", QDoubleSpinBox())
        self.widgets["repetition_penalty"].setRange(1, 2)
        self.widgets["repetition_penalty"].setValue(1.05)

        add("max_new_tokens", QSpinBox())
        self.widgets["max_new_tokens"].setRange(1, 32768)
        self.widgets["max_new_tokens"].setValue(2048)

        for w in self.widgets.values():
            if isinstance(w, (QSpinBox, QDoubleSpinBox, QCheckBox)):
                w.valueChanged.connect(self.update_kwargs) if hasattr(w, "valueChanged") else w.toggled.connect(self.update_kwargs)

        return box

    def _build_sub(self):
        box = QGroupBox("Sub-Talker")
        f = QFormLayout(box)

        def add(name, widget):
            f.addRow(name, self._wrap(widget, name))
            self.widgets[name] = widget

        add("subtalker_dosample", QCheckBox())
        self.widgets["subtalker_dosample"].setChecked(True)

        add("subtalker_top_k", QSpinBox())
        self.widgets["subtalker_top_k"].setRange(1, 500)
        self.widgets["subtalker_top_k"].setValue(50)

        add("subtalker_top_p", QDoubleSpinBox())
        self.widgets["subtalker_top_p"].setRange(0, 1)
        self.widgets["subtalker_top_p"].setValue(1.0)

        add("subtalker_temperature", QDoubleSpinBox())
        self.widgets["subtalker_temperature"].setRange(0.05, 2)
        self.widgets["subtalker_temperature"].setValue(0.9)

        for w in self.widgets.values():
            if isinstance(w, (QSpinBox, QDoubleSpinBox, QCheckBox)):
                w.valueChanged.connect(self.update_kwargs) if hasattr(w, "valueChanged") else w.toggled.connect(self.update_kwargs)

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
        pass

    def update_kwargs(self):
        self.current_kwargs = {k: w.value() if hasattr(w, "value") else w.isChecked() for k, w in self.widgets.items()}
        self.kwargs_changed.emit(dict(self.current_kwargs))

    def get_generation_kwargs(self):
        return dict(self.current_kwargs)

    def reset_defaults(self):
        for k, w in self.widgets.items():
            if isinstance(w, QCheckBox):
                w.setChecked(True)
        self.update_kwargs()


# -----------------------------
# MAIN WINDOW
# -----------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Qwen3 TTS Voice Clone Trials")
        self.resize(1040, 900)   # ← your change kept

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

        # -------- YOUR CHANGE --------
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
        # --------------------------------

        return box

    def _build_batch(self):
        box = QGroupBox("Voice Clone Batch Input")
        v = QVBoxLayout(box)

        self.script = QPlainTextEdit()
        self.script.setPlaceholderText("Enter one script per paragraph.\n\nBlank lines separate batch items.")
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

    # -----------------------------
    # Logic
    # -----------------------------
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
            with open(f, "r") as fh:
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
