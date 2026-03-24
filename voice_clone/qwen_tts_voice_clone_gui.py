import os
import sys
from datetime import datetime
from typing import Dict

from PyQt6.QtCore import QUrl
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
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

from qwen_tts_workers import BatchItem, ModelConfig, QwenTTSBackend
from model_tuning_panel import ModelTuningPanel


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Qwen3 TTS Voice Clone Trials")
        self.resize(1040, 900)

        self.backend = QwenTTSBackend()
        self.reference_text = ""
        self.generation_kwargs: Dict = {}

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
        self.language.setCurrentText("English")
        f.addRow("Language", self.language)

        self.batch = QSpinBox()
        self.batch.setRange(1, 64)
        self.batch.setValue(4)
        f.addRow("Batch size", self.batch)

        self.device = QComboBox()
        self.device.setEditable(True)
        self.device.addItems(["cuda:0", "cuda:1", "cpu"])
        self.device.setCurrentText("cuda:0")
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
        box = QGroupBox("Voice Clone Script")
        v = QVBoxLayout(box)

        self.script = QPlainTextEdit()
        self.script.setPlaceholderText(
            "Enter the script to generate.\n\n"
            "Batch size controls how many candidate audio files will be generated "
            "from this same script."
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

    def on_kwargs_changed(self, kwargs: Dict):
        self.generation_kwargs = dict(kwargs)

    def browse_output(self):
        d = QFileDialog.getExistingDirectory(self)
        if d:
            self.out.setText(d)

    def browse_audio(self):
        f, _ = QFileDialog.getOpenFileName(
            self,
            "Select Reference Audio",
            "",
            "Audio Files (*.wav *.flac *.mp3 *.m4a *.ogg);;All Files (*)",
        )
        if f:
            self.ref_audio.setText(f)

    def load_text(self):
        f, _ = QFileDialog.getOpenFileName(
            self,
            "Select Reference Text File",
            "",
            "Text Files (*.txt);;All Files (*)",
        )
        if f:
            self.ref_text.setText(f)
            try:
                with open(f, "r", encoding="utf-8") as fh:
                    self.reference_text = fh.read()
            except Exception as exc:
                QMessageBox.critical(self, "Error", str(exc))
                self.reference_text = ""

    def load_model(self):
        try:
            self.model_status_label.setText("loading...")
            self.backend.load_model(ModelConfig(self.model.text(), self.device.currentText()))
            self.model_status_label.setText("loaded")
        except Exception as exc:
            self.model_status_label.setText("failed")
            QMessageBox.critical(self, "Error", str(exc))

    def build_prompt(self):
        try:
            self.prompt_status_label.setText("building...")
            self.backend.ensure_prompt(self.ref_audio.text(), self.reference_text)
            self.prompt_status_label.setText("built")
        except Exception as exc:
            self.prompt_status_label.setText("failed")
            QMessageBox.critical(self, "Error", str(exc))

    def run_batch(self):
        try:
            script_text = self.script.toPlainText().strip()
            if not script_text:
                QMessageBox.warning(self, "Run Batch", "No script was provided.")
                return

            out = self.out.text().strip()
            if not out:
                QMessageBox.warning(self, "Run Batch", "Please select an output directory.")
                return

            if not self.ref_audio.text().strip():
                QMessageBox.warning(self, "Run Batch", "Please select a reference audio file.")
                return

            if not self.reference_text.strip():
                QMessageBox.warning(self, "Run Batch", "Please select a reference text file.")
                return

            os.makedirs(out, exist_ok=True)

            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            batch_items = []

            for i in range(1, self.batch.value() + 1):
                path = os.path.join(out, f"voice_clone_{stamp}_{i:03d}.wav")
                batch_items.append(
                    BatchItem(
                        i,
                        script_text,
                        self.language.currentText(),
                        path,
                    )
                )

            results = self.backend.generate_voice_clone_batch(
                batch_items,
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

        except Exception as exc:
            QMessageBox.critical(self, "Error", str(exc))

    def open_folder(self):
        path = self.out.text().strip()
        if path:
            QDesktopServices.openUrl(QUrl.fromLocalFile(path))


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
