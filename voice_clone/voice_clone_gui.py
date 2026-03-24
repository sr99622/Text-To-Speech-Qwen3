import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional

from PyQt6.QtCore import QThread, QUrl, Qt
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
    QInputDialog,
)

try:
    from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer

    QT_MULTIMEDIA_AVAILABLE = True
except Exception:
    QT_MULTIMEDIA_AVAILABLE = False

from qwen_tts_workers import (
    BatchGenerateWorker,
    BatchItem,
    ModelConfig,
    ModelLoadWorker,
    PromptBuildWorker,
    QwenTTSBackend,
)
from model_tuning_panel import ModelTuningPanel
from batch_browser_dialog import BatchBrowserDialog


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Qwen3 TTS Voice Clone Trials")
        self.resize(1040, 900)

        self.backend = QwenTTSBackend()
        self.reference_text = ""
        self.generation_kwargs: Dict = {}
        self.current_batch_dir: Optional[str] = None
        self.active_threads: List[QThread] = []
        self.active_workers = []

        self.player = None
        self.audio_output = None
        if QT_MULTIMEDIA_AVAILABLE:
            self.player = QMediaPlayer(self)
            self.audio_output = QAudioOutput(self)
            self.player.setAudioOutput(self.audio_output)

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
        self.language.addItems([
            "Chinese",
            "English",
            "Japanese",
            "Korean",
            "German",
            "French",
            "Russian",
            "Portuguese",
            "Spanish",
            "Italian",
        ])
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
        self.table.verticalHeader().setVisible(False)

        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)

        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.itemDoubleClicked.connect(self.on_file_double_clicked)

        v.addWidget(self.table)

        row = QHBoxLayout()

        self.open_batch_btn = QPushButton("Open Batch Folder")
        self.open_batch_btn.clicked.connect(self.open_batch_folder_dialog)

        self.play_btn = QPushButton("Play Selected")
        self.play_btn.clicked.connect(self.play_selected_file)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_playback)

        self.rename_btn = QPushButton("Rename Selected")
        self.rename_btn.clicked.connect(self.rename_selected_file)

        self.clear_btn = QPushButton("Clear List")
        self.clear_btn.clicked.connect(lambda: self.table.setRowCount(0))

        row.addWidget(self.open_batch_btn)
        row.addWidget(self.play_btn)
        row.addWidget(self.stop_btn)
        row.addWidget(self.rename_btn)
        row.addWidget(self.clear_btn)
        row.addStretch()

        v.addLayout(row)
        return box

    def set_controls_enabled(self, enabled: bool):
        self.load_model_btn.setEnabled(enabled)
        self.build_prompt_btn.setEnabled(enabled)
        self.run_btn.setEnabled(enabled)

    def _start_worker(self, worker, thread: QThread):
        self.active_workers.append(worker)

        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(lambda: self._cleanup_thread(thread))
        worker.finished.connect(lambda: self._cleanup_worker(worker))

        self.active_threads.append(thread)
        thread.start()

    def _cleanup_thread(self, thread: QThread):
        if thread in self.active_threads:
            self.active_threads.remove(thread)

    def _cleanup_worker(self, worker):
        if worker in self.active_workers:
            self.active_workers.remove(worker)

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
            self.set_controls_enabled(False)

            config = ModelConfig(
                self.model.text().strip(),
                self.device.currentText().strip(),
            )

            thread = QThread()
            worker = ModelLoadWorker(self.backend, config)
            worker.started_work.connect(lambda: print("ModelLoadWorker started"))
            worker.error.connect(self.on_worker_error)
            worker.model_loaded.connect(self.on_model_loaded)
            self._start_worker(worker, thread)
        except Exception as exc:
            self.model_status_label.setText("failed")
            self.set_controls_enabled(True)
            QMessageBox.critical(self, "Error", str(exc))

    def on_model_loaded(self, model_name: str):
        _ = model_name
        self.model_status_label.setText("loaded")
        self.set_controls_enabled(True)

    def build_prompt(self):
        try:
            if not self.ref_audio.text().strip():
                QMessageBox.warning(self, "Build Prompt", "Please select a reference audio file.")
                return

            if not self.reference_text.strip():
                QMessageBox.warning(self, "Build Prompt", "Please select a reference text file.")
                return

            self.prompt_status_label.setText("building...")
            self.set_controls_enabled(False)

            thread = QThread()
            worker = PromptBuildWorker(
                self.backend,
                self.ref_audio.text().strip(),
                self.reference_text,
                x_vector_only_mode=False,
            )
            worker.started_work.connect(lambda: print("PromptBuildWorker started"))
            worker.error.connect(self.on_worker_error)
            worker.prompt_ready.connect(self.on_prompt_ready)
            self._start_worker(worker, thread)
        except Exception as exc:
            self.prompt_status_label.setText("failed")
            self.set_controls_enabled(True)
            QMessageBox.critical(self, "Error", str(exc))

    def on_prompt_ready(self):
        self.prompt_status_label.setText("built")
        self.set_controls_enabled(True)

    def create_unique_batch_dir(self, output_root: str) -> str:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = os.path.join(output_root, stamp)
        candidate = base
        suffix = 1
        while os.path.exists(candidate):
            candidate = f"{base}_{suffix:02d}"
            suffix += 1
        os.makedirs(candidate, exist_ok=True)
        return candidate

    def save_batch_artifacts(
        self,
        batch_dir: str,
        script_text: str,
        results: List,
    ):
        script_path = os.path.join(batch_dir, "script.txt")
        ref_text_path = os.path.join(batch_dir, "reference_text.txt")
        tuning_path = os.path.join(batch_dir, "model_tuning_params.json")
        metadata_path = os.path.join(batch_dir, "batch_metadata.json")

        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script_text)

        with open(ref_text_path, "w", encoding="utf-8") as f:
            f.write(self.reference_text)

        tuning = self.tuning_panel.get_generation_kwargs()
        with open(tuning_path, "w", encoding="utf-8") as f:
            json.dump(tuning, f, indent=2)

        metadata = {
            "model_name": self.model.text().strip(),
            "device": self.device.currentText().strip(),
            "language": self.language.currentText().strip(),
            "batch_size": self.batch.value(),
            "reference_audio": self.ref_audio.text().strip(),
            "reference_text_file": self.ref_text.text().strip(),
            "script_file": "script.txt",
            "reference_text_saved_file": "reference_text.txt",
            "tuning_file": "model_tuning_params.json",
            "files": [
                {
                    "trial": trial,
                    "filename": os.path.basename(path),
                    "duration_sec": dur,
                }
                for trial, path, dur in results
            ],
        }

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    def populate_file_table_from_results(self, results: List):
        self.table.setRowCount(0)
        for trial, fname, dur in results:
            row = self.table.rowCount()
            self.table.insertRow(row)

            filename_item = QTableWidgetItem(os.path.basename(fname))
            filename_item.setData(Qt.ItemDataRole.UserRole, fname)

            self.table.setItem(row, 0, QTableWidgetItem(str(trial)))
            self.table.setItem(row, 1, filename_item)
            self.table.setItem(row, 2, QTableWidgetItem(f"{dur:.2f}"))

    def populate_file_table_from_batch_dir(self, batch_dir: str, metadata: Dict):
        self.table.setRowCount(0)
        for file_info in metadata.get("files", []):
            row = self.table.rowCount()
            self.table.insertRow(row)

            full_path = os.path.join(batch_dir, file_info["filename"])
            filename_item = QTableWidgetItem(file_info["filename"])
            filename_item.setData(Qt.ItemDataRole.UserRole, full_path)

            self.table.setItem(row, 0, QTableWidgetItem(str(file_info.get("trial", row + 1))))
            self.table.setItem(row, 1, filename_item)
            self.table.setItem(row, 2, QTableWidgetItem(f"{float(file_info.get('duration_sec', 0.0)):.2f}"))

    def load_batch_folder(self, batch_dir: str):
        metadata_path = os.path.join(batch_dir, "batch_metadata.json")
        script_path = os.path.join(batch_dir, "script.txt")
        tuning_path = os.path.join(batch_dir, "model_tuning_params.json")
        ref_text_saved_path = os.path.join(batch_dir, "reference_text.txt")

        if not os.path.isfile(metadata_path):
            raise FileNotFoundError(f"Missing metadata file: {metadata_path}")

        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        if os.path.isfile(script_path):
            with open(script_path, "r", encoding="utf-8") as f:
                self.script.setPlainText(f.read())

        if os.path.isfile(ref_text_saved_path):
            with open(ref_text_saved_path, "r", encoding="utf-8") as f:
                self.reference_text = f.read()

        if os.path.isfile(tuning_path):
            with open(tuning_path, "r", encoding="utf-8") as f:
                tuning = json.load(f)
            self.tuning_panel.set_generation_kwargs(tuning)

        self.model.setText(metadata.get("model_name", self.model.text()))
        self.device.setCurrentText(metadata.get("device", self.device.currentText()))
        self.language.setCurrentText(metadata.get("language", self.language.currentText()))
        self.batch.setValue(int(metadata.get("batch_size", self.batch.value())))
        self.ref_audio.setText(metadata.get("reference_audio", self.ref_audio.text()))
        self.ref_text.setText(metadata.get("reference_text_file", self.ref_text.text()))
        self.current_batch_dir = batch_dir

        self.populate_file_table_from_batch_dir(batch_dir, metadata)

    def run_batch(self):
        try:
            script_text = self.script.toPlainText().strip()
            if not script_text:
                QMessageBox.warning(self, "Run Batch", "No script was provided.")
                return

            output_root = self.out.text().strip()
            if not output_root:
                QMessageBox.warning(self, "Run Batch", "Please select an output directory.")
                return

            if not self.ref_audio.text().strip():
                QMessageBox.warning(self, "Run Batch", "Please select a reference audio file.")
                return

            if not self.reference_text.strip():
                QMessageBox.warning(self, "Run Batch", "Please select a reference text file.")
                return

            os.makedirs(output_root, exist_ok=True)
            batch_dir = self.create_unique_batch_dir(output_root)

            batch_items = []
            for i in range(1, self.batch.value() + 1):
                path = os.path.join(batch_dir, f"voice_clone_{i:03d}.wav")
                batch_items.append(
                    BatchItem(
                        i,
                        script_text,
                        self.language.currentText(),
                        path,
                    )
                )

            self.current_batch_dir = batch_dir
            self.run_btn.setEnabled(False)
            self.prompt_status_label.setText("generating...")
            self.set_controls_enabled(False)

            thread = QThread()
            worker = BatchGenerateWorker(
                self.backend,
                batch_items,
                self.ref_audio.text().strip(),
                self.reference_text,
                self.tuning_panel.get_generation_kwargs(),
                x_vector_only_mode=False,
            )
            worker.started_work.connect(lambda: print("BatchGenerateWorker started"))
            worker.error.connect(self.on_worker_error)
            worker.batch_complete.connect(
                lambda results, d=batch_dir, s=script_text: self.on_batch_complete(d, s, results)
            )
            self._start_worker(worker, thread)

        except Exception as exc:
            self.run_btn.setEnabled(True)
            self.set_controls_enabled(True)
            QMessageBox.critical(self, "Error", str(exc))

    def on_batch_complete(self, batch_dir: str, script_text: str, results: List):
        self.populate_file_table_from_results(results)
        self.save_batch_artifacts(batch_dir, script_text, results)
        self.current_batch_dir = batch_dir
        self.prompt_status_label.setText("built")
        self.run_btn.setEnabled(True)
        self.set_controls_enabled(True)

    def on_worker_error(self, text: str):
        self.model_status_label.setText(
            "failed" if "from_pretrained" in text or "Model is not loaded" in text else self.model_status_label.text()
        )
        if self.prompt_status_label.text() in ("building...", "generating..."):
            self.prompt_status_label.setText("failed")
        self.run_btn.setEnabled(True)
        self.set_controls_enabled(True)
        QMessageBox.critical(self, "Error", text)

    def get_selected_file_path(self) -> Optional[str]:
        row = self.table.currentRow()
        if row < 0:
            return None
        item = self.table.item(row, 1)
        if item is None:
            return None
        return item.data(Qt.ItemDataRole.UserRole)

    def play_selected_file(self):
        if not QT_MULTIMEDIA_AVAILABLE:
            QMessageBox.warning(self, "Playback", "PyQt6 multimedia is not available.")
            return

        path = self.get_selected_file_path()
        if not path or not os.path.isfile(path):
            QMessageBox.warning(self, "Playback", "No valid audio file is selected.")
            return

        self.player.setSource(QUrl.fromLocalFile(path))
        self.player.play()

    def stop_playback(self):
        if self.player is not None:
            self.player.stop()

    def on_file_double_clicked(self, item: QTableWidgetItem):
        _ = item
        self.play_selected_file()

    def rename_selected_file(self):
        row = self.table.currentRow()
        if row < 0:
            QMessageBox.warning(self, "Rename Selected", "No file is selected.")
            return

        item = self.table.item(row, 1)
        if item is None:
            QMessageBox.warning(self, "Rename Selected", "No file is selected.")
            return

        old_path = item.data(Qt.ItemDataRole.UserRole)
        if not old_path or not os.path.isfile(old_path):
            QMessageBox.warning(self, "Rename Selected", "Selected file does not exist.")
            return

        old_name = os.path.basename(old_path)
        old_root, old_ext = os.path.splitext(old_name)

        new_base, ok = QInputDialog.getText(
            self,
            "Rename Selected File",
            "New filename:",
            text=old_root,
        )
        if not ok:
            return

        new_base = new_base.strip()
        if not new_base:
            QMessageBox.warning(self, "Rename Selected", "Filename cannot be empty.")
            return

        new_name = f"{new_base}{old_ext}"
        if new_name == old_name:
            return

        new_path = os.path.join(os.path.dirname(old_path), new_name)
        if os.path.exists(new_path):
            QMessageBox.warning(self, "Rename Selected", "A file with that name already exists.")
            return

        try:
            os.rename(old_path, new_path)
        except Exception as exc:
            QMessageBox.critical(self, "Rename Selected", str(exc))
            return

        item.setText(new_name)
        item.setData(Qt.ItemDataRole.UserRole, new_path)

        if self.current_batch_dir:
            self.update_batch_metadata_after_file_rename(self.current_batch_dir, old_name, new_name)

    def update_batch_metadata_after_file_rename(self, batch_dir: str, old_name: str, new_name: str):
        metadata_path = os.path.join(batch_dir, "batch_metadata.json")
        if not os.path.isfile(metadata_path):
            return

        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            for file_info in metadata.get("files", []):
                if file_info.get("filename") == old_name:
                    file_info["filename"] = new_name

            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
        except Exception:
            pass

    def open_batch_folder_dialog(self):
        output_root = self.out.text().strip()
        if not output_root or not os.path.isdir(output_root):
            QMessageBox.warning(self, "Open Batch Folder", "Output directory does not exist.")
            return

        dlg = BatchBrowserDialog(output_root, self)
        if dlg.exec() and dlg.selected_batch_dir:
            try:
                self.load_batch_folder(dlg.selected_batch_dir)
            except Exception as exc:
                QMessageBox.critical(self, "Error", str(exc))


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
