import json
import os
import sys
import traceback

import soundfile

from PyQt6.QtCore import QThread, QUrl, pyqtSlot
from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QDialog,
)

from batch_selection_dialog import BatchSelectionDialog
from qwen_tts_workers import MODEL_ID, GenerateWorker, ModelLoadWorker


def format_seconds(seconds: float) -> str:
    return f"{seconds:.2f}"


def safe_read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.model = None
        self.load_thread = None
        self.load_worker = None
        self.generate_thread = None
        self.generate_worker = None

        self.output_dir = os.path.join(os.getcwd(), "outputs")
        os.makedirs(self.output_dir, exist_ok=True)

        self.results = []

        self.player = QMediaPlayer(self)
        self.audio_output = QAudioOutput(self)
        self.player.setAudioOutput(self.audio_output)
        self.audio_output.setVolume(1.0)

        self.setWindowTitle("Qwen3 TTS Voice Design Trials")
        self.resize(1000, 850)

        self._build_ui()
        self._set_busy(True)
        self.append_log("Application started.")
        self.load_model()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QVBoxLayout(central)

        controls_group = QGroupBox("Controls")
        controls_layout = QFormLayout()
        controls_group.setLayout(controls_layout)

        self.language_combo = QComboBox()
        self.language_combo.setEnabled(False)

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 128)
        self.batch_spin.setValue(4)

        self.output_dir_label = QLabel(self.output_dir)
        self.output_dir_label.setWordWrap(True)

        controls_layout.addRow("Language:", self.language_combo)
        controls_layout.addRow("Batch size:", self.batch_spin)
        controls_layout.addRow("Output directory:", self.output_dir_label)

        input_group = QGroupBox("Voice Design Input")
        input_layout = QVBoxLayout()
        input_group.setLayout(input_layout)

        self.text_edit = QPlainTextEdit()
        self.text_edit.setPlaceholderText("Enter the text to speak here...")

        self.instruct_edit = QPlainTextEdit()
        self.instruct_edit.setPlaceholderText("Enter the voice instruction here...")

        input_layout.addWidget(QLabel("Text"))
        input_layout.addWidget(self.text_edit)
        input_layout.addWidget(QLabel("Instruct"))
        input_layout.addWidget(self.instruct_edit)

        button_layout = QHBoxLayout()

        self.run_button = QPushButton("Run")
        self.run_button.setEnabled(False)
        self.run_button.clicked.connect(self.run_generation)

        button_layout.addWidget(self.run_button)
        button_layout.addStretch()

        results_group = QGroupBox("Generated Files")
        results_layout = QVBoxLayout()
        results_group.setLayout(results_layout)

        self.results_table = QTableWidget(0, 3)
        self.results_table.setHorizontalHeaderLabels(
            ["Trial", "Filename", "Duration (sec)"]
        )
        self.results_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.results_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.results_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.results_table.setAlternatingRowColors(True)
        self.results_table.cellDoubleClicked.connect(self.play_selected_file)

        header = self.results_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)

        playback_layout = QHBoxLayout()

        self.play_button = QPushButton("Play Selected")
        self.play_button.clicked.connect(self.play_selected_file)

        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_playback)

        self.open_folder_button = QPushButton("Open Batch Folder")
        self.open_folder_button.clicked.connect(self.open_batch_folder_dialog)

        self.clear_list_button = QPushButton("Clear List")
        self.clear_list_button.clicked.connect(self.clear_results)

        playback_layout.addWidget(self.play_button)
        playback_layout.addWidget(self.stop_button)
        playback_layout.addWidget(self.open_folder_button)
        playback_layout.addWidget(self.clear_list_button)
        playback_layout.addStretch()

        self.now_playing_label = QLabel("Now playing: none")
        self.now_playing_label.setWordWrap(True)

        results_layout.addWidget(self.results_table)
        results_layout.addLayout(playback_layout)
        results_layout.addWidget(self.now_playing_label)

        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout()
        log_group.setLayout(log_layout)

        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        log_layout.addWidget(self.log_edit)

        main_layout.addWidget(controls_group)
        main_layout.addWidget(input_group, 1)
        main_layout.addLayout(button_layout)
        main_layout.addWidget(results_group, 1)
        main_layout.addWidget(log_group, 1)

    def append_log(self, message: str):
        self.log_edit.append(message)

    def _set_busy(self, busy: bool):
        self.run_button.setEnabled(not busy and self.model is not None)
        self.language_combo.setEnabled(not busy and self.model is not None)
        self.batch_spin.setEnabled(not busy)
        self.text_edit.setEnabled(not busy)
        self.instruct_edit.setEnabled(not busy)

    def load_model(self):
        self._set_busy(True)
        self.append_log(f"Loading model: {MODEL_ID}")

        self.load_thread = QThread()
        self.load_worker = ModelLoadWorker()
        self.load_worker.moveToThread(self.load_thread)

        self.load_thread.started.connect(self.load_worker.run)
        self.load_worker.status.connect(self.append_log)
        self.load_worker.finished.connect(self.on_model_loaded)
        self.load_worker.error.connect(self.on_model_load_error)

        self.load_worker.finished.connect(self.load_thread.quit)
        self.load_worker.error.connect(self.load_thread.quit)

        self.load_thread.finished.connect(self.load_worker.deleteLater)
        self.load_thread.finished.connect(self.load_thread.deleteLater)

        self.load_thread.start()

    @pyqtSlot(object, list)
    def on_model_loaded(self, model, languages):
        self.model = model

        self.language_combo.clear()
        self.language_combo.addItems(languages)

        if languages:
            self.append_log(f"Model loaded successfully. {len(languages)} language(s) available.")
        else:
            self.append_log("Model loaded successfully, but no languages were returned.")

        self._set_busy(False)

    @pyqtSlot(str)
    def on_model_load_error(self, error_text):
        self.model = None
        self._set_busy(False)

        self.append_log("Model load failed.")
        self.append_log(error_text)

        QMessageBox.critical(self, "Model Load Error", error_text)

    def list_batch_dirs(self):
        if not os.path.isdir(self.output_dir):
            return []

        batch_dirs = []
        for name in os.listdir(self.output_dir):
            path = os.path.join(self.output_dir, name)
            if os.path.isdir(path):
                batch_dirs.append(path)

        batch_dirs.sort(reverse=True)
        return batch_dirs

    def load_batch_folder(self, batch_dir: str):
        manifest_path = os.path.join(batch_dir, "manifest.json")
        text_path = os.path.join(batch_dir, "text.txt")
        instruct_path = os.path.join(batch_dir, "instruct.txt")

        if not os.path.exists(text_path) or not os.path.exists(instruct_path):
            raise FileNotFoundError(
                f"Missing required files in batch folder:\n{batch_dir}\n\nExpected text.txt and instruct.txt"
            )

        text_value = safe_read_text_file(text_path)
        instruct_value = safe_read_text_file(instruct_path)

        loaded_results = []
        manifest = None

        if os.path.exists(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)

        if manifest and isinstance(manifest.get("files"), list):
            for item in manifest["files"]:
                path = item["path"]
                if not os.path.isabs(path):
                    path = os.path.join(batch_dir, os.path.basename(path))

                loaded_results.append(
                    {
                        "trial": item.get("trial", len(loaded_results) + 1),
                        "path": path,
                        "filename": os.path.basename(path),
                        "duration_sec": float(item.get("duration_sec", 0.0)),
                        "sample_rate": int(item.get("sample_rate", 0)),
                        "batch_dir": batch_dir,
                        "batch_name": os.path.basename(batch_dir),
                    }
                )
        else:
            wav_files = [
                os.path.join(batch_dir, name)
                for name in sorted(os.listdir(batch_dir))
                if name.lower().endswith(".wav")
            ]

            for idx, wav_path in enumerate(wav_files, start=1):
                info = soundfile.info(wav_path)
                duration_sec = 0.0
                if info.samplerate:
                    duration_sec = info.frames / float(info.samplerate)

                loaded_results.append(
                    {
                        "trial": idx,
                        "path": wav_path,
                        "filename": os.path.basename(wav_path),
                        "duration_sec": duration_sec,
                        "sample_rate": int(info.samplerate),
                        "batch_dir": batch_dir,
                        "batch_name": os.path.basename(batch_dir),
                    }
                )

        self.stop_playback()
        self.results.clear()
        self.results_table.setRowCount(0)

        self.text_edit.setPlainText(text_value)
        self.instruct_edit.setPlainText(instruct_value)

        if manifest:
            language = manifest.get("language", "")
            index = self.language_combo.findText(language)
            if index >= 0:
                self.language_combo.setCurrentIndex(index)

            batch_size = int(manifest.get("batch_size", len(loaded_results) or 1))
            self.batch_spin.setValue(batch_size)

        self.results.extend(loaded_results)
        for item in loaded_results:
            self.add_result_row(item)

        if loaded_results:
            self.results_table.selectRow(0)

        self.append_log(f"Loaded batch folder: {batch_dir}")

    def run_generation(self):
        if self.model is None:
            QMessageBox.warning(self, "Model Not Ready", "The model is not loaded yet.")
            return

        text = self.text_edit.toPlainText().strip()
        instruct = self.instruct_edit.toPlainText().strip()
        language = self.language_combo.currentText().strip()
        batch_size = self.batch_spin.value()

        if not text:
            QMessageBox.warning(self, "Missing Text", "Please enter the text to speak.")
            return

        if not instruct:
            QMessageBox.warning(self, "Missing Instruct", "Please enter the voice instruction.")
            return

        if not language:
            QMessageBox.warning(self, "Missing Language", "Please select a language.")
            return

        self._set_busy(True)
        self.append_log(
            f"Starting batch run: batch_size={batch_size}, language={language}"
        )

        self.generate_thread = QThread()
        self.generate_worker = GenerateWorker(
            model=self.model,
            text=text,
            language=language,
            instruct=instruct,
            batch_size=batch_size,
            output_dir=self.output_dir,
        )
        self.generate_worker.moveToThread(self.generate_thread)

        self.generate_thread.started.connect(self.generate_worker.run)
        self.generate_worker.status.connect(self.append_log)
        self.generate_worker.finished.connect(self.on_generation_finished)
        self.generate_worker.error.connect(self.on_generation_error)

        self.generate_worker.finished.connect(self.generate_thread.quit)
        self.generate_worker.error.connect(self.generate_thread.quit)

        self.generate_thread.finished.connect(self.generate_worker.deleteLater)
        self.generate_thread.finished.connect(self.generate_thread.deleteLater)

        self.generate_thread.start()

    @pyqtSlot(list)
    def on_generation_finished(self, new_results):
        self.results.clear()
        self.results_table.setRowCount(0)

        self.results.extend(new_results)
        for item in new_results:
            self.add_result_row(item)

        self.append_log(f"Generation complete. Added {len(new_results)} file(s).")
        self._set_busy(False)

        if new_results:
            self.results_table.selectRow(0)

    @pyqtSlot(str)
    def on_generation_error(self, error_text):
        self._set_busy(False)
        self.append_log("Generation failed.")
        self.append_log(error_text)

        QMessageBox.critical(self, "Generation Error", error_text)

    def add_result_row(self, result_item: dict):
        row = self.results_table.rowCount()
        self.results_table.insertRow(row)

        trial_item = QTableWidgetItem(str(result_item["trial"]))
        filename_item = QTableWidgetItem(result_item["filename"])
        duration_item = QTableWidgetItem(format_seconds(result_item["duration_sec"]))

        self.results_table.setItem(row, 0, trial_item)
        self.results_table.setItem(row, 1, filename_item)
        self.results_table.setItem(row, 2, duration_item)

    def get_selected_file_path(self):
        row = self.results_table.currentRow()
        if row < 0 or row >= len(self.results):
            return None

        return self.results[row]["path"]

    def open_batch_folder_dialog(self):
        batch_dirs = self.list_batch_dirs()
        if not batch_dirs:
            QMessageBox.information(
                self,
                "No Batch Folders",
                f"No batch folders were found in:\n\n{self.output_dir}",
            )
            return

        dialog = BatchSelectionDialog(batch_dirs, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        try:
            self.load_batch_folder(dialog.selected_batch_path)
        except Exception as e:
            QMessageBox.critical(self, "Load Batch Error", str(e))
            self.append_log(traceback.format_exc())

    def play_selected_file(self, *_args):
        file_path = self.get_selected_file_path()
        if not file_path:
            QMessageBox.information(self, "No Selection", "Please select a file to play.")
            return

        if not os.path.exists(file_path):
            QMessageBox.warning(
                self,
                "File Missing",
                f"The selected file no longer exists:\n\n{file_path}",
            )
            return

        self.player.setSource(QUrl.fromLocalFile(file_path))
        self.player.play()
        self.now_playing_label.setText(f"Now playing: {file_path}")
        self.append_log(f"Playing: {os.path.basename(file_path)}")

    def stop_playback(self):
        self.player.stop()
        self.now_playing_label.setText("Now playing: none")
        self.append_log("Playback stopped.")

    def clear_results(self):
        self.stop_playback()
        self.results.clear()
        self.results_table.setRowCount(0)
        self.append_log("Results list cleared.")


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
