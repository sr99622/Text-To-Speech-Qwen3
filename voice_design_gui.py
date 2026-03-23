import os
import sys
import traceback
from datetime import datetime

import torch
import soundfile as sf

from PyQt6.QtCore import QObject, QThread, pyqtSignal, pyqtSlot, QUrl
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
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
    QHeaderView,
)
from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer

from qwen_tts import Qwen3TTSModel


MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"


def format_seconds(seconds: float) -> str:
    return f"{seconds:.2f}"


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

            results = []

            for i, wav in enumerate(wavs, start=1):
                filename = os.path.join(
                    self.output_dir,
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
                    }
                )

                self.status.emit(
                    f"Saved {os.path.basename(filename)} ({duration_sec:.2f} sec)"
                )

            self.finished.emit(results)

        except Exception:
            self.error.emit(traceback.format_exc())


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.model = None
        self.load_thread = None
        self.load_worker = None
        self.generate_thread = None
        self.generate_worker = None

        self.output_dir = os.path.join(os.getcwd(), "outputs")
        self.results = []

        self.player = QMediaPlayer(self)
        self.audio_output = QAudioOutput(self)
        self.player.setAudioOutput(self.audio_output)
        self.audio_output.setVolume(1.0)

        self.setWindowTitle("Qwen3 TTS Voice Design Trials")
        self.resize(1100, 850)

        self._build_ui()
        self._set_busy(True)
        self.append_log("Application started.")
        self.load_model()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QVBoxLayout(central)

        # Controls
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

        self.browse_button = QPushButton("Browse...")
        self.browse_button.clicked.connect(self.choose_output_dir)

        output_layout = QHBoxLayout()
        output_layout.addWidget(self.output_dir_label, 1)
        output_layout.addWidget(self.browse_button)

        controls_layout.addRow("Language:", self.language_combo)
        controls_layout.addRow("Batch size:", self.batch_spin)
        controls_layout.addRow("Output directory:", output_layout)

        # Input group
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

        # Action buttons
        button_layout = QHBoxLayout()

        self.run_button = QPushButton("Run")
        self.run_button.setEnabled(False)
        self.run_button.clicked.connect(self.run_generation)

        self.reload_button = QPushButton("Reload Model")
        self.reload_button.clicked.connect(self.load_model)

        button_layout.addWidget(self.run_button)
        button_layout.addWidget(self.reload_button)
        button_layout.addStretch()

        # Results group
        results_group = QGroupBox("Generated Files")
        results_layout = QVBoxLayout()
        results_group.setLayout(results_layout)

        self.results_table = QTableWidget(0, 4)
        self.results_table.setHorizontalHeaderLabels(
            ["Trial", "Filename", "Duration (sec)", "Full Path"]
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
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)

        playback_layout = QHBoxLayout()

        self.play_button = QPushButton("Play Selected")
        self.play_button.clicked.connect(self.play_selected_file)

        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_playback)

        self.open_folder_button = QPushButton("Open Output Folder")
        self.open_folder_button.clicked.connect(self.open_output_folder_dialog)

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

        # Log
        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout()
        log_group.setLayout(log_layout)

        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)

        log_layout.addWidget(self.log_edit)

        # Assemble
        main_layout.addWidget(controls_group)
        main_layout.addWidget(input_group, 1)
        main_layout.addLayout(button_layout)
        main_layout.addWidget(results_group, 1)
        main_layout.addWidget(log_group, 1)

    def append_log(self, message: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_edit.append(f"[{timestamp}] {message}")

    def _set_busy(self, busy: bool):
        self.run_button.setEnabled(not busy and self.model is not None)
        self.reload_button.setEnabled(not busy)
        self.language_combo.setEnabled(not busy and self.model is not None)
        self.batch_spin.setEnabled(not busy)
        self.text_edit.setEnabled(not busy)
        self.instruct_edit.setEnabled(not busy)
        self.browse_button.setEnabled(not busy)

    def choose_output_dir(self):
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Output Directory",
            self.output_dir,
        )
        if directory:
            self.output_dir = directory
            self.output_dir_label.setText(directory)
            self.append_log(f"Output directory set to: {directory}")

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
        self.results.extend(new_results)

        for item in new_results:
            self.add_result_row(item)

        self.append_log(f"Generation complete. Added {len(new_results)} file(s).")
        self._set_busy(False)

        if new_results:
            first_new_row = self.results_table.rowCount() - len(new_results)
            self.results_table.selectRow(first_new_row)

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
        path_item = QTableWidgetItem(result_item["path"])

        self.results_table.setItem(row, 0, trial_item)
        self.results_table.setItem(row, 1, filename_item)
        self.results_table.setItem(row, 2, duration_item)
        self.results_table.setItem(row, 3, path_item)

    def get_selected_file_path(self):
        row = self.results_table.currentRow()
        if row < 0:
            return None

        path_item = self.results_table.item(row, 3)
        if path_item is None:
            return None

        return path_item.text()

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

    def open_output_folder_dialog(self):
        QMessageBox.information(
            self,
            "Output Folder",
            f"Output folder:\n\n{self.output_dir}",
        )


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
