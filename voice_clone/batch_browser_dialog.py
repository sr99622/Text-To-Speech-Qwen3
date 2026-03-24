import json
import os
from typing import Optional

from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QInputDialog,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
)


class BatchBrowserDialog(QDialog):
    def __init__(self, output_root: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Open Batch Folder")
        self.resize(700, 420)

        self.output_root = output_root
        self.selected_batch_dir: Optional[str] = None

        self.list_widget = QListWidget()
        self.preview = QPlainTextEdit()
        self.preview.setReadOnly(True)

        self.button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Open | QDialogButtonBox.StandardButton.Cancel
        )
        self.open_button = self.button_box.button(QDialogButtonBox.StandardButton.Open)
        self.open_button.setEnabled(False)

        self.rename_button = QPushButton("Rename Selected")
        self.rename_button.setEnabled(False)

        self.button_box.accepted.connect(self.accept_selection)
        self.button_box.rejected.connect(self.reject)
        self.rename_button.clicked.connect(self.rename_selected)

        body_layout = QHBoxLayout()
        body_layout.addWidget(self.list_widget, 1)
        body_layout.addWidget(self.preview, 2)

        bottom_row = QHBoxLayout()
        bottom_row.addWidget(self.rename_button)
        bottom_row.addStretch()
        bottom_row.addWidget(self.button_box)

        layout = QVBoxLayout(self)
        layout.addLayout(body_layout)
        layout.addLayout(bottom_row)

        self.list_widget.currentItemChanged.connect(self.update_preview)
        self.list_widget.currentItemChanged.connect(self.update_buttons_state)
        self.list_widget.itemDoubleClicked.connect(self.on_item_double_clicked)

        self.populate()

    def populate(self):
        self.list_widget.clear()

        if not os.path.isdir(self.output_root):
            self.preview.setPlainText("Output directory does not exist.")
            self.open_button.setEnabled(False)
            self.rename_button.setEnabled(False)
            return

        dirs = []
        for name in os.listdir(self.output_root):
            full = os.path.join(self.output_root, name)
            if os.path.isdir(full):
                dirs.append((os.path.getmtime(full), name, full))

        dirs.sort(reverse=True)

        for _, name, full in dirs:
            item = QListWidgetItem(name)
            item.setData(256, full)
            self.list_widget.addItem(item)

        if self.list_widget.count() > 0:
            self.list_widget.setCurrentRow(0)
        else:
            self.preview.setPlainText("No batch folders were found.")
            self.open_button.setEnabled(False)
            self.rename_button.setEnabled(False)

    def update_buttons_state(self, current: Optional[QListWidgetItem], previous: Optional[QListWidgetItem]):
        _ = previous
        has_item = current is not None
        self.open_button.setEnabled(has_item)
        self.rename_button.setEnabled(has_item)

    def update_preview(self, current: Optional[QListWidgetItem], previous: Optional[QListWidgetItem]):
        _ = previous

        if not current:
            self.preview.clear()
            return

        batch_dir = current.data(256)
        metadata_path = os.path.join(batch_dir, "batch_metadata.json")
        script_path = os.path.join(batch_dir, "script.txt")

        parts = [f"Folder: {os.path.basename(batch_dir)}"]

        if os.path.isfile(metadata_path):
            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                parts.append(f"Model: {data.get('model_name', '')}")
                parts.append(f"Language: {data.get('language', '')}")
                parts.append(f"Batch size: {data.get('batch_size', '')}")
                parts.append("")
            except Exception:
                parts.append("Metadata could not be read.")
                parts.append("")

        if os.path.isfile(script_path):
            try:
                with open(script_path, "r", encoding="utf-8") as f:
                    script = f.read().strip()
                parts.append("Script:")
                parts.append(script[:2000])
            except Exception:
                parts.append("Script could not be read.")

        self.preview.setPlainText("\n".join(parts))

    def accept_selection(self):
        item = self.list_widget.currentItem()
        if not item:
            return
        self.selected_batch_dir = item.data(256)
        self.accept()

    def on_item_double_clicked(self, item: QListWidgetItem):
        if item is None:
            return
        self.selected_batch_dir = item.data(256)
        self.accept()

    def rename_selected(self):
        item = self.list_widget.currentItem()
        if item is None:
            return

        old_path = item.data(256)
        old_name = os.path.basename(old_path)
        parent_dir = os.path.dirname(old_path)

        new_name, ok = QInputDialog.getText(
            self,
            "Rename Batch Folder",
            "New folder name:",
            text=old_name,
        )
        if not ok:
            return

        new_name = new_name.strip()
        if not new_name:
            QMessageBox.warning(self, "Rename Batch Folder", "Folder name cannot be empty.")
            return

        if new_name == old_name:
            return

        new_path = os.path.join(parent_dir, new_name)
        if os.path.exists(new_path):
            QMessageBox.warning(self, "Rename Batch Folder", "A folder with that name already exists.")
            return

        try:
            os.rename(old_path, new_path)
        except Exception as exc:
            QMessageBox.critical(self, "Rename Batch Folder", str(exc))
            return

        item.setText(new_name)
        item.setData(256, new_path)
        self.update_preview(item, None)
