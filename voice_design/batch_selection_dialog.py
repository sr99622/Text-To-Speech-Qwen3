import os

from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QVBoxLayout,
)


class BatchSelectionDialog(QDialog):
    def __init__(self, batch_dirs, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Batch Folder")
        self.resize(700, 450)
        self.selected_batch_path = None

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Available batch folders"))

        self.list_widget = QListWidget()
        layout.addWidget(self.list_widget, 1)

        for batch_dir in batch_dirs:
            item = QListWidgetItem(os.path.basename(batch_dir))
            item.setData(256, batch_dir)
            self.list_widget.addItem(item)

        self.list_widget.itemDoubleClicked.connect(self.accept_selection)

        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept_selection)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        if self.list_widget.count() > 0:
            self.list_widget.setCurrentRow(0)

    def accept_selection(self):
        item = self.list_widget.currentItem()
        if item is None:
            QMessageBox.information(self, "No Selection", "Please select a batch folder.")
            return

        self.selected_batch_path = item.data(256)
        self.accept()
