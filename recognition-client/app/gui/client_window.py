from __future__ import annotations

import sys

import cv2
from PySide6.QtCore import QThread, Qt, QObject, Signal, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.core.pipeline import PipelineStatus, RecognitionPipeline


class PipelineWorker(QObject):
    frame_ready = Signal(QImage, object)
    error = Signal(str)
    finished = Signal()

    def __init__(self, pipeline: RecognitionPipeline) -> None:
        super().__init__()
        self.pipeline = pipeline
        self._running = False

    @Slot()
    def run(self) -> None:
        self._running = True

        try:
            self.pipeline.camera.open()
            while self._running:
                frame = self.pipeline.camera.read()
                processed_frame, status = self.pipeline.process_frame(frame)
                image = self._to_qimage(processed_frame)
                self.frame_ready.emit(image, status)
                QThread.msleep(15)
        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            self.pipeline.camera.release()
            self.finished.emit()

    def stop(self) -> None:
        self._running = False

    @staticmethod
    def _to_qimage(frame) -> QImage:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb_frame.shape
        bytes_per_line = channels * width
        return QImage(
            rgb_frame.data,
            width,
            height,
            bytes_per_line,
            QImage.Format.Format_RGB888,
        ).copy()


class ClientWindow(QMainWindow):
    def __init__(self, pipeline: RecognitionPipeline, window_title: str = "Attendance Client") -> None:
        self.app = QApplication.instance()
        self._owns_app = self.app is None
        if self.app is None:
            self.app = QApplication(sys.argv)

        super().__init__()
        self.pipeline = pipeline
        self.setWindowTitle(window_title)
        self.resize(1280, 760)
        self.setMinimumSize(1100, 680)

        self.is_running = False
        self.worker_thread: QThread | None = None
        self.worker: PipelineWorker | None = None

        self.video_label = QLabel("Camera preview will appear here")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(860, 640)
        self.video_label.setStyleSheet(
            "background:#1a1f1c; color:#f5f1ea; border-radius:12px; font: 15px 'Segoe UI';"
        )

        self.camera_value = QLabel("Stopped")
        self.backend_value = QLabel("No requests yet")
        self.faces_value = QLabel("0")
        self.last_match_value = QLabel("unknown")
        self.confirmed_value = QLabel("-")
        self.reason_value = QLabel("idle")
        self.provider_value = QLabel("unknown")
        self.processing_value = QLabel("0.0 ms")
        self.events_list = QListWidget()
        self.events_list.setStyleSheet(
            "background:#f9f4ec; color:#2b2b2b; border:none; padding:6px; font: 11px 'Consolas';"
        )

        self._build_layout()
        self._apply_styles()

    def _build_layout(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)

        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(16, 16, 16, 16)
        root_layout.setSpacing(12)

        header = QLabel("Attendance Recognition Client")
        header.setObjectName("headerTitle")
        root_layout.addWidget(header)

        content_layout = QHBoxLayout()
        content_layout.setSpacing(12)
        root_layout.addLayout(content_layout, 1)

        video_card = self._make_card()
        video_layout = QVBoxLayout(video_card)
        video_layout.setContentsMargins(12, 12, 12, 12)
        video_layout.addWidget(self.video_label, 1)
        content_layout.addWidget(video_card, 3)

        side_column = QVBoxLayout()
        side_column.setSpacing(12)
        content_layout.addLayout(side_column, 1)

        status_card = self._make_card()
        status_layout = QGridLayout(status_card)
        status_layout.setContentsMargins(12, 12, 12, 12)
        status_layout.setHorizontalSpacing(10)
        status_layout.setVerticalSpacing(8)

        rows = [
            ("Camera", self.camera_value),
            ("Backend", self.backend_value),
            ("Faces", self.faces_value),
            ("Last match", self.last_match_value),
            ("Confirmed", self.confirmed_value),
            ("Reason", self.reason_value),
            ("Provider", self.provider_value),
            ("Latency", self.processing_value),
        ]
        for row_index, (title, value_label) in enumerate(rows):
            title_label = QLabel(title)
            title_label.setObjectName("metaLabel")
            value_label.setObjectName("metaValue")
            value_label.setWordWrap(True)
            status_layout.addWidget(title_label, row_index, 0)
            status_layout.addWidget(value_label, row_index, 1)
        side_column.addWidget(status_card)

        controls_card = self._make_card()
        controls_layout = QGridLayout(controls_card)
        controls_layout.setContentsMargins(12, 12, 12, 12)
        controls_layout.setHorizontalSpacing(10)
        controls_layout.setVerticalSpacing(10)

        start_button = QPushButton("Start")
        start_button.clicked.connect(self.start)
        stop_button = QPushButton("Stop")
        stop_button.clicked.connect(self.stop)
        exit_button = QPushButton("Exit")
        exit_button.clicked.connect(self.close)

        controls_layout.addWidget(start_button, 0, 0)
        controls_layout.addWidget(stop_button, 0, 1)
        controls_layout.addWidget(exit_button, 1, 0, 1, 2)
        side_column.addWidget(controls_card)

        events_card = self._make_card()
        events_layout = QVBoxLayout(events_card)
        events_layout.setContentsMargins(12, 12, 12, 12)
        events_title = QLabel("Recent Attendance Events")
        events_title.setObjectName("sectionTitle")
        events_layout.addWidget(events_title)
        events_layout.addWidget(self.events_list, 1)
        side_column.addWidget(events_card, 1)

    def _apply_styles(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow {
                background: #f4efe7;
            }
            #headerTitle {
                color: #222222;
                font: 700 28px "Segoe UI";
            }
            #metaLabel {
                color: #5e5041;
                font: 11px "Segoe UI";
            }
            #metaValue {
                color: #111111;
                font: 700 14px "Segoe UI";
            }
            #sectionTitle {
                color: #111111;
                font: 700 16px "Segoe UI";
            }
            QFrame[card="true"] {
                background: #fffaf2;
                border-radius: 14px;
            }
            QPushButton {
                background: #d97b3d;
                color: white;
                border: none;
                border-radius: 10px;
                min-height: 42px;
                font: 700 12px "Segoe UI";
            }
            QPushButton:hover {
                background: #c96725;
            }
            QPushButton:pressed {
                background: #a9521a;
            }
            """
        )

    @staticmethod
    def _make_card() -> QFrame:
        frame = QFrame()
        frame.setProperty("card", True)
        return frame

    def start(self) -> None:
        if self.is_running:
            return

        self.camera_value.setText("Running")
        self.is_running = True
        self.worker_thread = QThread(self)
        self.worker = PipelineWorker(self.pipeline)
        self.worker.moveToThread(self.worker_thread)
        self.worker.frame_ready.connect(self._handle_frame)
        self.worker.error.connect(self._handle_error)
        self.worker.finished.connect(self._worker_finished)
        self.worker_thread.started.connect(self.worker.run)
        self.worker_thread.start()

    def stop(self) -> None:
        if self.worker is not None:
            self.worker.stop()
        if self.worker_thread is not None:
            self.worker_thread.quit()
            self.worker_thread.wait(2000)
            self.worker_thread = None
        self.worker = None
        self.is_running = False
        self.camera_value.setText("Stopped")

    @Slot(QImage, object)
    def _handle_frame(self, image: QImage, status: PipelineStatus) -> None:
        self._render_image(image)
        self._update_status(status)
        self._update_events()

    @Slot(str)
    def _handle_error(self, error_message: str) -> None:
        self.camera_value.setText(f"Error: {error_message}")
        self.stop()

    @Slot()
    def _worker_finished(self) -> None:
        self.is_running = False

    def _render_image(self, image: QImage) -> None:
        pixmap = QPixmap.fromImage(image)
        scaled = pixmap.scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.video_label.setPixmap(scaled)

    def _update_status(self, status: PipelineStatus) -> None:
        backend_text = status.backend_message if status.backend_ok else f"Error: {status.backend_message}"
        self.backend_value.setText(backend_text)
        self.faces_value.setText(str(status.faces_in_frame))
        self.last_match_value.setText(f"{status.last_match_label} ({status.last_match_score:.2f})")
        self.confirmed_value.setText(status.confirmed_label or "-")
        self.reason_value.setText(status.last_reason)
        self.provider_value.setText(status.provider_name)
        self.processing_value.setText(f"{status.processing_ms:.1f} ms")

    def _update_events(self) -> None:
        self.events_list.clear()
        for item in self.pipeline.recent_events:
            QListWidgetItem(item, self.events_list)

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self.stop()
        self.pipeline.shutdown()
        super().closeEvent(event)

    def run(self) -> None:
        self.show()
        if self._owns_app and self.app is not None:
            self.app.exec()
