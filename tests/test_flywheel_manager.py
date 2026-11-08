import os
import sqlite3
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

# Keep these storage tests runnable without installing the image-processing stack.
sys.modules.setdefault(
    "cv2",
    SimpleNamespace(data=SimpleNamespace(haarcascades=""), CascadeClassifier=lambda _: None),
)

from src.flywheel.manager import FlywheelManager


class FlywheelManagerTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.db_path = os.path.join(self.temp_dir.name, "feedback.sqlite")
        self.image_root = os.path.join(self.temp_dir.name, "images")
        self.source_path = os.path.join(self.temp_dir.name, "source.jpg")
        with open(self.source_path, "wb") as image_file:
            image_file.write(b"redacted demo image bytes")

        self.patches = [
            mock.patch("builtins.print"),
            mock.patch("src.flywheel.manager.DB_PATH", self.db_path),
            mock.patch("src.flywheel.manager.DRIVE_ROOT", self.image_root),
            mock.patch(
                "src.flywheel.manager.FlywheelManager._redact_pii",
                return_value=False,
            ),
        ]
        for patcher in self.patches:
            patcher.start()
            self.addCleanup(patcher.stop)

        self.manager = FlywheelManager(
            supabase_url="https://example.supabase.co", supabase_key="test-key"
        )

    @mock.patch.dict(
        os.environ, {"SUPABASE_URL": "https://example.supabase.co", "SUPABASE_KEY": "test-key"}
    )
    @mock.patch("src.flywheel.manager.urlopen")
    def test_save_feedback_uploads_image_to_configured_bucket(self, mocked_urlopen):
        response = mock.MagicMock()
        response.__enter__.return_value = response
        mocked_urlopen.return_value = response

        object_path = self.manager.save_feedback(
            self.source_path, "scratches", 0.9, 0.7, "CORRECT"
        )

        request = mocked_urlopen.call_args.args[0]
        self.assertEqual(request.get_method(), "POST")
        endpoint_path = request.full_url.split("/storage/v1/object/", 1)[1]
        self.assertEqual(endpoint_path, f"defect-images/{object_path}")
        self.assertEqual(
            request.get_header("Authorization"),
            "Bearer test-key",
        )
        self.assertEqual(request.data, b"redacted demo image bytes")
        self.assertEqual(request.get_header("Content-type"), "image/jpeg")
        self.assertEqual(mocked_urlopen.call_args.kwargs["timeout"], 30)

        record_id = os.path.basename(object_path).removesuffix(".jpg")
        connection = sqlite3.connect(self.db_path)
        try:
            stored_path = connection.execute(
                "SELECT image_path FROM feedback_records WHERE id = ?", (record_id,)
            ).fetchone()[0]
        finally:
            connection.close()
        self.assertEqual(stored_path, object_path)
        self.assertTrue(stored_path.startswith("20"))

    @mock.patch.dict(
        os.environ, {"SUPABASE_URL": "https://example.supabase.co", "SUPABASE_KEY": "test-key"}
    )
    @mock.patch("src.flywheel.manager.urlopen", side_effect=OSError("upload failed"))
    def test_upload_error_is_raised_without_saving_metadata(self, mocked_urlopen):
        with self.assertRaisesRegex(OSError, "upload failed"):
            self.manager.save_feedback(
                self.source_path, "scratches", 0.9, 0.7, "CORRECT"
            )

        connection = sqlite3.connect(self.db_path)
        try:
            count = connection.execute(
                "SELECT COUNT(*) FROM feedback_records"
            ).fetchone()[0]
        finally:
            connection.close()
        self.assertEqual(count, 0)
        mocked_urlopen.assert_called_once()

    @mock.patch("src.flywheel.manager.urlopen")
    def test_redacted_image_bytes_are_uploaded_and_marked_in_metadata(self, mocked_urlopen):
        response = mock.MagicMock()
        response.__enter__.return_value = response
        mocked_urlopen.return_value = response

        def redact_image(path):
            with open(path, "wb") as image_file:
                image_file.write(b"redacted image")
            return True

        with mock.patch.object(
            self.manager, "_redact_pii", side_effect=redact_image
        ):
            object_path = self.manager.save_feedback(
                self.source_path, "scratches", 0.9, 0.7, "CORRECT"
            )

        self.assertEqual(mocked_urlopen.call_args.args[0].data, b"redacted image")
        connection = sqlite3.connect(self.db_path)
        try:
            pii_redacted = connection.execute(
                "SELECT pii_redacted FROM feedback_records WHERE image_path = ?",
                (object_path,),
            ).fetchone()[0]
        finally:
            connection.close()
        self.assertEqual(pii_redacted, 1)

    @mock.patch.dict(os.environ, {}, clear=True)
    def test_missing_supabase_configuration_fails_before_upload(self):
        manager = FlywheelManager(supabase_url=None, supabase_key=None)
        with mock.patch("src.flywheel.manager.urlopen") as mocked_urlopen:
            with self.assertRaisesRegex(RuntimeError, "SUPABASE_URL and SUPABASE_KEY"):
                manager.save_feedback(
                    self.source_path, "scratches", 0.9, 0.7, "CORRECT"
                )
        mocked_urlopen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
