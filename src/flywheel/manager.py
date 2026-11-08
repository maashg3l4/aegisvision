"""AegisVision - Privacy-First Data Flywheel Manager.
Handles image storage, PII redaction, metadata logging, and GDPR erasure requests.
"""
import os
import json
import sqlite3
import uuid
import cv2
import numpy as np
from datetime import datetime
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

import sys
# Detect if we are running in Streamlit Cloud or Colab
if 'google.colab' in sys.modules:
    DRIVE_ROOT = "/content/drive/MyDrive/AegisVision/Data_Flywheel"
    DB_PATH = "/content/drive/MyDrive/AegisVision/data/flywheel/aegis_flywheel.db"
else:
    # Streamlit Cloud fallback (ephemeral storage)
    DRIVE_ROOT = "/tmp/AegisVision_Data_Flywheel"
    DB_PATH = "/tmp/aegis_flywheel.db"

class FlywheelManager:
    STORAGE_BUCKET = "defect-images"

    def __init__(self, supabase_url=None, supabase_key=None):
        self.supabase_url = supabase_url or os.environ.get("SUPABASE_URL")
        self.supabase_key = supabase_key or os.environ.get("SUPABASE_KEY")
        os.makedirs(DRIVE_ROOT, exist_ok=True)
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        self._init_db()
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def _storage_request(self, method, object_path, body=None, content_type=None):
        if method not in ("POST", "DELETE"):
            raise ValueError(f"Unsupported Supabase Storage method: {method}")
        if not self.supabase_url or not self.supabase_key:
            raise RuntimeError(
                "Supabase storage is not configured. Set SUPABASE_URL and SUPABASE_KEY."
            )

        encoded_path = quote(object_path, safe="/")
        if method == "POST":
            endpoint = f"{self.supabase_url.rstrip('/')}/storage/v1/object/{self.STORAGE_BUCKET}/{encoded_path}"
        else:
            endpoint = f"{self.supabase_url.rstrip('/')}/storage/v1/object/{self.STORAGE_BUCKET}"
        headers = {
            "apikey": self.supabase_key,
            "Authorization": f"Bearer {self.supabase_key}",
        }
        if content_type:
            headers["Content-Type"] = content_type
        if method == "DELETE":
            headers["Content-Type"] = "application/json"
            body = json.dumps({"prefixes": [object_path]}).encode("utf-8")

        request = Request(endpoint, data=body, headers=headers, method=method)
        try:
            with urlopen(request, timeout=30) as response:
                response.read()
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Supabase Storage {method} failed with HTTP {error.code}: {detail}"
            ) from error

    def _upload_image(self, image_path, object_path):
        with open(image_path, "rb") as image_file:
            image_bytes = image_file.read()
        self._storage_request(
            "POST", object_path, body=image_bytes, content_type="image/jpeg"
        )

    def _init_db(self):
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feedback_records (
                id TEXT PRIMARY KEY,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                image_path TEXT UNIQUE,
                detected_class TEXT,
                confidence REAL,
                risk_score REAL,
                human_feedback TEXT,
                is_processed_for_training INTEGER DEFAULT 0,
                consent_granted INTEGER DEFAULT 1,
                pii_redacted INTEGER DEFAULT 0
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS audit_log (
                log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                record_id TEXT,
                action TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()

    def _redact_pii(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Unable to read feedback image for privacy processing: {image_path}")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        if len(faces) > 0:
            for (x, y, w, h) in faces:
                roi = img[y:y+h, x:x+w]
                img[y:y+h, x:x+w] = cv2.GaussianBlur(roi, (99, 99), 30)
            cv2.imwrite(image_path, img)
            print(f"🔒 [Privacy] PII redacted in {os.path.basename(image_path)}")
            return True
        return False

    def save_feedback(self, image_path, detected_class, confidence, risk_score, human_feedback, consent_granted=True):
        pii_found = self._redact_pii(image_path)
        record_id = str(uuid.uuid4())
        date_str = datetime.now().strftime("%Y-%m-%d")
        object_path = f"{date_str}/{record_id}.jpg"
        self._upload_image(image_path, object_path)

        conn = None
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO feedback_records 
                (id, image_path, detected_class, confidence, risk_score, human_feedback, consent_granted, pii_redacted)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (record_id, object_path, detected_class, confidence, risk_score, human_feedback, int(consent_granted), int(pii_found)))
            cursor.execute("INSERT INTO audit_log (record_id, action) VALUES (?, ?)", (record_id, "CREATED"))
            conn.commit()
            return object_path
        except Exception as error:
            try:
                self._storage_request("DELETE", object_path)
            except Exception as cleanup_error:
                raise RuntimeError(
                    f"Flywheel metadata save failed and uploaded image cleanup failed: {cleanup_error}"
                ) from error
            raise
        finally:
            if conn is not None:
                conn.close()

    def request_erasure(self, record_id):
        conn = sqlite3.connect(DB_PATH)
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT image_path FROM feedback_records WHERE id = ?", (record_id,))
            result = cursor.fetchone()
            if result:
                image_path = result[0]
                if os.path.isabs(image_path):
                    if os.path.exists(image_path):
                        os.remove(image_path)
                elif image_path != "DELETED":
                    self._storage_request("DELETE", image_path)
                cursor.execute("UPDATE feedback_records SET image_path = 'DELETED', consent_granted = 0 WHERE id = ?", (record_id,))
                cursor.execute("INSERT INTO audit_log (record_id, action) VALUES (?, ?)", (record_id, "ERASED"))
                conn.commit()
                print(f"✅ GDPR Erasure completed for {record_id}.")
        finally:
            conn.close()

    def get_pending_training_data(self, limit=50):
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT id, image_path, detected_class, human_feedback FROM feedback_records WHERE is_processed_for_training = 0 AND consent_granted = 1 AND image_path != 'DELETED' LIMIT ?", (limit,))
        records = cursor.fetchall()
        conn.close()
        return records
