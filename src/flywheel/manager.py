"""AegisVision - Privacy-First Data Flywheel Manager.
Handles image storage, PII redaction, metadata logging, and GDPR erasure requests.
"""
import os
import sqlite3
import uuid
import shutil
import time
import cv2
import numpy as np
from datetime import datetime

DRIVE_ROOT = "/content/drive/MyDrive/AegisVision/Data_Flywheel"
DB_PATH = "/content/drive/MyDrive/AegisVision/data/flywheel/aegis_flywheel.db"

class FlywheelManager:
    def __init__(self):
        os.makedirs(DRIVE_ROOT, exist_ok=True)
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        self._init_db()
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

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
        if img is None: return False
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
        partition_dir = os.path.join(DRIVE_ROOT, date_str)
        os.makedirs(partition_dir, exist_ok=True)
        
        dest_image_path = os.path.join(partition_dir, f"{record_id}.jpg")
        shutil.copy2(image_path, dest_image_path)
        os.system("sync")
        time.sleep(1) # Shortened sleep for better UX
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        try:
            cursor.execute("""
                INSERT INTO feedback_records 
                (id, image_path, detected_class, confidence, risk_score, human_feedback, consent_granted, pii_redacted)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (record_id, dest_image_path, detected_class, confidence, risk_score, human_feedback, int(consent_granted), int(pii_found)))
            cursor.execute("INSERT INTO audit_log (record_id, action) VALUES (?, ?)", (record_id, "CREATED"))
            conn.commit()
            print(f"✅ [Flywheel] Saved {record_id} with privacy safeguards.")
            return record_id
        except sqlite3.IntegrityError:
            print(f"⚠️ [Flywheel] Image already exists. Skipped.")
            return None
        finally:
            conn.close()

    def request_erasure(self, record_id):
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT image_path FROM feedback_records WHERE id = ?", (record_id,))
        result = cursor.fetchone()
        if result:
            image_path = result[0]
            if os.path.exists(image_path):
                os.remove(image_path)
            cursor.execute("UPDATE feedback_records SET image_path = 'DELETED', consent_granted = 0 WHERE id = ?", (record_id,))
            cursor.execute("INSERT INTO audit_log (record_id, action) VALUES (?, ?)", (record_id, "ERASED"))
            conn.commit()
            print(f"✅ GDPR Erasure completed for {record_id}.")
        conn.close()

    def get_pending_training_data(self, limit=50):
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT id, image_path, detected_class, human_feedback FROM feedback_records WHERE is_processed_for_training = 0 AND consent_granted = 1 AND image_path != 'DELETED' LIMIT ?", (limit,))
        records = cursor.fetchall()
        conn.close()
        return records
