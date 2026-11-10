import os
import uuid
import streamlit as st
from supabase import create_client, Client
import cv2
import time

# Standard, safe initialization
url: str = st.secrets["SUPABASE_URL"]
key: str = st.secrets["SUPABASE_KEY"]
supabase: Client = create_client(url, key)

BUCKET_NAME = "aegis-flywheel-images"

class FlywheelManager:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

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
            return True
        return False

    def save_feedback(self, image_path, detected_class, confidence, risk_score, human_feedback, consent_granted=True):
        pii_found = self._redact_pii(image_path)
        record_id = str(uuid.uuid4())
        storage_path = f"feedback/{record_id}.jpg"

        # Robust Upload with Retry Logic for Streamlit Cloud
        max_retries = 3
        upload_success = False
        last_error = None

        for attempt in range(max_retries):
            try:
                with open(image_path, 'rb') as f:
                    supabase.storage.from_(BUCKET_NAME).upload(
                        path=storage_path,
                        file=f,
                        file_options={"content-type": "image/jpeg"}
                    )
                upload_success = True
                break  # Exit loop if successful
            except Exception as e:
                last_error = e
                print(f"⚠️ Upload attempt {attempt + 1} failed. Retrying in 2 seconds...")
                time.sleep(2) # Wait before retrying

        if not upload_success:
            # If all retries fail, raise the error to the UI
            raise Exception(f"Failed to upload image to Supabase after 3 attempts. Last error: {last_error}")

        # Insert metadata into Database
        supabase.table('feedback_records').insert({
            "id": record_id,
            "image_path": storage_path,
            "detected_class": detected_class,
            "confidence": confidence,
            "risk_score": risk_score,
            "human_feedback": human_feedback,
            "consent_granted": consent_granted,
            "pii_redacted": pii_found
        }).execute()

        # Log the action
        supabase.table('audit_log').insert({
            "record_id": record_id,
            "action": "CREATED"
        }).execute()

        print(f"✅ [Flywheel] Saved {record_id} to Supabase.")
        return record_id

    def request_erasure(self, record_id):
        res = supabase.table('feedback_records').select('image_path').eq('id', record_id).execute()
        if res.data:
            image_path = res.data[0]['image_path']
            supabase.storage.from_(BUCKET_NAME).remove([image_path])
            supabase.table('feedback_records').update({
                'image_path': 'DELETED',
                'consent_granted': False
            }).eq('id', record_id).execute()
            supabase.table('audit_log').insert({
                "record_id": record_id,
                "action": "ERASED"
            }).execute()
            print(f"✅ GDPR Erasure completed for {record_id}.")
