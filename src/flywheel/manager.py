import os
import uuid
import streamlit as st
from supabase import create_client, Client
from supabase.lib.client_options import ClientOptions
import httpx
import cv2
import time

# Helper to create a dedicated httpx client
def _make_httpx() -> httpx.Client:
    return httpx.Client(
        http2=False,
        timeout=httpx.Timeout(30.0, connect=10.0),
        limits=httpx.Limits(max_keepalive_connections=10, max_connections=40, keepalive_expiry=15.0),
    )

# Create two separate clients to prevent URL clobbering
_postgrest_httpx_client = _make_httpx()
_storage_httpx_client = _make_httpx()

url: str = st.secrets["SUPABASE_URL"]
key: str = st.secrets["SUPABASE_KEY"]

# Initialize main client with the PostgREST client
supabase: Client = create_client(
    url, key,
    options=ClientOptions(httpx_client=_postgrest_httpx_client)
)

# Re-bind the storage sub-client to use its own isolated httpx client
try:
    supabase._storage = supabase._init_storage_client(
        storage_url=supabase.storage_url,
        headers=supabase.options.headers,
        storage_client_timeout=supabase.options.storage_client_timeout,
        http_client=_storage_httpx_client,
    )
except Exception as e:
    print(f"Storage isolation warning: {e}")

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

        # Upload to Supabase Storage with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                with open(image_path, 'rb') as f:
                    supabase.storage.from_(BUCKET_NAME).upload(
                        path=storage_path,
                        file=f,
                        file_options={"content-type": "image/jpeg"}
                    )
                break  # Success
            except (httpx.ConnectError, httpx.ReadError, httpx.WriteError) as e:
                if attempt == max_retries - 1:
                    raise e
                time.sleep(2 ** attempt)  # Exponential backoff

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
