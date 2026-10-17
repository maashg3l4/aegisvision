FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code
COPY src/ ./src/

# Expose port 7860 (Hugging Face Spaces standard)
EXPOSE 7860

# Start FastAPI and Streamlit (using a simple process manager)
# In production, you would use Gunicorn + Nginx, but this works for a demo
CMD ["python", "src/api/main.py"]
