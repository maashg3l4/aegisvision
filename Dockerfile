FROM python:3.10-slim

WORKDIR /app

# Install supervisor
RUN apt-get update && apt-get install -y supervisor && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app code and supervisord config
COPY . .
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Hugging Face Spaces require port 7860
EXPOSE 7860

# Start supervisor
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
