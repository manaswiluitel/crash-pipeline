# ===============================
# 🧱 Base image
# ===============================
FROM python:3.12-slim

# ===============================
# 🏗️ Environment setup
# ===============================
WORKDIR /app

# Prevent Python from writing .pyc files and buffering logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ===============================
# 🧩 System dependencies
# ===============================
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ===============================
# 📦 Copy source and install deps
# ===============================
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your cleaner code
COPY . .

# ===============================
# 📁 Ensure /data/gold directory exists (for persistent DuckDB)
# ===============================
RUN mkdir -p /data/gold

# ===============================
# 🚀 Default command: run the RabbitMQ consumer
# ===============================
CMD ["python", "cleaner.py"]
