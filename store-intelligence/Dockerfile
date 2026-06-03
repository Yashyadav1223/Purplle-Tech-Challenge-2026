FROM python:3.11-slim

# Install system dependencies required by OpenCV and YOLO
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency files
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose the API port
EXPOSE 10000

# Start the FastAPI server
CMD ["uvicorn", "api.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "10000"]
