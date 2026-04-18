FROM python:3.11-slim

# Install system dependencies including wkhtmltopdf for PDF generation
RUN apt-get update && apt-get install -y \
    build-essential \
    wkhtmltopdf \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Start server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
