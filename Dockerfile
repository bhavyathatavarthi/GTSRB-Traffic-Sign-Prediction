# 1. Base image with Python
FROM python:3.10-slim

# 2. Set working directory inside container
WORKDIR /app

# 3. Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy all project files (FastAPI code + model)
COPY . .

# 5. Expose port FastAPI will run on
EXPOSE 8000

# 6. Start the FastAPI server
CMD ["uvicorn", "gtsrb_app:app", "--host", "0.0.0.0", "--port", "8000"]
