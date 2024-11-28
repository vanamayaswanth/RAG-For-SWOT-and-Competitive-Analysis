# Use the official Python 3.12 slim image as the base image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy requirements.txt into the container
COPY reqs.txt ./

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get purge -y build-essential \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy the application code into the container
COPY app.py ./

# Expose the port the app runs on
EXPOSE 8501

# Set the entry point to run the app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.enableCORS=false"]