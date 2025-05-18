# Start with an official Python base image
FROM python:3.10-slim-bookworm

# Set the working directory in the container
WORKDIR /service

# Install system dependencies needed for building some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY ./requirements.txt /service/requirements.txt

# Install Python dependencies
# Use --no-cache-dir to reduce image size
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r /service/requirements.txt # This should now succeed for hdbscan

# Copy the rest of your application's code into the container
COPY ./app /service/app
COPY ./bertopic_model_final_guided_multilang_gensim.joblib /service/bertopic_model_final_guided_multilang_gensim.joblib

# Expose the port the app runs on
EXPOSE 8001

# Command to run the application using Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]