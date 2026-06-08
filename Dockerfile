# Use an official Python slim image as base
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the project files
COPY . .

# Install the package and its dependencies
RUN pip install --no-cache-dir .

# Pre-fetch the satellite TLEs to populate the database
RUN satlinksim-update

# Expose Streamlit's default port
EXPOSE 8501

# Run the UI by default, listening on all interfaces
ENTRYPOINT ["satlinksim-ui"]
CMD ["--server.address=0.0.0.0"]
