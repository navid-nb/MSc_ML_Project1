FROM python:3.10-slim

# PREVENT Python from buffering stdout/stderr (vital for logging in Docker)
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DAGSTER_HOME=/opt/dagster/dagster_home

# RUN apt-get update && apt-get install -y \
#     build-essential \
#     libpq-dev \
#     git \
#     && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y  \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

COPY . .

#prod: dagster api grpc -h 0.0.0.0 -p 4000 -m dagster_pipeline
#dev: dagster dev -m dagster_pipeline
CMD ["dagster", "api", "grpc", "-h", "0.0.0.0", "-p", "4000", "-m", "dagster_pipeline"]
#CMD ["tail", "-f", "/dev/null"]
