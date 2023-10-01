# Pull base image
FROM python:3.10.2-slim-bullseye

# Set environment variables
ENV PIP_DISABLE_PIP_VERSION_CHECK 1
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set work directory
WORKDIR /bestproperties

# Install dependencies
COPY ./requirements.txt .
RUN pip install -r requirements.txt

# Copy project
COPY . .

COPY ./bpaa/property_price_model.h5 /bestproperties/bpaa/property_price_model.h5
COPY ./bpaa/scaler.pkl /bestproperties/bpaa/scaler.pkl

