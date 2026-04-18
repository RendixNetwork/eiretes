# syntax=docker/dockerfile:1

FROM python:3.12-slim AS base
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY eiretes /app/eiretes

RUN pip install --no-cache-dir "eirel>=0.2.0,<1" \
    && pip install --no-cache-dir /app/eiretes

EXPOSE 8095
CMD ["eiretes-judge-service"]
