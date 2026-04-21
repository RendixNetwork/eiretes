# syntax=docker/dockerfile:1
#
# Production image for the eiretes LLM judge service. Builds from this
# repo's root; the eirel SDK is pulled from PyPI via pyproject.toml deps.

FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY . /app/eiretes

RUN pip install --no-cache-dir /app/eiretes

EXPOSE 8095
CMD ["eiretes-judge-service"]
