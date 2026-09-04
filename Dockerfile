FROM pytorch/pytorch:2.7.1-cuda11.8-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        vim \
    && rm -rf /var/lib/apt/lists/*

COPY my-ultralytics/ ./ultralytics/
RUN pip install -e ./ultralytics

CMD ["/bin/bash"]