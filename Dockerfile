FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install any project-specific dependencies.
COPY requirements.txt ./
RUN pip install -r requirements.txt

# Install the customized local Ultralytics package and its dependencies.
COPY ultralytics/ ./ultralytics/
RUN pip install ./ultralytics

# Copy application code, model weights, and datasets separately so that
# changing the dataset does not invalidate the dependency layers.
COPY *.py ./
COPY *.pt ./
COPY datasets/ ./datasets/

CMD ["/bin/bash"]