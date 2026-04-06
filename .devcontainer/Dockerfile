FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System dependencies + Python 3.11
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    build-essential \
    gcc \
    g++ \
    git \
    wget \
    curl \
    bedtools \
    libhdf5-dev \
    zlib1g-dev \
    libbz2-dev \
    libcurl4-openssl-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.11 /usr/bin/python && \
    python3 -m pip install --upgrade pip setuptools wheel

WORKDIR /workspace

# PyTorch 2.11
RUN pip install --no-cache-dir torch==2.11.0 torchvision torchaudio

# Core scientific stack
RUN pip install --no-cache-dir \
    "numpy==1.26.4" \
    "scipy==1.17.1" \
    "pandas==2.3.3" \
    "scikit-learn==1.8.0" \
    "pyarrow==14.0.2"

# pyBigWig first — tangermeme depends on it and will try to build it otherwise
RUN pip install --no-cache-dir pyBigWig==0.3.25

# Bio/genomics packages
RUN pip install --no-cache-dir \
    "anndata==0.12.10" \
    "scanpy==1.11.5" \
    "harmonypy==0.2.0" \
    "gtfparse==2.5.0" \
    "pyfaidx==0.9.0.4" \
    "biopython==1.87" \
    "tangermeme==0.4.4"

# Clustering / dimensionality reduction
RUN pip install --no-cache-dir \
    "igraph==1.0.0" \
    leidenalg \
    "umap-learn==0.5.11" \
    "numba>=0.59.0"

# Viz / utils
RUN pip install --no-cache-dir \
    "matplotlib==3.10.8" \
    "seaborn==0.13.2" \
    "natsort==8.4.0" \
    "tqdm==4.67.3" \
    "decoupler==2.1.4" \
    "session-info2==0.4" \
    "polars==0.20.31"

# Jupyter
RUN pip install --no-cache-dir \
    ipykernel ipython jupyter notebook

# scDoRI (pinned to same commit as dev)
RUN pip install --no-cache-dir \
    "git+https://github.com/bioFAM/scDoRI@08aac82d12df2b1ab4f0eb159a7c4b384b861f77#egg=scdori"

# Copy project
COPY . /workspace/

CMD ["bash"]
