FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget build-essential cmake libboost-all-dev \
    libfreeimage-dev libgoogle-glog-dev libgflags-dev \
    libatlas-base-dev libsuitesparse-dev ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install conda
RUN wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/conda.sh \
    && bash /tmp/conda.sh -b -p $CONDA_DIR && rm /tmp/conda.sh

# Create environment
COPY environment.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml && conda clean -afy

# Clone gaussian-splatting-lightning
RUN git clone https://github.com/yzslab/gaussian-splatting-lightning /opt/gs-lightning \
    && cd /opt/gs-lightning && conda run -n gsplat pip install -r requirements.txt

# Copy pipeline
COPY . /app
WORKDIR /app

ENV GS_LIGHTNING_PATH=/opt/gs-lightning

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "gsplat", "python", "pipeline.py"]
