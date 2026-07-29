FROM nvidia/cuda:13.3.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH
ARG MINIFORGE_VERSION=26.3.2-3
ARG MINIFORGE_SHA256=848194851a98903134187fbb4ab50efe87b003e0c0f808f97644b7524a62bf2c
ARG GS_LIGHTNING_REF=ebc2e44886725a5270ff931fd029de6d541ce694

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget build-essential cmake libboost-all-dev \
    libfreeimage-dev libgoogle-glog-dev libgflags-dev \
    libatlas-base-dev libsuitesparse-dev ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install a checksum-pinned Miniforge release.
RUN wget -q "https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/Miniforge3-Linux-x86_64.sh" -O /tmp/conda.sh \
    && echo "${MINIFORGE_SHA256}  /tmp/conda.sh" | sha256sum -c - \
    && bash /tmp/conda.sh -b -p $CONDA_DIR && rm /tmp/conda.sh

# Create environment
COPY environment.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml && conda clean -afy

# Clone a reviewed gaussian-splatting-lightning revision.
RUN git clone https://github.com/yzslab/gaussian-splatting-lightning /opt/gs-lightning \
    && cd /opt/gs-lightning \
    && git checkout --detach "${GS_LIGHTNING_REF}" \
    && conda run -n gsplat pip install -r requirements.txt

# Copy pipeline
COPY . /app
WORKDIR /app

ENV GS_LIGHTNING_PATH=/opt/gs-lightning

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "gsplat", "python", "pipeline.py"]
