FROM nvidia/cuda:12.6.2-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_NO_CACHE_DIR=1
ENV PYTHONUNBUFFERED=1
ENV PATH="/opt/venv/bin:${PATH}"
ENV UV_UNMANAGED_INSTALL=/usr/local/bin

WORKDIR /workspace

COPY requirements.txt /workspace/requirements.txt
COPY codebase/data/cached_fineweb10B.py /workspace/data/cached_fineweb10B.py
COPY tools /opt/nanogpt/tools
COPY submit.sh /usr/local/bin/submit

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential ca-certificates curl git gnupg \
    && rm -rf /var/lib/apt/lists/*
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
RUN apt-get update && apt-get install -y --no-install-recommends nodejs && rm -rf /var/lib/apt/lists/*
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
RUN uv python install 3.10.12
RUN uv venv /opt/venv --python 3.10.12 --clear
RUN uv pip install --python /opt/venv/bin/python setuptools==80.9.0
RUN uv pip install --python /opt/venv/bin/python \
    "torch @ https://download.pytorch.org/whl/nightly/cu126/torch-2.9.0.dev20250718%2Bcu126-cp310-cp310-manylinux_2_28_x86_64.whl" \
    "pytorch-triton @ https://download.pytorch.org/whl/nightly/pytorch_triton-3.4.0%2Bgit11ec6354-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
RUN uv pip install --python /opt/venv/bin/python -r /workspace/requirements.txt
RUN uv pip install --python /opt/venv/bin/python "flash_attn_3 @ https://github.com/varunneal/flash-attention/releases/download/v3.0.0b1-alpha/flash_attn_3-3.0.0b1-cp39-abi3-linux_x86_64.whl"
RUN uv pip install --python /opt/venv/bin/python --reinstall triton==3.4.0 setuptools==80.9.0
RUN chmod 755 /usr/local/bin/submit
RUN /opt/venv/bin/python /workspace/data/cached_fineweb10B.py 9

CMD ["bash"]
