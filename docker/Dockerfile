# One image, every experiment: OSG's agent, the ASCENT reference, and all five
# perception servers.
#
#   docker compose -f docker/compose.yaml --env-file docker/.env build nav
#
# The build context is `docker/` and everything else is fetched, so no working
# tree state is required -- the ASCENT reference is cloned at the pin in
# ASCENT_REF below.
#
# TWO CONDA ENVS, ONE IMAGE. They cannot be merged into one interpreter and
# every attempt is encoded below:
#
#   habitat  python 3.9, torch 2.4.1+cu121, numpy<2   -- habitat-sim, OSG, YOLOE
#   ascent   python 3.9, torch 2.1.0+cu118, numpy 1.23.5, transformers 4.37.0
#            -- BLIP-2 (lavis), MobileSAM, GroundingDINO, RAM++, D-FINE
#
# habitat-sim 0.3.1 pins numpy<1.24; lavis needs a transformers that the OSG
# side does not want; and the two torch builds differ. Process-per-model over
# HTTP is ASCENT's own architecture, and the split env is why it is necessary.
#
# THE BASE IMAGE IS `11.8.0-devel`, NOT `12.1.1-runtime`, for two reasons that
# together admit no other choice:
#   1. GroundingDINO's fused attention kernel must be compiled, and a
#      `-runtime-` base has no nvcc. Adding one via conda (`cuda-nvcc`,
#      `cuda-toolkit=11.8`) mixed runtime and dev components across 11.8/12.4
#      and never produced a working toolchain. `-devel-` ships nvcc, headers
#      and stubs as ONE consistent toolchain.
#   2. torch's cpp_extension refuses to build when nvcc's version differs from
#      `torch.version.cuda`, and ASCENT's torch is cu118 -- so nvcc must be
#      11.8, which fixes the base.
# The habitat env's torch 2.4.1+cu121 is unaffected: torch wheels bundle their
# own CUDA runtime, and nothing on the OSG side compiles a CUDA extension. The
# base image's CUDA version constrains compilation, not execution.

FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

RUN rm -f /etc/apt/apt.conf.d/docker-clean; \
    echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache

# EGL/GL userland for headless habitat-sim (driver libs are injected by
# nvidia-container-toolkit at run time); build-essential/ninja for the CUDA
# extension; tmux because the model servers run in a tmux session.
RUN --mount=type=cache,target=/var/cache/apt,sharing=private \
    apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates bzip2 unzip build-essential ninja-build \
    libegl1 libgl1 libglvnd0 libx11-6 libxext6 libopengl0 libglx0 \
    libjpeg-dev libpng-dev libglib2.0-0 sudo tmux \
    && rm -rf /var/lib/apt/lists/*

RUN curl -L https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
      -o /tmp/miniforge.sh \
    && bash /tmp/miniforge.sh -b -p /opt/conda \
    && rm /tmp/miniforge.sh
ENV PATH=/opt/conda/bin:$PATH

# ---------------------------------------------------------------------------
# 1. `habitat`: the OSG side. habitat-sim 0.3.1 is a python 3.9 build.
# ---------------------------------------------------------------------------
RUN conda create -y -n habitat python=3.9 cmake=3.14.0 \
    && conda install -y -n habitat habitat-sim=0.3.1 withbullet headless \
       -c conda-forge -c aihabitat \
    && conda clean -ya

SHELL ["conda", "run", "--no-capture-output", "-n", "habitat", "/bin/bash", "-c"]

RUN git clone --depth 1 --branch v0.3.1 \
      https://github.com/facebookresearch/habitat-lab.git /opt/habitat-lab \
    && pip install --no-cache-dir -e /opt/habitat-lab/habitat-lab

RUN pip install --no-cache-dir torch==2.4.1 torchvision==0.19.1 \
      --index-url https://download.pytorch.org/whl/cu121 \
    && pip install --no-cache-dir \
       "numpy<2" "ultralytics==8.3.*" openai "hydra-core>=1.3" omegaconf \
       scipy scikit-image scikit-learn opencv-python-headless pytest pytest-timeout \
       imageio matplotlib anthropic "rerun-sdk"

# CLIP tokenizer for YOLOE's text prompts (otherwise ultralytics AutoUpdate
# re-installs it in every ephemeral container).
RUN pip install --no-cache-dir git+https://github.com/ultralytics/CLIP.git

# ---------------------------------------------------------------------------
# 2. `ascent`: the five perception servers and the native reference.
#
# ASCENT's own README is the spine (python 3.9, torch 2.1.0+cu118,
# habitat-sim 0.3.1, transformers==4.37.0 installed LAST -- that override of
# lavis's `<4.27` pin is ASCENT's, and it is what lets BLIP-2 and Qwen2.5
# share one environment).
# ---------------------------------------------------------------------------
RUN conda create -y -n ascent python=3.9 cmake=3.14.0 && conda clean -ya
SHELL ["conda", "run", "--no-capture-output", "-n", "ascent", "/bin/bash", "-c"]

RUN pip install --no-cache-dir torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
      --index-url https://download.pytorch.org/whl/cu118

RUN conda install -y -n ascent habitat-sim=0.3.1 withbullet headless \
      -c conda-forge -c aihabitat && conda clean -ya

# The numpy floor, set BEFORE anything that links against it. habitat-sim 0.3.1
# requires <1.24; scipy 1.10.1 is the last release built against that line.
RUN pip install --no-cache-dir "numpy==1.23.5" "scipy==1.10.1" "pandas==1.5.3"

# The reference source, CLONED rather than COPYed. compose sets the build
# context to `docker/`, so the submodule is not in it -- and widening the
# context to the repo root would ship outputs/, data/ and 4.6 GB of weights to
# the daemon on every build. Cloning also means the image matches the pin
# regardless of the state of anyone's working tree.
#
# ASCENT_REF must equal the submodule pin recorded in git. That is enforced by
# tests/unit/test_dockerfile_pins.py, so the two cannot drift silently.
ARG ASCENT_REF=8f7bbf906c8433a47c4d2c46a268cd1258c4057c
RUN git clone https://github.com/Zeying-Gong/ascent.git /opt/ascent \
    && cd /opt/ascent \
    && git checkout -q "$ASCENT_REF" \
    && git submodule update --init --recursive \
    && rm -rf /opt/ascent/.git
WORKDIR /opt/ascent

RUN pip install --no-cache-dir -e third_party/habitat-lab/habitat-lab \
    && pip install --no-cache-dir -e third_party/habitat-lab/habitat-baselines

# With a -devel- base this builds the fused CUDA kernel. Without it,
# GroundingDINO silently falls back to CPU -- at one call per step that is
# 14-40 hours for a 100-episode run -- or dies with `NameError: _C`.
ENV CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST="8.6;8.9;9.0"
RUN cd third_party/GroundingDINO \
    && python setup.py build_ext --inplace \
    && pip install --no-cache-dir -e . --no-build-isolation --no-deps \
    && python -c "from groundingdino import _C; print('GroundingDINO CUDA extension: OK')"

RUN pip install --no-cache-dir -e third_party/MobileSAM \
    && pip install --no-cache-dir -e third_party/vlfm

# ASCENT's requirements.txt, minus the parts that cannot resolve here.
# `salesforce-lavis==1.0.2` declares dependencies that cannot resolve on python
# 3.9 -- spacy pulls thinc>=8.3.12, which requires >=3.10 -- so it goes in with
# --no-deps and its real imports are pinned by hand. spacy 3.7.5 is the last
# release supporting 3.9, and only `img2prompt_vqa` needs it at all.
RUN pip install --no-cache-dir --no-deps salesforce-lavis==1.0.2 \
    && pip install --no-cache-dir \
       "spacy==3.7.5" "timm==0.4.12" "opencv-python-headless==4.10.0.84" \
       iopath decord webdataset fairscale einops pycocoevalcap \
       omegaconf plotly flask open3d dash scikit-learn joblib seaborn \
       "supervision==0.25.1" openpyxl "yapf==0.43.0" \
    && pip install --no-cache-dir --no-deps \
       "recognize-anything @ git+https://github.com/xinyu1205/recognize-anything.git"

# Four imports requirements.txt never declares, each found by a server failing
# at import time:
#  - scikit-image: `ascent/llm_planner.py` imports skimage.metrics.ssim, so the
#    eval dies before habitat starts. 0.19.3 is the last line built against
#    numpy <1.24.
#  - faster-coco-eval + calflops + accelerate: pulled in transitively by
#    `from src.core import YAMLConfig` in model_api/dfine_out.py -- D-FINE's
#    __init__ imports its TRAINING stack even though only inference is used.
#    1.6.7 is the floor for `faster_coco_eval.utils.pytorch`. calflops and
#    accelerate go in --no-deps, or accelerate's solver pulls a transformers
#    newer than the 4.37.0 pinned below.
#
# Without these, D-FINE is the only server that fails, and ASCENT's own clients
# swallow connection errors -- so a native run would silently evaluate with one
# of its two detectors dead. (This repo's clients fail loud instead.)
RUN pip install --no-cache-dir "scikit-image==0.19.3" "faster-coco-eval==1.6.7" \
    && pip install --no-cache-dir --no-deps calflops "accelerate==0.26.1"

# LAST, and deliberately: ASCENT's override of lavis's transformers pin. 4.37.0
# is the floor for Qwen2.5 and works with BLIP-2 in practice, which is what
# lets all the model servers share one environment.
RUN pip install --no-cache-dir "transformers==4.37.0" \
    && python -c "import torch, habitat_sim, transformers, numpy, cv2; \
       import lavis.models, mobile_sam, groundingdino.util.inference, ram; \
       from groundingdino import _C; \
       print('ascent env: all model packages import OK')"

# Exactly one opencv build may be installed: two of them abort at import with
# "CPU dispatcher tracer already initlized".
RUN pip uninstall -y opencv-python opencv-contrib-python 2>/dev/null; \
    python -c "import cv2; print('cv2', cv2.__version__)"

# ---------------------------------------------------------------------------
# 3. Runtime
# ---------------------------------------------------------------------------
SHELL ["/bin/bash", "-c"]

ENV NVIDIA_DRIVER_CAPABILITIES=all \
    MAGNUM_LOG=quiet \
    HABITAT_SIM_LOG=quiet \
    __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json \
    PYTHONPATH=/workspace/src \
    ASCENT_PYTHON=/opt/conda/envs/ascent/bin/python

# wgpu (the Rerun native viewer) probes every backend it is built with,
# including GLES-over-EGL, before picking one. That probe walks X11/GBM/Wayland
# and panics under the virtualized X server this container runs against, even
# though __EGL_VENDOR_LIBRARY_FILENAMES makes plain EGL fine for habitat-sim.
# Vulkan-only skips the probe; the GPU is still found as a Vulkan adapter.
ENV WGPU_BACKEND=vulkan

# GLVND dispatch + GLX for habitat-sim's bindings, and the five libraries winit
# dlopen()s at run time -- without them a GUI app fails with the misleading
# "Failed to load one of xlib's shared libraries" even though libX11 is present.
# libvulkan1 + mesa give wgpu a backend (and a llvmpipe fallback). This is only
# the CLIENT side: reaching a display also needs DISPLAY and the X socket from
# compose.yaml's x11 block.
RUN --mount=type=cache,target=/var/cache/apt,sharing=private \
    apt-get update && apt-get install -y --no-install-recommends \
    libxcursor1 libxrandr2 libxi6 libxkbcommon0 libxkbcommon-x11-0 \
    libxcomposite1 libxinerama1 libxtst6 \
    libvulkan1 mesa-vulkan-drivers libgl1-mesa-dri \
    x11-utils xauth \
    && rm -rf /var/lib/apt/lists/*

RUN printf '\n# habitat env for `docker exec -it ... bash`\nif [ -z "$CONDA_DEFAULT_ENV" ]; then\n  source /opt/conda/etc/profile.d/conda.sh\n  conda activate habitat\nfi\n' >> /etc/bash.bashrc

COPY modules/install_x11_opengl_vulkan.sh /tmp/install_x11_opengl_vulkan.sh
RUN --mount=type=cache,target=/var/cache/apt,sharing=private \
    /tmp/install_x11_opengl_vulkan.sh && rm /tmp/install_x11_opengl_vulkan.sh

COPY /entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ARG USERNAME=user
ARG USER_UID=1000
ARG USER_GID=1000
RUN --mount=type=cache,target=/var/cache/apt,sharing=private \
    apt-get update && apt-get install -y sudo \
    && groupadd -g $USER_GID $USERNAME \
    && useradd -m -s /bin/bash -u $USER_UID -g $USER_GID -G sudo $USERNAME \
    && echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers \
    && chown -R $USER_UID:$USER_GID /opt/ascent \
    && rm -rf /var/lib/apt/lists/*

USER $USERNAME

COPY --chown=$USERNAME:$USERNAME \
    modules/install_claude_code.sh /tmp/install_claude_code.sh
RUN --mount=type=cache,target=/var/cache/apt,sharing=private \
    /tmp/install_claude_code.sh && rm /tmp/install_claude_code.sh

# Hardcoded rather than via ~/.bashrc: compose.yaml sets HOME=/tmp at run time,
# which both points $HOME/.local/bin at the wrong directory and is never sourced
# by the non-interactive `conda run` shell entrypoint.sh execs into.
ENV PATH="/home/$USERNAME/.local/bin:$PATH"

# Model weights are NOT baked in (~5 GB, some need a HuggingFace token):
# `bash scripts/fetch_ascent_weights.sh` downloads them into the mounted
# workspace. See docs/SETUP.md.
WORKDIR /workspace
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
