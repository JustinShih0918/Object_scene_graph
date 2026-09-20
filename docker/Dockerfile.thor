# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------
# The pipeline and the perception models, on a Jetson AGX Thor.
#
# This is the Orin-style half of the deployment: an L4T base, arm64, the
# NVIDIA-built PyTorch, and NOTHING ELSE from the x86 image. Two things it
# deliberately does not contain:
#
#   HABITAT. There is no aarch64 build of habitat-sim -- the `aihabitat` conda
#   channel is linux-64 only -- and the robot does not need one. Verified: the
#   whole robot path (Ros2Env -> NavAgent -> run_episode -> save_map) runs with
#   habitat blocked at import. `eval.mode=ros2` is the only mode this image can
#   run; the benchmark modes stay on the x86 image.
#
#   ROS. The bridge lives in its own container (docker/Dockerfile.bridge),
#   because `src/osg/ros2/` was written to import under two interpreters and a
#   bridge that needs only numpy + rclpy has no business carrying CUDA. That
#   split is also what lets the bridge match the robot's Ubuntu 22.04 / Humble
#   exactly while this image stays on JetPack's own Ubuntu -- no ROS built from
#   source, no distro gap to bridge, no Python-version fight with torch.
#
#   docker compose -f docker/compose.thor.yaml --env-file docker/.env build
#   docker compose -f docker/compose.thor.yaml --env-file docker/.env up -d
#
# See docs/THOR.md. Every ARG below is a "verify on YOUR unit" item.
# ---------------------------------------------------------------------------

# The L4T image matching the JetPack ON THE DEVICE. Check it with
#     cat /etc/nv_tegra_release
# and set L4T_BASE to the matching tag; a mismatch between the container's CUDA
# userspace and the host driver is the failure that looks like a broken GPU.
# `l4t-jetpack` rather than `l4t-base`: GroundingDINO compiles a fused
# attention kernel and needs a real nvcc, exactly as on x86.
ARG L4T_BASE=nvcr.io/nvidia/l4t-jetpack:r38.2.0
FROM ${L4T_BASE}

ARG DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Thor's GPU is Blackwell-generation. Orin is 8.7; check with
#     python3 -c "import torch; print(torch.cuda.get_device_capability())"
# once torch is in, and rebuild with the right value if this is wrong -- a
# kernel compiled for the wrong arch falls back to CPU, which for GroundingDINO
# is the difference between a run and a day.
ARG CUDA_ARCH=11.0
# NVIDIA does not publish Jetson torch wheels on PyPI; they come from a Jetson
# index keyed by JetPack version. Verify this matches your JetPack.
ARG TORCH_INDEX_URL=https://pypi.jetson-ai-lab.dev/jp7/cu130
ARG TORCH_SPEC=torch
ARG TORCHVISION_SPEC=torchvision
# BLIP-2 arrives through salesforce-lavis, which has no Jetson wheels and pins
# an old transformers. It feeds only the value map and ASCENT's commit gate,
# and `exploration.value_model=clip` replaces it -- so it is optional rather
# than allowed to block the whole image.
ARG WITH_BLIP2=1

RUN --mount=type=cache,target=/var/cache/apt,sharing=private \
    rm -f /etc/apt/apt.conf.d/docker-clean \
    && apt-get update && apt-get install -y --no-install-recommends \
       git curl ca-certificates build-essential ninja-build cmake pkg-config \
       python3-dev python3-pip python3-venv \
       libjpeg-dev libpng-dev libopenblas-dev libgl1 libglib2.0-0 \
       sudo tmux ffmpeg \
    && rm -rf /var/lib/apt/lists/*

ENV CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST=${CUDA_ARCH} \
    PATH=/usr/local/cuda/bin:${PATH} \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PIP_ROOT_USER_ACTION=ignore

# ---------------------------------------------------------------------------
# 1. The pipeline environment: the system python, plus NVIDIA's torch.
# ---------------------------------------------------------------------------
# numpy<2 is the same constraint the x86 image holds: the pinned opencv and
# ultralytics builds are compiled against the 1.x ABI, and torch's own
# `_ARRAY_API not found` is what a 2.x numpy produces underneath them.
RUN printf 'numpy<2\n' > /opt/pip-constraints.txt
ENV PIP_CONSTRAINT=/opt/pip-constraints.txt

RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install --index-url ${TORCH_INDEX_URL} --extra-index-url https://pypi.org/simple \
        ${TORCH_SPEC} ${TORCHVISION_SPEC} \
 && python3 - <<'PY'
import torch
print("torch", torch.__version__, "cuda build", torch.version.cuda)
if not torch.cuda.is_available():
    # Not fatal at BUILD time -- the builder may have no GPU visible -- but the
    # message has to name the two knobs that actually cause it.
    print("WARNING: torch.cuda unavailable during build. If this persists at "
          "run time, L4T_BASE does not match the device's JetPack.")
PY

# The pipeline's own dependencies, from pyproject. Installed by name rather
# than `pip install -e .` so a source checkout is not required in the image:
# the repo is bind-mounted and reached through PYTHONPATH, exactly as on x86.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install \
        "numpy<2" scipy scikit-learn scikit-image \
        opencv-python-headless "hydra-core>=1.3" omegaconf "openai>=1.30" \
        imageio matplotlib pillow pyyaml tqdm requests \
        "ultralytics==8.3.*" pytest pytest-timeout

# ---------------------------------------------------------------------------
# 2. The five perception model servers.
# ---------------------------------------------------------------------------
# A venv WITH system site-packages, so torch is shared rather than installed a
# second time (several GB on this platform), while lavis's old transformers and
# whatever numpy it drags along land in the venv layer and shadow the
# pipeline's rather than overwriting it. Same separation the x86 image gets
# from its second conda env, for the same reason -- and `serve_perception.sh`
# already selects it through ASCENT_PYTHON, so that script needs no changes.
ENV ASCENT_PYTHON=/opt/venvs/models/bin/python
RUN python3 -m venv --system-site-packages /opt/venvs/models

# The reference source. `relative_work/ascent` is a git submodule, so the build
# FAILS HERE unless it was checked out recursively first:
#     git submodule update --init --recursive
COPY relative_work/ascent /opt/ascent

COPY docker/thor-model-requirements.txt /opt/thor-model-requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    ${ASCENT_PYTHON} -m pip install --upgrade pip setuptools wheel \
 && ${ASCENT_PYTHON} -m pip install -r /opt/thor-model-requirements.txt

WORKDIR /opt/ascent
# The editable installs the servers import. --no-deps throughout: the resolver
# is what pulls numpy 2.x over the constraint and breaks torch underneath.
RUN for pkg in third_party/MobileSAM \
               third_party/frontier_exploration \
               third_party/depth_camera_filtering; do \
        ${ASCENT_PYTHON} -m pip install --no-cache-dir --no-deps -e "$pkg" || exit 1; \
    done

# GroundingDINO: compile the fused attention kernel against THIS platform's
# nvcc. `pip install -e .` is not used -- setuptools' develop re-invokes pip in
# an isolated env with no torch, and GroundingDINO's setup.py responds by
# trying to install torch from inside it. A .pth makes it importable instead,
# which is how the x86 image carries it too.
RUN cd /opt/ascent/third_party/GroundingDINO \
 && ${ASCENT_PYTHON} setup.py build_ext --inplace \
 && echo /opt/ascent/third_party/GroundingDINO \
      > /opt/venvs/models/lib/python3*/site-packages/_ascent_gdino.pth \
 && ${ASCENT_PYTHON} -c "import torch; import groundingdino; \
      from groundingdino import _C; print('GroundingDINO kernel OK')"

RUN echo /opt/ascent/third_party/recognize-anything \
      > /opt/venvs/models/lib/python3*/site-packages/_ascent_ram.pth

# BLIP-2 last, and optional: see WITH_BLIP2 above.
RUN --mount=type=cache,target=/root/.cache/pip \
    if [ "${WITH_BLIP2}" = "1" ]; then \
        ${ASCENT_PYTHON} -m pip install --no-cache-dir salesforce-lavis \
          && ${ASCENT_PYTHON} -m pip install --no-cache-dir "transformers==4.37.0" \
          && ${ASCENT_PYTHON} -c "import lavis; print('lavis OK')" \
          || { echo "BLIP-2 did not install on aarch64. Rebuild with"; \
               echo "  --build-arg WITH_BLIP2=0"; \
               echo "and run with exploration.value_model=clip."; exit 1; }; \
    else \
        echo "BLIP-2 skipped (WITH_BLIP2=0): use exploration.value_model=clip"; \
    fi

# ---------------------------------------------------------------------------
# 3. Runtime
# ---------------------------------------------------------------------------
ENV PYTHONPATH=/workspace/src \
    PIP_CONSTRAINT= \
    YOLO_CONFIG_DIR=/tmp/Ultralytics \
    OPENBLAS_NUM_THREADS=4

ARG USERNAME=user
ARG USER_UID=1000
ARG USER_GID=1000
RUN groupadd -g $USER_GID $USERNAME 2>/dev/null || true \
 && useradd -m -s /bin/bash -u $USER_UID -g $USER_GID -G sudo $USERNAME 2>/dev/null || true \
 && echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers \
 && chown -R $USER_UID:$USER_GID /opt/ascent /opt/venvs

# The bind mount is shared with the host user; without this every file written
# from a `docker exec` shell comes out 0644 and the host cannot modify it.
RUN printf 'umask 002\n' > /etc/profile.d/00-shared-umask.sh \
 && printf '\numask 002\n' >> /etc/bash.bashrc

COPY docker/entrypoint.thor.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

USER $USERNAME
WORKDIR /workspace
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
