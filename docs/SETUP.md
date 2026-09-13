# Setup: the repo, the submodule, and the model weights

Everything below runs inside the `nav` container (`README.md` → Quick start).
Two things this repo does **not** contain and cannot run without: the ASCENT
reference checkout (a pinned submodule) and ~5 GB of model weights (downloaded).

---

## 1. Clone with the submodule

`relative_work/ascent` is a submodule pinned to
[Zeying-Gong/ascent](https://github.com/Zeying-Gong/ascent) at `8f7bbf9`. It is
where the five perception models live, so nothing runs without it.

```bash
git clone --recursive <this repo>
# already cloned without --recursive:
git submodule update --init --recursive
```

**`--recursive` is not optional.** ASCENT itself carries nine nested
submodules (GroundingDINO, MobileSAM, D-FINE, RAM++, places365, vlfm,
frontier_exploration, depth_camera_filtering, habitat-lab), and a non-recursive
clone leaves all nine as empty directories. The servers import from them.

### The compiled CUDA extension

GroundingDINO ships a CUDA op (`MultiScaleDeformableAttention`) that must be
**built in the working tree**. It is not in git, so a fresh clone does not have
it, and the failure is not obvious — the server starts, answers its health
check, and then returns HTTP 500 on the first real frame with

```
NameError: name '_C' is not defined
```

which the run surfaces as `PerceptionUnavailable: …/gdino: HTTP Error 500`.
Build it once, in the `ascent` env:

```bash
/workspace/.conda-envs/ascent/bin/pip install -e relative_work/ascent/third_party/GroundingDINO
```

It produces `third_party/GroundingDINO/groundingdino/_C.cpython-39-*.so`; check
that file exists before blaming the model.

---

## 2. Model weights

Nothing is committed. `pretrained_weights/` inside the submodule is gitignored,
and so is `data/`.

### The five served models + the mover

```bash
bash scripts/fetch_ascent_weights.sh        # -> relative_work/ascent/pretrained_weights/
```

| file | size | used by | source |
|---|---|---|---|
| `groundingdino_swint_ogc.pth` | 662 MB | stair detector (:13184) | GroundingDINO v0.1.0-alpha release |
| `ram_plus_swin_large_14m.pth` | 2.9 GB | RAM++ tagger (:13185) | HF `xinyu1205/recognize-anything-plus-model` |
| `dfine_x_obj2coco.pth` | 240 MB | D-FINE detector (:13186) | Peterande/storage dfinev1.0 |
| `mobile_sam.pt` | 39 MB | mask per detection (:13183) | linked from `data/weights/` |
| `rednet_semmap_mp3d_40.pth` | 626 MB | stair segmentation | linked from `data/weights/` |
| `resnet50_places365.pth.tar` | 93 MB | room labels (in-process) | linked from `data/place365/` |
| `pointnav_weights.pth` | 32 MB | the frozen mover | linked from `data/weights/` |

The script links the four this repo already has and downloads the other three;
run it again any time — it skips what is present. **BLIP-2** (the value map and
commit gate, :13182) is not a file here: `lavis` pulls it from HF on the
server's first start, which is why that server takes longest to come up.

It also offers Qwen2.5-7B-Instruct (~15 GB) for ASCENT's own LLM server. **You
do not need it**: both this repo and the reference call `qwen2.5:7b` on the
local ollama instead. Skip it unless you are running native ASCENT with its
bf16 server.

### The OSG-side weights

```bash
python scripts/download_weights.py                 # detector + mobileclip
python scripts/download_weights.py --pointnav      # the mover
python scripts/download_weights.py --rednet        # stair segmentation
python scripts/download_weights.py --clip          # value-map image-text model
```

### The LLM

```bash
docker exec docker-ollama-1 ollama pull qwen2.5:7b     # 4.7 GB, Q4_K_M
```

Note it is **quantised** — ASCENT's paper configuration is bf16. Both sides
here use the same quantised model, so comparisons between them are fair; see
`docs/METHODOLOGY.md` §5.4.

### Datasets

```bash
python scripts/download_data.py --username <TOKEN_ID> --password <TOKEN_SECRET> \
    --uids hm3d_val_v0.2
python scripts/download_data.py --episodes-only
```

HM3D scenes need a free licence from
https://matterport.com/habitat-matterport-3d-research-dataset.

---

## 3. Check it works

```bash
bash scripts/serve_perception.sh                     # five servers, ~60 s to load
python scripts/run_eval.py eval.num_episodes=1       # one live episode
```

A green run prints one `[1/1] … success=…` line. If it raises
`PerceptionUnavailable`, the named port tells you which model is unhappy;
`relative_work/ascent/debug/vlm_logs/<name>.log` has the server-side traceback.

Total on disk: ~5 GB of weights, ~1 GB of nested submodule source, plus HM3D
scenes (~10 GB for val).

---

## 4. The `ascent` conda environment

The five servers do **not** run in habitat's interpreter. They run from
`/workspace/.conda-envs/ascent`, a python 3.9 env with torch 2.1.0+cu118,
numpy 1.23.5 and transformers 4.37.0 — a combination habitat-sim cannot share,
which is the whole reason for process-per-model over HTTP.

**That directory is gitignored and exists only on this machine.** It was cloned
from a container built by `docker/Dockerfile.ascent`, and the clone source is
already gone. If it is lost, rebuild it:

```bash
docker build -f docker/Dockerfile.ascent -t ascent:native .
# then copy /opt/conda/envs/ascent out of a container from that image into
# /workspace/.conda-envs/ascent, or run the servers inside it with
# relative_work/ascent and pretrained_weights mounted.
```

`docker/Dockerfile.ascent` is therefore **not part of the running HM3D
pipeline** — `docker/compose.yaml` never builds it and nothing imports from the
image — but it is the only record of how that environment resolves. Its header
documents each pin against the failure that forced it: no nvcc in a `-runtime-`
base, lavis's unsatisfiable `spacy` chain on python 3.9, the numpy <1.24 floor,
D-FINE's hidden training-stack imports, and transformers installed last. Do not
delete it.

Verify the env is intact:

```bash
/workspace/.conda-envs/ascent/bin/python -c "
import habitat_sim, lavis, mobile_sam, groundingdino.util.inference, ram
from groundingdino import _C
print('ascent env OK')"
```

## 5. What the submodule does and does not carry

The submodule is pinned to **upstream, unmodified**. During this work the
checkout carried three local patches which are deliberately no longer applied:

| patch | what it did | consequence of dropping it |
|---|---|---|
| behaviour recorder (+94 lines in `ascent_policy.py`, `ascent_trainer.py`, `llm_planner.py`, plus `behaviour_recorder.py`) | produced ASCENT's per-step trace | the reference trace cannot be regenerated. The trace **itself is preserved** in this repo at `data/reference/ascent_behaviour_100/` (gzipped, 772 KB), which is what `scripts/compare_ascent_osg.py` reads by default |
| a lock in `model_api/server_wrapper_out.py` | serialised model calls | the servers are **not reentrant** again. GroundingDINO caches per-image state on the module, so two concurrent evals race and the loser gets HTTP 500. **Run one evaluation at a time.** This repo's client retries once, which covers a transient hit, not sustained concurrency |
| `model_api/qwen25_ollama.py` | served ASCENT's Qwen endpoint from ollama | only matters when running native ASCENT; its own launcher expects the bf16 server |

Nothing in this repo's own pipeline depends on those patches — verified by a
live episode against the clean submodule after the conversion.
