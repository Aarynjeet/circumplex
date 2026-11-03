"""Generate notebooks/train_kaggle.ipynb and notebooks/analysis.ipynb.

The notebooks are generated (not hand-edited) so their code stays reviewable in
this file and in diffs. Re-run after editing: ``python scripts/build_notebooks.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_URL = "https://github.com/Gurehmat/circumplex.git"


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.strip("\n").splitlines(keepends=True),
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip("\n").splitlines(keepends=True),
    }


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


TRAIN = [
    md(f"""
# Circumplex: all GPU training, end to end (Kaggle)

**Settings (right-hand panel → Session options):** Accelerator **GPU T4 x2** or **GPU P100**
(only one GPU is used), **Internet: On**, Persistence: off is fine.

Run with **Save Version → Save & Run All (Commit)** so it keeps running with the
browser closed. Every training cell skips runs that already finished, so re-running
after an interruption resumes instead of starting over.

Estimated times on a T4 (rough; P100 is similar or faster). Total ≈ 3 to 4 h, well
inside Kaggle's 12 h session limit:

| Step | Estimate |
|---|---|
| Setup + data + VAD lexicon | 3 min |
| TF-IDF baseline (CPU) | 1 min |
| BiLSTM | 5 to 10 min |
| RoBERTa full fine-tune × 3 seeds | 30 to 45 min |
| RoBERTa LoRA × 3 seeds | 30 to 45 min |
| Ablations (pos_weight, focal, LoRA r16, BiLSTM max-pool) | 35 to 50 min |
| Embeddings for probing | 5 min |
| Token attributions | 2 min |
| Transfer (24 small runs) | 30 to 45 min |
| Analysis + zip | 5 to 10 min |

At the end, download **`circumplex_results.zip`** from the Output tab (that is the
file to send back). `circumplex_checkpoints.zip` holds model weights for the demo.

Repo: {REPO_URL}
"""),
    code("""
# 1. GPU check (about 5 s)
!nvidia-smi
import torch

print("torch", torch.__version__, "| CUDA available:", torch.cuda.is_available())
assert torch.cuda.is_available(), "Turn on a GPU accelerator in the notebook settings."
"""),
    code(f"""
# 2. Clone + install (about 1 to 2 min). Keeps Kaggle's own torch/CUDA.
%cd /kaggle/working
!rm -rf circumplex && git clone --depth 1 {REPO_URL} circumplex
%cd /kaggle/working/circumplex
!pip install -q -r requirements-kaggle.txt
!pip install -q -e . --no-deps
!git log --oneline -1
"""),
    code("""
# 3. Helpers: run a shell command and fail loudly; skip runs that already finished.
import json
import time
from pathlib import Path


def sh(cmd: str) -> None:
    print("$", cmd, flush=True)
    get_ipython().system(cmd)
    code = get_ipython().user_ns.get("_exit_code", 0)
    if code:
        raise RuntimeError(f"command failed with exit code {code}: {cmd}")


def train(config: str, name: str, seed: int = 42, extra: str = "") -> None:
    if Path(f"runs/{name}/metrics.json").exists():
        print(f"skip {name} (already done)")
        return
    t0 = time.time()
    sh(f"python -m circumplex.train --config {config} --set name={name} seed={seed} {extra}")
    print(f"== {name} finished in {(time.time() - t0) / 60:.1f} min")


def show(name: str) -> None:
    m = json.loads(Path(f"runs/{name}/metrics.json").read_text())
    t = m["tuned"]
    print(f"{name}: test macro-F1 tuned {t['macro_f1']:.4f} "
          f"[{t['macro_f1_ci']['lo']:.4f}, {t['macro_f1_ci']['hi']:.4f}] | @0.5 {m['fixed_0.5']['macro_f1']:.4f} "
          f"| micro {t['micro_f1']:.4f} | {m['train_seconds'] / 60:.1f} min | trainable {m['params']['trainable']:,}")
"""),
    code("""
# 4. Data check + NRC VAD lexicon -> data/emotion_vad.csv (about 1 min)
sh("python -c \\"from circumplex.data import load_goemotions; s = load_goemotions(); print({k: len(v) for k, v in s.items()})\\"")
try:
    sh("python scripts/build_vad.py")
except RuntimeError:
    print("!! NRC VAD download failed. RQ2/RQ3 distance analyses will be skipped; everything else still runs.")
"""),
    code("""
# 5. TF-IDF + logistic regression baseline (CPU, about 1 min)
if not Path("runs/tfidf_lr/predictions.npz").exists():
    sh("python -m circumplex.baselines --config configs/tfidf_lr.yaml")
show("tfidf_lr")
"""),
    code("""
# 6. BiLSTM from scratch (about 5 to 10 min)
train("configs/bilstm.yaml", "bilstm")
show("bilstm")
"""),
    code("""
# 7. RoBERTa-base: full fine-tuning and LoRA, seed 42 (about 25 to 30 min)
train("configs/transformer_full.yaml", "roberta_full_s42", 42)
show("roberta_full_s42")
train("configs/transformer_lora.yaml", "roberta_lora_s42", 42)
show("roberta_lora_s42")
"""),
    code("""
# 8. Extra seeds for mean ± std (about 50 to 60 min). Set to False if short on time.
RUN_EXTRA_SEEDS = True
if RUN_EXTRA_SEEDS:
    for seed in (43, 44):
        train("configs/transformer_full.yaml", f"roberta_full_s{seed}", seed, "output.save_checkpoint=false")
        train("configs/transformer_lora.yaml", f"roberta_lora_s{seed}", seed, "output.save_checkpoint=false")
"""),
    code("""
# 9. Ablations (about 35 to 50 min). Set to False if short on time.
RUN_ABLATIONS = True
if RUN_ABLATIONS:
    for cfg in ("roberta_pos_weight", "roberta_focal", "lora_r16", "bilstm_maxpool"):
        name = {"lora_r16": "abl_lora_r16_qkvo", "bilstm_maxpool": "abl_bilstm_maxpool"}.get(cfg, f"abl_{cfg}")
        train(f"configs/ablations/{cfg}.yaml", name, 42, "output.save_checkpoint=false")
"""),
    code("""
# 10. RQ3 embeddings: fine-tuned vs pretrained-only vs random-init encoder (about 5 min)
if not Path("runs/roberta_full_s42/embeddings_random_test.npz").exists():
    sh("python scripts/extract_embeddings.py --run runs/roberta_full_s42")
"""),
    code("""
# 11. Token attributions (Integrated Gradients) for curated examples (about 2 min)
sh("python scripts/attributions.py --run runs/roberta_full_s42 --out figures/attributions.html")
"""),
    code("""
# 12. RQ5 transfer to thinking traps: 2 inits x 4 fractions x 3 seeds (about 30 to 45 min)
if not Path("runs/transfer/metrics.json").exists():
    sh("python -m circumplex.transfer --config configs/transfer.yaml")
"""),
    code("""
# 12b. Worked traces of one example through the trained models (for LEARNING.md)
sh("python scripts/trace_example.py --run runs/bilstm > runs/traces.txt")
sh("python scripts/trace_example.py --run runs/roberta_full_s42 >> runs/traces.txt")
print(Path("runs/traces.txt").read_text())
"""),
    code("""
# 13. All analyses + figures (about 5 to 10 min; 10,000-permutation tests)
sh("python scripts/run_analysis.py --main-run roberta_full_s42")
print(Path("runs/analysis/results_table.md").read_text())
sh("python scripts/check_leakage.py > /dev/null")
print(Path("runs/analysis/leakage.json").read_text()[:1500])
"""),
    code("""
# 14. Package outputs. circumplex_results.zip is small: send this one back.
#     circumplex_checkpoints.zip has the seed-42 weights + tokenizers for the demo.
%cd /kaggle/working/circumplex
!rm -f /kaggle/working/circumplex_results.zip /kaggle/working/circumplex_checkpoints.zip
!zip -qr /kaggle/working/circumplex_results.zip runs figures data/emotion_vad.csv -x "*.pt" -x "*embeddings_*.npz" -x "runs/*/tokenizer/*"
!zip -qr /kaggle/working/circumplex_checkpoints.zip runs/roberta_full_s42 runs/roberta_lora_s42 runs/bilstm -x "*embeddings_*.npz"
!ls -lh /kaggle/working/*.zip
"""),
]

ANALYSIS = [
    md("""
# Circumplex: analysis and figures

Regenerates every README figure and statistic from `runs/` (unzip
`circumplex_results.zip` from Kaggle at the repo root first). Runs on CPU in a few
minutes. Sections whose inputs are missing are skipped and listed under `status`.
"""),
    code("""
import json
from pathlib import Path

from IPython.display import HTML, Image, Markdown, display

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
%cd {ROOT}
"""),
    code("""
!python scripts/run_analysis.py --main-run roberta_full_s42
A = json.loads(Path("runs/analysis/analysis.json").read_text())
A["status"]
"""),
    md("## RQ1: models"),
    code("""
p = Path("runs/analysis/results_table.md")
display(Markdown(p.read_text() if p.exists() else "_no runs yet_"))
for t in A.get("pairwise_tests", []):
    print(f"{t['a']} vs {t['b']}: Δ macro-F1 = {t['delta']:+.4f} [{t['lo']:+.4f}, {t['hi']:+.4f}], permutation p = {t['perm_p_value']:.4g}")
for f in ("class_distribution", "rq1_per_class_f1", "calibration"):
    if Path(f"figures/{f}.png").exists():
        display(Image(f"figures/{f}.png"))
"""),
    md("## RQ2: confusion vs valence/arousal distance"),
    code("""
for k in ("rq2_model_vs_distance", "rq2_human_vs_distance", "rq2_model_vs_human"):
    print(k, A.get(k, "TBD"))
for f in ("rq2_model_confusion_vs_distance", "rq2_human_confusion_vs_distance", "rq2_model_confusion", "rq2_human_confusion"):
    if Path(f"figures/{f}.png").exists():
        display(Image(f"figures/{f}.png"))
"""),
    md("## RQ3: does the embedding space recover valence/arousal?"),
    code("""
print(json.dumps(A.get("rq3", "TBD"), indent=1, default=str)[:3000])
for f in ("rq3_layerwise_valence", "rq3_layerwise_arousal", "rq3_pca_finetuned", "rq3_pca_base", "rq3_pca_random"):
    if Path(f"figures/{f}.png").exists():
        display(Image(f"figures/{f}.png"))
"""),
    md("## RQ4: mixed emotions"),
    code("""
print(A.get("rq4_human_vs_model", "TBD"))
for f in ("rq4_human_npmi", "rq4_model_npmi", "rq4_cardinality"):
    if Path(f"figures/{f}.png").exists():
        display(Image(f"figures/{f}.png"))
"""),
    md("## RQ5: transfer to cognitive distortions"),
    code("""
print(A.get("rq5_paired_differences", "TBD"))
if Path("figures/rq5_learning_curves.png").exists():
    display(Image("figures/rq5_learning_curves.png"))
"""),
    md("## Token attributions and error analysis"),
    code("""
if Path("figures/attributions.html").exists():
    display(HTML(Path("figures/attributions.html").read_text()))
errs = Path("runs/roberta_full_s42/errors.jsonl")
if errs.exists():
    for line in errs.read_text().splitlines()[:15]:
        e = json.loads(line)
        print(f"{e['wrong_prob']:.2f} said {e['wrong_label']:<14} gold {e['gold']}  | {e['text'][:90]}")
"""),
]


def main() -> None:
    """Write both notebooks."""
    out = Path(__file__).resolve().parents[1] / "notebooks"
    out.mkdir(exist_ok=True)
    for name, cells in (("train_kaggle.ipynb", TRAIN), ("analysis.ipynb", ANALYSIS)):
        (out / name).write_text(json.dumps(notebook(cells), indent=1) + "\n")
        print("wrote", out / name)


if __name__ == "__main__":
    main()
