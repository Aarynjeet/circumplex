# Model card: Circumplex emotion classifiers

## Model details

- **Models:** (1) TF-IDF + one-vs-rest logistic regression; (2) BiLSTM with
  attention pooling, trained from scratch; (3) `roberta-base` with a linear
  28-way sigmoid head, both fully fine-tuned and with LoRA (r=8 on query/value).
- **Task:** multi-label classification of short English text into 27 emotions +
  neutral. Each label gets an independent probability. Per-class thresholds,
  tuned on the dev set, turn probabilities into labels.
- **Training code:** plain PyTorch loop in `src/circumplex/train.py` (no
  high-level trainer). Configs in `configs/`.
- **Version:** 0.1.

## Intended use

- Research and teaching: studying how text classifiers represent emotion, and
  comparing their errors with psychological theory.
- Aggregate, non-consequential analysis of public text where errors cost
  nothing, e.g. exploring a corpus.

## Out-of-scope uses (do not use for these)

- Any clinical, diagnostic, triage or mental-health screening purpose,
  including detecting depression, suicidality, distress or "cognitive
  distortions" in real people.
- Decisions about individuals: hiring, moderation penalties, insurance,
  education, law enforcement, surveillance.
- Profiling or tracking individual users over time.
- Languages other than English, or domains far from Reddit comments (clinical
  notes, legal text, long documents).

## Training data

- **GoEmotions** (Demszky et al., 2020), official "simplified" splits: 43,410
  train / 5,426 dev / 5,427 test Reddit comments, with labels kept when at least
  2 raters agreed. Names and religion terms are masked in the source data.
  Apache-2.0.
- **Imbalance:** `neutral` appears in 32.8% of training examples, `grief` in
  0.2% (77 examples). The ratio between the most and least frequent label is
  about 185×.
- **RQ5 only:** thinking-traps data from Sharma et al. (2023), CC BY-NC-ND 4.0,
  used for research experiments only.

## Metrics

Primary: macro-F1 over the 28 labels on the test set, with per-class thresholds
tuned on dev. Also reported: micro-F1, per-class precision/recall/F1, Ekman-
and sentiment-level F1, expected calibration error, 95% bootstrap CIs, and
paired permutation tests between models.

| Model | Test macro-F1 (95% CI) | Notes |
|---|---|---|
| TF-IDF + LR | 0.471 (0.448 to 0.490) | real CPU run |
| BiLSTM | TBD | |
| RoBERTa LoRA | TBD | |
| RoBERTa full FT | TBD | |

## Known biases and risks

- **Population:** Reddit users are not representative of the general population
  (commonly reported to skew young, male, US-based and English-speaking).
  Emotional expression, slang and sarcasm norms differ across communities, so
  accuracy will be uneven across dialects and groups. That unevenness is not
  measured here.
- **Rater bias:** labels come from a small pool of crowd raters (82 rater ids in
  the raw release). Their cultural reading of English Reddit text can differ
  from the authors' intent. *TBD: cite the paper's description of the rater pool.*
- **Label noise:** many categories have low inter-rater agreement. The model
  learns these ambiguities and inconsistencies.
- **Sarcasm and context:** comments are judged without their thread, so sarcasm
  and replies are often misread (see `runs/*/errors.jsonl`).
- **Rare emotions** (grief, pride, relief, nervousness) have few training
  examples and much less reliable predictions.

## Limitations

- Probabilities are calibrated only to the extent ECE shows, on this test
  distribution. They are not reliable confidence scores elsewhere.
- Valence/arousal outputs in the demo are a probability-weighted average of
  lexicon coordinates for emotion *words*, not a measurement of the text.
- Multi-seed variance for the transformer is reported where it was run. The
  BiLSTM and baseline are single-seed.
