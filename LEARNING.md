# LEARNING.md: how every piece works, with numbers

This is a guided tour of the codebase. Each section takes a file in the order
data flows through it, explains it chunk by chunk, and plugs in concrete
numbers. Where a number comes from a real run it is marked **(real)**. Toy
numbers chosen to make the arithmetic visible are marked **(toy)**. Every toy
calculation was checked in Python. Numbers that need the GPU runs are `TBD`.

Contents

1. [The big picture as an AST](#1-the-big-picture-as-an-ast)
2. [`labels.py` and `data.py`: from TSV to tensors](#2-labelspy-and-datapy-from-tsv-to-tensors)
3. [`models/bilstm.py`: one example, end to end](#3-modelsbilstmpy-one-example-end-to-end)
4. [`models/transformer.py`: the same example through RoBERTa](#4-modelstransformerpy-the-same-example-through-roberta)
5. [`train.py`: one training step by hand](#5-trainpy-one-training-step-by-hand)
6. [Why BCEWithLogitsLoss and not CrossEntropyLoss](#6-why-bcewithlogitsloss-and-not-crossentropyloss)
7. [`evaluate.py`: macro-F1, thresholds, calibration, bootstrap](#7-evaluatepy-macro-f1-thresholds-calibration-bootstrap)
8. [LoRA with real dimensions](#8-lora-with-real-dimensions)
9. [`baselines.py`: TF-IDF + logistic regression](#9-baselinespy-tf-idf--logistic-regression)
10. [`analysis/confusion.py`: soft confusion and the permutation test (RQ2)](#10-analysisconfusionpy-soft-confusion-and-the-permutation-test-rq2)
11. [`analysis/probing.py`: linear probes (RQ3)](#11-analysisprobingpy-linear-probes-rq3)
12. [`analysis/cooccurrence.py`: PMI (RQ4)](#12-analysiscooccurrencepy-pmi-rq4)
13. [`analysis/attribution.py`: Integrated Gradients](#13-analysisattributionpy-integrated-gradients)
14. [`transfer.py`: RQ5](#14-transferpy-rq5)
15. [Config, utils, scripts, tests, demo](#15-config-utils-scripts-tests-demo)
16. [25 interview questions with answers](#16-25-interview-questions-with-answers)

---

## 1. The big picture as an AST

If the project were a program's syntax tree, the root is an *experiment*, and
every node is a pure-ish transformation of its children's output:

```
Experiment(config.yaml)
├── Data
│   ├── load_goemotions()          -> Split(texts, label_ids, ids) x 3
│   ├── multi_hot()                -> Y: float32[N, 28]
│   └── tokenize + DataLoader      -> batches {input_ids[B,T], attention_mask[B,T], labels[B,28]}
├── Model                           (BiLSTMClassifier | TransformerClassifier)
│   └── forward(input_ids, mask)   -> logits[B, 28]
├── Train (train.fit)
│   └── for each batch: loss -> backward -> (every k) clip, step, schedule
├── Evaluate
│   ├── predict_proba              -> P: float[N, 28]
│   ├── tune_thresholds(dev)       -> t: float[28]
│   └── metrics(test, P >= t)      -> macro-F1, CI, ECE, ...
└── Analyse
    ├── soft_confusion, mantel_test         (RQ2)
    ├── extract_features, ridge_probe       (RQ3)
    ├── npmi, cardinality                   (RQ4)
    └── integrated_gradients
```

Two invariants make the tree composable:

* **Every model has the same signature**: `forward(input_ids, attention_mask) -> logits (B, 28)`.
  That's why one `fit()` trains both neural models, and why `predict_proba` works
  for both.
* **Everything after the model works on numpy arrays** `y (N, 28)`, `probs (N, 28)`,
  `pred (N, 28)`. That's why the TF-IDF baseline (which has no PyTorch at all)
  goes through exactly the same evaluation code.

---

## 2. `labels.py` and `data.py`: from TSV to tensors

### 2.1 The label space (`labels.py`)

`data/goemotions/emotions.txt` lists 28 labels in a fixed order: `admiration`
is index 0 and `neutral` is index 27. The order matters because the raw data
stores labels as indices, e.g. `"2"` = anger, `"27"` = neutral.

The Ekman mapping groups the 27 emotions into 6 (anger, disgust, fear, joy,
sadness, surprise), and the sentiment mapping into positive/negative/ambiguous.
Both are the *official files* from the GoEmotions release. `neutral` is added
as its own group so every label lands somewhere. `to_groups` maps a fine-label
matrix up a level with a logical OR, done as matrix multiplication:

```
y (N, 28) @ M (28, 7) -> counts (N, 7);   group on  <=>  count > 0
```

Example: a comment labelled `{anger, annoyance}` has two 1s in the `anger`
column of M, so its Ekman vector has `anger = 1` and nothing else.

### 2.2 Loading (`data.py: load_goemotions`)

A line of `train.tsv` **(real)**:

```
WHY THE FUCK IS BAYLESS ISOING	2	eezlygj
```

That's text, the label ids `"2"` (anger), and the comment id. Multi-label rows look
like `...\t0,27\t...`. `load_goemotions` parses each split into a `Split`
dataclass and **checks the sizes** against the official 43,410 / 5,426 / 5,427,
so a silently truncated download can't slip through.

### 2.3 Multi-hot (`multi_hot`)

```python
multi_hot([[0, 27], [3]])
# row 0: 1 at index 0 (admiration) and 27 (neutral); row 1: 1 at index 3 (annoyance)
# shape (2, 28), dtype float32
```

Float, not int, because `BCEWithLogitsLoss` wants float targets. It raises on
an out-of-range index or an empty label list: every GoEmotions example has at
least one label (neutral counts), so an empty row means a parsing bug.

### 2.4 The data, in numbers (real)

* `neutral`: 14,219 training examples (32.8%). `grief`: 77 (0.2%). Ratio ≈ 185×.
* Average labels per training example: 1.18. 16.4% of examples have more than one label.

That long tail is why accuracy is useless here (section 7) and why
per-class thresholds matter so much.

### 2.5 Tokenising and batching

* **BiLSTM:** `simple_tokenize` lowercases and splits into words and punctuation,
  keeping `[name]` and contractions intact. `Vocab.build` keeps words seen at
  least twice in *train* (13,644 words, **real**); index 0 is `<pad>`, 1 is `<unk>`.
* **Transformer:** the pretrained tokenizer (RoBERTa's byte-level BPE).

`EncodedDataset` stores variable-length id lists. `make_collate` pads each batch
to *its own* longest sequence (dynamic padding), so a batch of short comments
stays short. That's a free speed-up, since GoEmotions comments are short.

```
batch of 2: [5, 6, 7] and [8]
input_ids      = [[5, 6, 7], [8, 0, 0]]
attention_mask = [[1, 1, 1], [1, 0, 0]]
```

---

## 3. `models/bilstm.py`: one example, end to end

The text: **"Thank you so much, this made my day!"**

Below is a **real trace** from `scripts/trace_example.py --untrained-bilstm`: the
real vocabulary, but **untrained (random, seed 0) weights**. The shapes and ids
are exactly what training sees. The output values are meaningless *because the
model hasn't learned anything yet*, which is itself instructive. The trained
version of this trace is `TBD`, produced by the Kaggle notebook into
`runs/traces.txt`.

```
tokens        : ['thank', 'you', 'so', 'much', ',', 'this', 'made', 'my', 'day', '!']
token ids     : [93, 8, 28, 94, 5, 18, 181, 23, 154, 10]
input_ids     : (1, 10)                     # batch=1, seq_len=10
embeddings    : (1, 10, 300)                # each id -> a 300-dim learned vector
LSTM output   : (1, 10, 512)                # 256 forward + 256 backward, per token
attn weights  : thank 0.105, you 0.105, so 0.11, much 0.10, ',' 0.10, this 0.094, ...
pooled        : (1, 512)                    # attention-weighted average over the 10 tokens
logits        : (1, 28)                     # e.g. excitement 0.08, disapproval -0.07, ...
sigmoid top-5 : excitement 0.5197, optimism 0.5135, fear 0.5135, approval 0.5133, surprise 0.5128
predicted     : 15 of 28 labels at threshold 0.5  # every logit is near 0, so every p is near 0.5
```

Read it like a pipeline:

1. **Embedding** `nn.Embedding(13644, 300)` is a lookup table: row 93 is the
   vector for "thank". (B, T) → (B, T, E).
2. **Packing**: `pack_padded_sequence` tells the LSTM each sequence's true
   length, so in a padded batch it never reads pad positions. The test
   `test_bilstm_ignores_padding` checks that adding pads doesn't change the output.
3. **BiLSTM** reads left→right and right→left. At "day", the forward state has
   seen "thank you so much, this made my day" and the backward state has seen
   "day !". Concatenated, that's 512 numbers per token.
4. **Attention pooling**: score each token with `w · tanh(W h_t)`, softmax over the
   real (non-pad) tokens, take the weighted average. Untrained, the weights are
   near-uniform (≈ 1/10 each, as above). After training they should concentrate on
   words like "thank" (gratitude). Max pooling (the ablation) instead takes, per
   feature, the largest value over time.
5. **Head** `Linear(512, 28)`: 28 logits, one independent score per label.
6. **Sigmoid** turns each logit into a probability *independently*. They don't sum
   to 1. Untrained, all logits ≈ 0 so all p ≈ 0.5, and a 0.5 threshold switches on
   15 labels. Training pushes most logits strongly negative.
7. **Thresholds**: per-class values tuned on dev (section 7) decide which labels
   are on.

---

## 4. `models/transformer.py`: the same example through RoBERTa

`TransformerClassifier` = HF encoder + **our own** dropout + `Linear(768, 28)`.
We write the head ourselves (rather than `AutoModelForSequenceClassification`)
so that the exact vector used for prediction is also the one we probe (RQ3) and
attribute (IG).

Shapes for roberta-base (the token ids are `TBD`, from `runs/traces.txt`, since
the RoBERTa tokenizer couldn't be downloaded in the build sandbox):

```
tokens        : <s> Thank Ġyou Ġso Ġmuch , Ġthis Ġmade Ġmy Ġday ! </s>   # Ġ = "preceded by a space"
input_ids     : (1, 12)
word embeds   : (1, 12, 768)       # token + position embeddings, then LayerNorm
12 layers     : each (1, 12, 768)  # self-attention + feed-forward, residuals, LayerNorm
hidden_states : 13 x (1, 12, 768)  # layer 0 = embeddings, then layers 1..12
<s> vector    : (1, 768)           # first token of the last layer = sentence summary
logits        : (1, 28)            # TBD values
sigmoid       : (1, 28)            # TBD
predicted     : TBD                # likely {gratitude, joy} or similar; must be checked
```

Why the first token? RoBERTa has no next-sentence objective, so `<s>` isn't
specially pretrained. But fine-tuning trains the head on it, and self-attention
lets it gather information from every token. `all_layer_features` also returns
the mean over tokens, because for the *not fine-tuned* model (RQ3's control)
mean pooling is usually the better sentence summary.

`build_encoder("tiny-random")` builds a 2-layer, 32-dim BERT with random weights
and `train_tiny_tokenizer` trains a 2,000-word WordPiece vocabulary on the
smoke-test texts. Together they let every transformer code path (including
LoRA, checkpoint reload, embeddings and IG) run offline on a CPU in seconds.

---

## 5. `train.py`: one training step by hand

### 5.1 A tiny batch, by hand (toy, checked in Python)

Two examples, three labels, two input features, and a linear "model"
`logits = x Wᵀ + b`:

```
x = [[1.0,  2.0],      y = [[1, 0, 0],      W = [[ 0.1, -0.2],    b = [0, 0, 0]
     [0.5, -1.0]]           [0, 1, 1]]           [ 0.0,  0.3],
                                                 [-0.1,  0.1]]
```

**Forward.** Row 1: `[1·0.1 + 2·(-0.2), 1·0 + 2·0.3, 1·(-0.1) + 2·0.1] = [-0.3, 0.6, 0.1]`.

```
logits  = [[-0.30,  0.60,  0.10],      sigmoid = [[0.4256, 0.6457, 0.5250],
           [ 0.25, -0.30, -0.15]]                 [0.5622, 0.4256, 0.4626]]
```

**Loss.** BCE per (example, label): `-[y log p + (1-y) log(1-p)]`.
Example 1, label 0: y=1, p=0.4256, so `-log 0.4256 = 0.8544`.

```
per-term = [[0.8544, 1.0375, 0.7444],
            [0.8259, 0.8544, 0.7710]]       mean over 6 terms -> loss = 0.8479
```

**`.backward()`.** Autograd walks the graph in reverse, applying the chain rule.
For sigmoid + BCE the gradient w.r.t. each logit simplifies beautifully to
`(p - y) / 6` (the 6 is from the mean):

```
dL/dlogits = [[-0.0957,  0.1076,  0.0875],
              [ 0.0937, -0.0957, -0.0896]]
```

Then `dL/dW[j,k] = Σ_n dL/dlogit[n,j] · x[n,k]`. For W[0,0]:
`-0.0957·1.0 + 0.0937·0.5 = -0.0489`. So `W.grad[0,0] = -0.0489`. Negative means
increasing W[0,0] lowers the loss: it raises the logit for label 0 on
example 1, whose true label 0 is *on*.

**What `.backward()` really computes:** it fills `param.grad` for every parameter
with ∂loss/∂param, **adding** to whatever is already there. That accumulation
behaviour is why `optimizer.zero_grad()` exists, and it's exactly what gradient
accumulation exploits.

**The optimiser step (AdamW, lr = 0.01, first step).** Adam keeps running averages
m (of the gradient) and v (of the squared gradient), bias-corrected:

```
m = 0.1 · g = -0.00489          m̂ = m / (1 - 0.9)  = -0.0489
v = 0.001 · g² = 2.39e-6        v̂ = v / (1 - 0.999) = 0.00239
step = lr · m̂ / (√v̂ + ε) = 0.01 · (-0.0489 / 0.0489) = -0.01
W[0,0]: 0.1 -> 0.1 - (-0.01) = 0.11       (then decoupled weight decay: -lr·wd·W = -1e-5)
```

On the first step Adam moves every weight by about `lr`, whatever the size of its
gradient. That's why the learning rate needs a **warmup**: early steps with a
full learning rate on random-ish head weights can wreck pretrained features.

### 5.2 The real loop (`fit`), chunk by chunk

```python
with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
    logits = model(input_ids, attention_mask)
    loss = loss_fn(logits.float(), labels)
scaler.scale(loss / grad_accum_steps).backward()
if i % grad_accum_steps == 0:
    scaler.unscale_(optimizer)
    clip_grad_norm_(model.parameters(), max_grad_norm)
    scaler.step(optimizer); scaler.update()
    scheduler.step(); optimizer.zero_grad()
```

* **Mixed precision (why).** On a T4, fp16 matrix multiplies run on tensor cores
  several times faster than fp32 and use half the memory. `autocast` runs each op in
  fp16 where that's safe (matmuls) and fp32 where it isn't (reductions, softmax).
  The loss is computed in fp32 (`logits.float()`).
* **GradScaler (why).** fp16's smallest normal number is about 6e-5. Gradients like
  1e-7 underflow to 0. The scaler multiplies the loss by, say, 65,536 before
  `backward`, so gradients are big enough to represent. It divides them back
  (`unscale_`) before clipping and stepping. If any gradient overflowed to inf/nan,
  it **skips that step** and lowers the scale.
* **Gradient accumulation (why).** Want an effective batch of 64 but only 32 fits in
  memory? Run two batches of 32, call `backward` on each with `loss / 2`, and step
  once. Because `.grad` *adds up*, the result equals one batch of 64. The test
  `test_grad_accumulation_matches_full_batch` checks this numerically.
* **Gradient clipping.** If the total gradient norm exceeds 1.0, scale all gradients
  down so it equals 1.0. That prevents one weird batch from taking a huge step.
* **Schedule.** `linear_warmup_decay`: the LR multiplier goes 0→1 over the first 10% of
  steps, then down linearly to 0. Test: with 2 warmup steps of 6, the multipliers
  are `[0.5, 1.0, 1.0, 0.75, 0.5, 0.25]`.
* **AdamW parameter groups.** No weight decay on biases and LayerNorm weights.
  Shrinking those toward 0 doesn't regularise anything useful.
* **Evaluation + early stopping.** A few times per epoch: dev loss, dev macro-F1 at
  0.5, and dev macro-F1 with thresholds tuned on dev. The best checkpoint is chosen
  by the *tuned* score, because that's how the model will be used. Patience = 2
  evaluations without improvement stops training.
* **Checkpointing.** Only trainable weights are saved. For LoRA that's about 1.3 MB
  instead of 500 MB. `load_trained` rebuilds the model and checks that every
  trainable weight was found.

### 5.3 The sanity test that matters most

`test_loop_overfits_a_batch_of_8`: a model that can't drive the loss on 8
examples close to 0 has a bug (wrong targets, detached graph, LR schedule stuck
at 0, ...). The test requires the loss to fall below 0.02.

### 5.4 Class imbalance options (ablations)

* `pos_weight = #neg / #pos` per label (capped at 30) multiplies the loss on
  *positive* examples. For grief in train: 77 positives, 43,333 negatives →
  ratio 563, capped to 30. A missed grief costs 30× a false alarm.
* **Focal loss** multiplies each term by `(1 - p_t)^γ`. A negative the model already
  gets right with p_t = 0.95 is down-weighted by `0.05² = 0.0025`, so the thousands
  of easy negatives stop drowning out the hard cases. With γ = 0 it equals BCE
  (tested).

---

## 6. Why BCEWithLogitsLoss and not CrossEntropyLoss

`CrossEntropyLoss` applies **softmax**, which forces the 28 probabilities to sum to
1: the labels compete. Take example 2 above, whose gold labels are {1, 2}.
Softmax of its logits gives `[0.445, 0.257, 0.298]`. Even a perfect model could
give at most 0.5 to each of two correct labels, and the target "one class" doesn't
even exist. GoEmotions is **multi-label** (16% of examples have 2+ labels): "gratitude
*and* joy" is a correct answer, not a contradiction.

`BCEWithLogitsLoss` treats each label as its own yes/no question with its own
sigmoid. It's numerically safer than `sigmoid` + `BCELoss`, because it uses the
log-sum-exp trick on the logits directly (no `log(0)`), and it plays well with
fp16 autocast.

---

## 7. `evaluate.py`: macro-F1, thresholds, calibration, bootstrap

### 7.1 Why macro-F1 and not accuracy

* **Exact-match accuracy** (all 28 bits right) is brutal and uninformative.
* **Per-label accuracy** is trivially high: a model that never predicts grief is
  99.8% "accurate" on grief.
* **F1 = 2PR / (P + R)** ignores true negatives, so it only rewards finding the label.
* **Macro**-F1 averages per-class F1, so grief counts as much as neutral. That's the
  right incentive for a study of *subtle* emotions. **Micro**-F1 pools all decisions
  and is dominated by frequent labels; it's reported too.

### 7.2 Per-class threshold tuning (real numbers)

For each label, try thresholds 0.02 to 0.95 on **dev** and keep the one with the best
F1. Since class j's F1 depends only on column j, this maximises macro-F1 over the
grid. Test is then scored once with those thresholds.

The TF-IDF baseline (C = 1, no class weighting) shows why this matters **(real)**:

| | test macro-F1 | labels predicted per example |
|---|---|---|
| threshold 0.5 | 0.287 | 0.57 (humans: 1.17) |
| dev-tuned per class | **0.471** | 1.53 |

Logistic regression trained on a 0.2%-positive label rarely outputs p > 0.5 for
it, so at 0.5 most rare labels are *never* predicted (F1 = 0). The tuned thresholds
for rare labels are much lower.

### 7.3 Calibration (ECE)

Treat each of the N × 28 sigmoid outputs as a forecast. Put them in 15 bins by
predicted probability, then compare each bin's average prediction with how often
the label was really on:

```
ECE = Σ_bins (n_bin / n) · |accuracy_bin - confidence_bin|
```

Toy (tested): a model always saying 0.9 on labels that are on half the time has
ECE = |0.5 - 0.9| = 0.4. The TF-IDF baseline's test ECE is **0.008** **(real)**.
That's very low, but most of its outputs are near 0 for labels that are off, and
those bins are easy to get right. The reliability diagram shows the bins
separately.

### 7.4 Bootstrap confidence intervals

Resample the 5,427 test examples *with replacement* 1,000 times and recompute
macro-F1 each time. The 2.5th and 97.5th percentiles form the 95% CI. The baseline
gets **[0.448, 0.490]** **(real)**. Implementation trick: instead of copying arrays,
draw how many times each example is picked (multinomial counts) and weight the
TP/FP/FN sums.

### 7.5 Comparing two models

* **Paired bootstrap:** the *same* resample is scored for both models, so example
  difficulty cancels out, and we get a CI for the difference.
* **Approximate randomization:** under "the two models are equally good", which
  model made which prediction on an example is a coin flip. Swap predictions per
  example at random 10,000 times and count how often |Δ| is at least as large as
  observed. p = (hits + 1) / (10,000 + 1).

---

## 8. LoRA with real dimensions

A RoBERTa-base attention projection (query or value) is a **768 × 768** matrix W:
589,824 parameters. Full fine-tuning learns an update ΔW of the same size.

LoRA says: learn ΔW as a product of two thin matrices,

```
ΔW = (α / r) · B A,    A: r × 768,   B: 768 × r
```

With **r = 8**: A has 8 × 768 = 6,144 numbers and B has 768 × 8 = 6,144, so
**12,288 per matrix**, about **2.1%** of 589,824. B starts at zero, so ΔW = 0 at
step 0 and training starts exactly at the pretrained model.

For the whole model (query + value in each of 12 layers = 24 matrices):
`24 × 12,288 = 294,912` adapter parameters, plus our head `768 × 28 + 28 = 21,532`,
gives **316,444 trainable**, versus about 124M for full fine-tuning (the exact
counts are written to `metrics.json` by each run). The LoRA unit test checks this
formula on the tiny model: 4 matrices × 256 + head 924 = 1,948.

**Why it works:** fine-tuning updates have been found to be approximately low rank.
Adapting a pretrained model to one task mostly needs a few new "directions" in
each layer, not a full re-learning of 590k numbers. Rank 8 means ΔW can change
how W acts on at most 8 directions of the input.

**Trade-offs to report (TBD):** macro-F1 (LoRA is often close to full fine-tuning,
sometimes slightly worse), trainable parameters (~400× fewer), checkpoint size, and
training time. Time savings are smaller than the parameter savings, because the
forward and backward passes still run through the whole frozen network. The LR
needs to be higher (5e-4 vs 3e-5) because so few weights are learning, starting
from zero.

---

## 9. `baselines.py`: TF-IDF + logistic regression

* **TF-IDF:** each comment becomes a sparse vector over 1- to 2-word n-grams (100k
  features) plus character 2- to 5-grams inside words (100k), weighted by sublinear
  term frequency × inverse document frequency. Char n-grams catch "thx",
  "thaaaanks" and typos.
* **One-vs-rest:** 28 independent logistic regressions on the same features. That's
  the classical version of "28 sigmoids".
* **Tuned on dev** (`runs/tfidf_sweep.json`, real): C ∈ {0.5, …, 8}, class_weight
  ∈ {none, balanced}. Best C = 1, no weighting, dev tuned macro-F1 = 0.513. Every
  setting was within 0.02 once thresholds were tuned, because class weighting and
  threshold tuning fix the same problem.
* Trainable parameters: 28 × (feature count + 1) = **4,079,852** **(real)**.

---

## 10. `analysis/confusion.py`: soft confusion and the permutation test (RQ2)

### 10.1 Soft confusion, defined precisely

Single-label confusion matrices don't work when an example can have several
labels. For gold label i and another label j:

```
C[i, j] = P( model predicts j AND misses i  |  i is gold AND j is not gold )
```

Hand-computed example (the unit test) with 3 labels and 4 examples:

```
y    = [[1,0,0],   pred = [[0,1,0],   # missed 0, said 1
        [1,0,0],           [1,0,0],   # correct
        [1,1,0],           [0,1,1],   # missed 0; said 1 (gold, so not a confusion) and 2
        [0,1,0]]           [0,0,1]]   # missed 1, said 2
```

* C[0,1]: rows where 0 is gold and 1 is not = {row 0, row 1}. Of those, missed 0 and
  said 1: row 0. **1/2.**
* C[0,2]: rows where 0 is gold and 2 is not = {0, 1, 2}. Missed 0 and said 2: row 2. **1/3.**
* C[1,2]: rows {2, 3}. Row 2 got label 1 right; row 3 missed 1 and said 2. **1/2.**
* Row 2 is undefined (label 2 is never gold), and so is the diagonal.

Conditioning on "j is not gold" matters. In row 2, predicting label 1 is
*correct*, so it mustn't count as confusing 0 with 1.

**Human version:** each rater is treated as a "predictor" of the consensus label,
using the same formula over (comment, rater) pairs. Real result: the top
rater confusions are grief→sadness (0.065), remorse→sadness (0.063),
confusion↔curiosity (0.056) and anger↔annoyance (0.050) **(real)**.

### 10.2 The permutation (Mantel) test with toy numbers

Hypothesis: confusion is higher for emotions close in valence/arousal (VA) space,
so **ρ(confusion, distance) < 0**.

Toy (checked in Python): 4 emotions with made-up coordinates and symmetrised
confusion:

| pair | VA distance | confusion |
|---|---|---|
| anger / annoyance | 0.224 | 0.08 |
| joy / love | 0.206 | 0.06 |
| annoyance / joy | 0.608 | 0.02 |
| anger / joy | 0.707 | 0.01 |
| annoyance / love | 0.658 | 0.01 |
| anger / love | 0.808 | 0.00 |

Spearman ρ = **-0.928**. Is that surprising? The 6 pairs are not independent
(each emotion appears in 3), so a textbook p-value for ρ is wrong. Instead:
**shuffle which emotion owns which coordinates**, recompute ρ, and repeat. With 4
emotions there are only 4! = 24 relabellings. Exactly 2 of them give ρ ≤ -0.928, so
p = 2/24 = **0.083**: not significant, however strong ρ looks, because 4 emotions
can't produce stronger evidence than 1 in 24. With 27 emotions there are 27! ≈ 10²⁸
relabellings, so we sample 10,000, and `p = (hits + 1) / (10,001)`.

The tests check that the procedure finds a real effect (p < 0.01 on synthetic data
built with `confusion = exp(-4 · distance)`) and that p-values are roughly
uniform (mean ≈ 0.5) when there is no effect.

Real results: model ρ, human ρ and model-vs-human ρ are **TBD** (they need the NRC
VAD table and the trained model).

---

## 11. `analysis/probing.py`: linear probes (RQ3)

**The question:** is valence *linearly readable* from the 768-dim sentence vector?

**Target:** each example's average lexicon (valence, arousal) over its gold
emotions. Toy: gold {joy, love} with joy = (0.9, 0.7) and love = (1.0, 0.5) gives
the target (0.95, 0.6). Neutral-only examples are excluded.

**Probe:** standardise the features, fit ridge regression on dev embeddings
(alpha chosen by internal cross-validation), and report **R² on test**. Toy linear
probe (checked): 4 examples with 2 features `X = [[1,0],[2,1],[0,1],[3,2]]` and
targets `[0.7, 0.9, 0.3, 1.2]`. Least squares gives weights `[0.317, -0.067]`,
bias 0.367, predictions `[0.683, 0.933, 0.300, 1.183]`, and
R² = 1 - SS_res/SS_tot = **0.996**. On real data, R² is computed on held-out
examples, so it can't be inflated by memorising the training points.

**Why "linear"?** A powerful nonlinear probe can decode almost anything from
almost any representation, including from random networks. A linear probe asks
whether the information is *laid out* along a direction, which is the claim
"the space has a valence axis".

**The circularity trap, and its controls.** The fine-tuned model was trained to
output the 28 labels, and our target is computed *from those labels*. So high R²
for the fine-tuned model partly just says "it knows the labels". The controls:

| Probe input | What a high R² would mean |
|---|---|
| random-init RoBERTa | the architecture alone (e.g. word identity) carries VA: floor |
| pretrained-only RoBERTa | VA structure exists **before** any emotion training |
| fine-tuned RoBERTa | VA after emotion training |
| fine-tuned model's 28 output probabilities | VA recoverable from labels alone |

The interesting comparison is **pretrained vs random** (did language-model
pretraining already organise affect?) and **embedding vs its own outputs**
(does the 768-d space hold VA structure beyond the 28 labels?). **Layer-wise**
probing shows *where* valence becomes decodable. All numbers are **TBD**.

**PCA of emotion means:** average the embeddings of all examples carrying each
emotion, giving 27 points in 768-d. PCA projects them to 2-d. If PC1 correlates
with lexicon valence across the 27 emotions (permutation p-value as in
section 10), the space's main axis of variation is valence. UMAP is not run on the
27 means (27 points is too few for a neighbour graph) and is left optional.

---

## 12. `analysis/cooccurrence.py`: PMI (RQ4)

`PMI(i, j) = log P(i, j) / (P(i) P(j))`. It's 0 when labels co-occur at chance
level, positive when they co-occur more often. NPMI divides by `-log P(i, j)` to
land in [-1, 1]: 1 means always together, -1 means never.

Toy (tested): labels A and B always appear together, so NPMI = 1. A and C have
P(A, C) = P(A)P(C), so NPMI = 0.

Real human co-labelling (train): the top pairs are fear/nervousness (NPMI 0.38),
grief/sadness (0.30), anger/annoyance (0.22), disappointment/sadness (0.21) and
confusion/curiosity (0.21) **(real)**. These are VA-adjacent pairs, which is
encouraging for RQ2. The model side (does it co-predict the same pairs? does it
predict as many multi-label examples?) is **TBD**.

---

## 13. `analysis/attribution.py`: Integrated Gradients

To ask "which words made the model say *gratitude*?", the plain gradient
∂logit/∂embedding is noisy and *saturates*: once the model is sure, small
changes don't matter, so the gradient is ~0 even for the key word.

IG integrates the gradient along a straight path from a **baseline** x′ (pad
tokens everywhere except `<s>`/`</s>`) to the real input x:

```
IG_i = (x_i - x'_i) · average over k=1..m of ∂F(x' + (k/m)(x - x')) / ∂x_i
```

summed over the 768 embedding dimensions to give one score per token.
**Completeness:** the scores add up to `F(x) - F(x′)`. The code reports the gap
as a correctness check, and the unit test requires it to be small. It's
implemented in about 30 lines of PyTorch: one batched forward and backward over all
m interpolation steps.

---

## 14. `transfer.py`: RQ5

* **Data (real):** 1,077 lines → 698 unique thoughts, 14 traps after merging
  spelling variants ("Overgeneralization"/"Overgeneralizing"), 1.54 traps per
  thought on average. A fixed 70/15/15 split by unique thought gives about 488 / 105
  / 105.
* **Design:** for each fraction (10/25/50/100%) and seed (0/1/2), train two models
  that differ *only* in the encoder's starting weights: vanilla roberta-base vs the
  GoEmotions-fine-tuned encoder. Both get a fresh head, the same data subset and the
  same seed.
* **Threshold:** one global threshold tuned on dev. Per-class tuning on a 105-example
  dev set would overfit.
* **Analysis:** paired per-seed differences (emotion - vanilla) per fraction. The
  hypothesis predicts the largest gain at 10%. With about 105 test examples, expect
  large variance and say so.

---

## 15. Config, utils, scripts, tests, demo

* **`config.py`:** YAML → nested dataclasses. Unknown keys raise (a typo like
  `learning_rate:` fails loudly instead of being ignored). `base:` gives
  inheritance (`transformer_lora.yaml` = full config + LoRA changes), and
  `--set train.lr=1e-5` gives overrides. A YAML quirk the tests caught: `1e-4`
  (no decimal point) loads as a *string* in YAML 1.1, so numeric fields are
  coerced by their declared type.
* **`utils.py`:** `set_seed` (python, numpy, torch CPU+CUDA, cuDNN deterministic),
  `RunDir` (writes `config.yaml` with git hash and GPU name, `metrics.json`,
  `curve.csv`), and `count_parameters`.
* **`scripts/`:** thin CLIs (build VAD table, tune baseline, extract embeddings,
  attributions, analysis, traces, notebook generation, Space packaging).
* **`tests/`:** 56 tests. The ones worth knowing for interviews: overfit-8, grad
  accumulation = big batch, the hand-computed soft confusion matrix, the Mantel
  test on synthetic data with a known answer (plus p-value uniformity under the
  null), the LoRA parameter count, IG completeness, and the checkpoint round trip
  (train → save → reload → identical test probabilities). That last one caught a
  real bug: a LoRA checkpoint of the random tiny model lost its frozen base.
* **`demo/app.py`:** Gradio UI showing the top emotions, the probability-weighted
  position on the VA plane, and IG highlights for the top emotion. The disclaimer
  is in the UI itself.

---

## 16. 25 interview questions with answers

**ML fundamentals**

1. **Why sigmoid + BCE instead of softmax + cross-entropy?**
   The task is multi-label: 16% of training comments have 2+ labels. Softmax makes
   labels compete for one unit of probability. Independent sigmoids let "gratitude"
   and "joy" both be near 1.

2. **What does `loss.backward()` do?**
   It traverses the autograd graph from the loss back to the leaves, applying the
   chain rule, and *adds* ∂loss/∂param into each `param.grad`. It doesn't update
   weights; `optimizer.step()` does. Since it accumulates, we zero gradients
   between steps, and gradient accumulation relies on exactly this.

3. **Why mixed precision, and what does GradScaler do?**
   fp16 doubles throughput on tensor cores and halves activation memory. But small
   gradients underflow in fp16, so the scaler multiplies the loss by a large factor
   before backward, unscales before clipping and stepping, and skips steps that
   produced inf/nan, lowering the scale.

4. **Why warmup?**
   Adam's first steps move every weight by roughly the learning rate regardless of
   gradient size, and the head starts random. A gentle start avoids large, noisy
   updates destroying pretrained features.

5. **Why clip gradients?**
   A single outlier batch can produce a huge gradient. Capping the global norm
   bounds the step size without changing its direction.

6. **Why macro-F1 rather than accuracy?**
   With 185× class imbalance, accuracy rewards ignoring rare emotions. Macro-F1
   weights grief equally with neutral, and F1 ignores the easy true negatives.

7. **Why tune thresholds per class, and why on dev?**
   A 0.2%-positive label rarely gets p > 0.5, so at 0.5 it's never predicted. On my
   baseline, tuning took test macro-F1 from 0.287 to 0.471. It's done on dev because
   tuning on test would leak test information into the reported number.

8. **What does a 95% bootstrap CI on macro-F1 mean here?**
   If the test set were redrawn from the same distribution, the macro-F1 would vary.
   Resampling the test set 1,000 times estimates that variation. It covers test-set
   sampling noise, not training randomness (that's what multiple seeds are for).

9. **How do you test whether model A beats model B?**
   Paired methods on the same test examples: a paired bootstrap CI on the difference
   and an approximate randomization test (swap the two models' predictions per
   example at random). Unpaired comparisons waste power, because example difficulty
   is shared.

10. **What is calibration and how did you measure it?**
    Whether p = 0.7 means "on 70% of the time". Measured with ECE over 15 bins of
    all N × 28 probabilities, plus reliability diagrams.

**Design decisions**

11. **Why write the training loop instead of using HF Trainer?**
    To understand and control every step (AMP, accumulation, clipping, scheduling,
    selection), and to train the BiLSTM and the transformer with identical code,
    which makes the comparison fair.

12. **Why RoBERTa instead of DeBERTa-v3?**
    DeBERTa-v3 has known fp16 overflow problems, and the free Kaggle T4 needs fp16
    for speed. RoBERTa-base is stable under fp16 and a standard strong baseline.

13. **Why have a TF-IDF baseline at all?**
    Without it you can't tell how much of the transformer's score comes from
    pretraining versus from just having a classifier. My tuned TF-IDF baseline
    reached 0.471, so the bar is meaningful.

14. **Explain LoRA's parameter count.**
    A 768 × 768 matrix has 589,824 weights. Rank-8 LoRA learns A (8 × 768) and B
    (768 × 8): 12,288, or 2.1%. Over 24 matrices plus the head that's 316,444
    trainable parameters, versus about 124M for full fine-tuning. B starts at 0, so
    training begins at the pretrained model.

15. **Why does LoRA need a higher learning rate?**
    Few parameters, B initialised at 0, and the update scaled by α/r. Small LRs barely
    move the model. 5e-4 versus 3e-5 for full fine-tuning is typical.

16. **How did you handle class imbalance?**
    Per-class thresholds (biggest effect), plus ablations: BCE pos_weight (#neg/#pos,
    capped at 30) and focal loss (γ = 2). Results: TBD.

17. **How did you make it reproducible?**
    Seeds for every RNG, cuDNN deterministic mode, YAML configs, and every run saving
    its resolved config, git commit, library versions, GPU name, metrics, curve and
    predictions. I also note that GPU kernels aren't bit-exact even with seeds, so I
    report mean ± std over 3 seeds.

18. **How do you know your training loop is correct?**
    Unit tests: it overfits 8 examples to near-zero loss; accumulation matches a big
    batch numerically; the scheduler produces the expected multipliers; and save →
    reload reproduces the test probabilities exactly.

**Psychology findings**

19. **Define your multi-label confusion matrix.**
    C[i, j] = among examples where i is gold and j isn't, the fraction where the
    model missed i and predicted j. Conditioning on j not being gold avoids counting
    a correct second label as a confusion.

20. **Why a permutation test instead of the p-value from `spearmanr`?**
    The 351 pairs aren't independent: each emotion appears in 26 of them. Shuffling
    which emotion owns which coordinates (a Mantel test) keeps that dependence
    structure in the null distribution.

21. **What did humans confuse most?** (real)
    grief→sadness, remorse→sadness, confusion↔curiosity, anger↔annoyance. Those are
    all pairs close in valence and arousal, which motivated the RQ2 hypothesis test.
    The test itself: TBD.

22. **Isn't the RQ3 probe circular?**
    Partly, for the fine-tuned model, because the target is computed from labels it
    was trained on. That's why I compare with a random-init encoder, the pretrained
    encoder, and a probe on the model's own 28 outputs. The defensible claims are
    "pretraining alone already encodes X" and "the embedding holds more or less VA
    than the labels".

23. **What did RQ5 show?** TBD. (If emotion tuning doesn't help, say so and suggest
    why: tiny test set, different label semantics, or the vanilla model already
    encodes enough affect.)

**What would you do differently?**

24. **Biggest weaknesses?**
    Noisy crowd labels cap achievable F1. The VA coordinates describe emotion
    *words*, not texts. Human confusion uses raters who also formed the consensus.
    RQ5's test set is about 105 examples. With more time: leave-one-rater-out
    consensus, a VA-annotated text dataset (e.g. EmoBank) as a direct probe target,
    and more seeds for RQ5.

25. **What would you try next for accuracy?**
    Larger encoders (roberta-large with LoRA fits a T4), longer training with
    layer-wise LR decay, label-correlation-aware heads (classifier chains), and
    training on soft labels from rater votes instead of the ≥2-agreement binary
    labels. Soft labels would also give better-calibrated probabilities.
