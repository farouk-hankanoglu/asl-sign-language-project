"""
eval_multiseed_nn.py
====================

Word-level accuracy of the confidence-aware correction layer, using the
NEURAL NETWORK classifier -- the same model that is exported to the web app.

WHY THIS SCRIPT EXISTS
----------------------
evaluate_word_level_accuracy_multiseed.py measured word-level accuracy using
the Random Forest. The model actually deployed in the application, and the one
whose on-device cost is reported in the evaluation chapter, is the neural
network. Reporting word-level results from one model and deployment results
from another is an inconsistency an examiner will find. This script closes it
by running the identical protocol with the network.

PROTOCOL (unchanged from the Random Forest version)
---------------------------------------------------
1. Train the classifier on the TRAINING datasets only.
2. Split the HELD-OUT dataset into two disjoint halves, stratified by letter:
      Half A -- used ONLY to estimate the letter confusion matrix
      Half B -- used ONLY to evaluate word-level accuracy
   The halves never overlap. This is what prevents the circularity of
   estimating a channel model on the same images used to score it.
3. Sample words uniformly from the dictionary.
4. For each letter of each word, draw one unseen image from Half B, take the
   network's predicted letter and its softmax confidence.
5. Apply the noisy-channel correction:
      cost(p -> c) = (1 - f[c][p]) * conf(p)
   where f[c][p] is P(predicted p | true c) estimated on Half A. Accept the
   best dictionary candidate if its mean cost per letter is below THRESHOLD.
6. Repeat over N seeds and report mean +/- standard deviation.

TWO VARIABILITY MODES
---------------------
  fixed    - one classifier, trained once; seeds vary only word sampling and
             image selection. This is what the earlier results measured, and
             it is why the error bars are small: they do NOT include
             classifier variability.
  retrain  - the classifier is retrained with a different initialisation on
             every seed, so the error bars include classifier variability too.

The script runs BOTH and prints both, so the dissertation can state exactly
what its +/- figures represent. Supervisor feedback specifically asked for
this to be made explicit.

USAGE
-----
    python eval_multiseed_nn.py

If it cannot find your CSV files it will say so and list what it did find.
Edit the CONFIG block below if your filenames differ.
"""

import os
import sys
import glob
import json
import random
import argparse

import numpy as np

# ----------------------------------------------------------------------------
# CONFIG -- edit these if your filenames differ
# ----------------------------------------------------------------------------

# CSVs used to TRAIN the classifier. Globs are allowed.
# Every dataset available. --holdout picks which one is held out; the rest
# become the training set. This makes leave-one-dataset-out possible without
# editing anything.
ALL_DATASETS = {
    "kaggle":       "landmarks_train.csv",
    "aslhg":        "landmarks_aslhg.csv",
    "signalphaset": "landmarks_signalphaset.csv",
}

TRAIN_CSVS = [
    "landmarks_train.csv",     # Kaggle ASL Alphabet
    "landmarks_aslhg.csv",     # ASL-HG
]

# The HELD-OUT CSV. Never used for training; split into Half A / Half B.
HELDOUT_CSV = "landmarks_signalphaset.csv"

DICTIONARY_FILE = "words10k.txt"

N_SEEDS = 10
N_WORDS = 300           # words sampled per seed
MIN_WORD_LEN = 3
MAX_WORD_LEN = 8
THRESHOLD = 0.5         # max mean cost per letter to accept a correction
EPOCHS = 60
RF_TREES = 200          # only used by --model rf / both
BATCH_SIZE = 32
OUT_DIR = "outputs"

LETTERS = [chr(ord("A") + i) for i in range(26)]
L2I = {c: i for i, c in enumerate(LETTERS)}


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------

# Where to look for data files. The folder this script lives in is searched
# as well as the current working directory, so that running
#     python C:\...\asl_scripts\eval_multiseed_nn.py
# from somewhere else still finds the CSVs sitting next to the script.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SEARCH_DIRS = []
for _d in (os.getcwd(), SCRIPT_DIR):
    if _d not in SEARCH_DIRS:
        SEARCH_DIRS.append(_d)


def _resolve(patterns):
    """
    Expand globs against every search directory.

    Returns (found, missing). A pattern that matches nothing is reported in
    `missing` rather than silently ignored: training on fewer datasets than
    intended changes the result without changing anything visible, which is
    how a wrong number ends up in a write-up.
    """
    found, seen, missing = [], set(), []
    for p in patterns:
        if os.path.isabs(p):
            candidates = sorted(glob.glob(p))
        else:
            candidates = []
            for d in SEARCH_DIRS:
                candidates.extend(sorted(glob.glob(os.path.join(d, p))))
        hit = False
        for m in candidates:
            real = os.path.realpath(m)
            if os.path.isfile(m):
                hit = True
                if real not in seen:
                    seen.add(real)
                    found.append(m)
        if not hit:
            missing.append(p)
    return found, missing


def _find_one(name):
    """Locate a single data file in any search directory. None if absent."""
    hits, _ = _resolve([name])
    return hits[0] if hits else None


def _die_with_listing(message):
    print("\nERROR: " + message, file=sys.stderr)
    for d in SEARCH_DIRS:
        here = sorted(os.path.basename(f) for f in glob.glob(os.path.join(d, "*.csv")))
        print("\nCSV files in " + d + ":", file=sys.stderr)
        if here:
            for f in here:
                print("   " + f, file=sys.stderr)
        else:
            print("   (none)", file=sys.stderr)
    print("\nEdit the CONFIG block at the top of this script so TRAIN_CSVS and "
          "HELDOUT_CSV\nmatch the names listed above, then run it again.",
          file=sys.stderr)
    sys.exit(1)


def load_csv(path):
    """Read a landmarks CSV: label,x0,y0,z0,...,x20,y20,z20 -> (X, y)."""
    import csv
    X, y = [], []
    bad = 0
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        if len(header) != 64 or header[0].lower() != "label":
            raise ValueError(
                f"{path}: expected 64 columns starting with 'label', "
                f"got {len(header)} starting with '{header[0]}'"
            )
        for row in reader:
            if len(row) != 64:
                bad += 1
                continue
            lab = row[0].strip().upper()
            if lab not in L2I:
                bad += 1
                continue
            try:
                vals = [float(v) for v in row[1:]]
            except ValueError:
                bad += 1
                continue
            X.append(vals)
            y.append(L2I[lab])
    if not X:
        raise ValueError(f"{path}: no usable rows")
    if bad:
        print(f"    (skipped {bad} unusable rows in {os.path.basename(path)})")
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64)


def load_dictionary(name):
    path = _find_one(name)
    if path is None:
        print(f"\nERROR: dictionary '{name}' not found in any of:", file=sys.stderr)
        for d in SEARCH_DIRS:
            print("   " + d, file=sys.stderr)
        sys.exit(1)
    words = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            w = line.strip().upper()
            if w.isalpha() and all(c in L2I for c in w):
                words.append(w)
    # dedupe, keep order
    seen, out = set(), []
    for w in words:
        if w not in seen:
            seen.add(w)
            out.append(w)
    return out


# ----------------------------------------------------------------------------
# Model -- identical architecture to the one exported to the web app
# ----------------------------------------------------------------------------

def build_model(seed):
    import tensorflow as tf
    from tensorflow import keras

    tf.keras.utils.set_random_seed(seed)
    model = keras.Sequential([
        keras.layers.Input(shape=(63,)),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(26, activation="softmax"),
    ])
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


class _RFWrapper:
    """
    Gives a scikit-learn RandomForest the same .predict(X) -> probabilities
    interface the rest of this script expects from Keras, so that BOTH
    classifiers go through byte-identical evaluation code. Without this the
    forest and the network would be compared across two different scripts,
    and any difference could be the code rather than the model.
    """

    def __init__(self, rf):
        self.rf = rf

    def predict(self, X, verbose=0):
        proba = self.rf.predict_proba(X)
        # map the forest's class list onto the full 26-letter probability row
        full = np.zeros((len(X), 26), dtype=np.float64)
        for col, cls in enumerate(self.rf.classes_):
            full[:, int(cls)] = proba[:, col]
        return full


def train_model(Xtr, ytr, seed, kind="nn", verbose=0):
    if kind == "nn":
        model = build_model(seed)
        model.fit(Xtr, ytr, epochs=EPOCHS, batch_size=BATCH_SIZE,
                  verbose=verbose, shuffle=True)
        return model
    if kind == "rf":
        from sklearn.ensemble import RandomForestClassifier
        rf = RandomForestClassifier(n_estimators=RF_TREES, random_state=seed,
                                    n_jobs=-1)
        rf.fit(Xtr, ytr)
        return _RFWrapper(rf)
    raise ValueError("unknown model kind: " + kind)


# ----------------------------------------------------------------------------
# Circularity-safe split
# ----------------------------------------------------------------------------

def stratified_halves(y, rng):
    """Split indices into two disjoint halves, balanced per letter."""
    a, b = [], []
    for c in range(26):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        mid = len(idx) // 2
        a.extend(idx[:mid].tolist())
        b.extend(idx[mid:].tolist())
    return np.asarray(a), np.asarray(b)


def confusion_from(pred, true):
    """f[c][p] = P(predicted p | true c), rows sum to 1."""
    f = np.zeros((26, 26), dtype=np.float64)
    for p, t in zip(pred, true):
        f[t][p] += 1
    rows = f.sum(axis=1, keepdims=True)
    # letters absent from Half A get a uniform row rather than NaN
    f = np.where(rows > 0, f / np.maximum(rows, 1e-12), 1.0 / 26.0)
    return f


# ----------------------------------------------------------------------------
# Noisy-channel correction
# ----------------------------------------------------------------------------

def substitution_cost(f, pred_idx, cand_idx, conf):
    """cost(p -> c) = (1 - f[c][p]) * conf(p)"""
    return (1.0 - f[cand_idx][pred_idx]) * conf


def correct_word(observed, confs, f, dictionary_by_len, threshold):
    """
    Cheapest same-length dictionary word under the channel model.
    Returns (corrected_or_None, mean_cost).
    """
    n = len(observed)
    best_word, best_cost = None, float("inf")
    for cand in dictionary_by_len.get(n, ()):
        total = 0.0
        for i in range(n):
            p = observed[i]
            c = L2I[cand[i]]
            total += 0.0 if p == c else substitution_cost(f, p, c, confs[i])
            if total >= best_cost:
                break
        if total < best_cost:
            best_cost, best_word = total, cand
    if best_word is None:
        return None, float("inf")
    mean_cost = best_cost / n
    return (best_word if mean_cost <= threshold else None), mean_cost


# ----------------------------------------------------------------------------
# One evaluation pass
# ----------------------------------------------------------------------------

def run_seed(seed, model, Xh, yh, dictionary_by_len, words_pool, n_words):
    rng = np.random.default_rng(seed)
    pyrng = random.Random(seed)

    # --- circularity-safe split of the held-out set -------------------------
    idx_a, idx_b = stratified_halves(yh, rng)

    # Half A -> confusion matrix
    prob_a = model.predict(Xh[idx_a], verbose=0)
    pred_a = prob_a.argmax(axis=1)
    f = confusion_from(pred_a, yh[idx_a])

    # Half B -> word evaluation pool, indexed by true letter
    prob_b = model.predict(Xh[idx_b], verbose=0)
    pred_b = prob_b.argmax(axis=1)
    conf_b = prob_b.max(axis=1)
    true_b = yh[idx_b]

    by_letter = {c: np.where(true_b == c)[0] for c in range(26)}
    usable = {c for c, v in by_letter.items() if len(v) > 0}

    # only sample words whose letters all have held-out images available
    pool = [w for w in words_pool if all(L2I[ch] in usable for ch in w)]
    if len(pool) < 50:
        raise RuntimeError(
            f"only {len(pool)} words can be built from the held-out letters "
            f"available ({len(usable)}/26 letters present in Half B)"
        )
    words = [pyrng.choice(pool) for _ in range(n_words)]

    raw_correct = 0
    corrected_correct = 0
    letter_correct = 0
    letter_total = 0
    attempted = 0

    for w in words:
        observed, confs = [], []
        for ch in w:
            j = int(pyrng.choice(by_letter[L2I[ch]]))
            observed.append(int(pred_b[j]))
            confs.append(float(conf_b[j]))
            letter_total += 1
            if int(pred_b[j]) == L2I[ch]:
                letter_correct += 1

        raw = "".join(LETTERS[p] for p in observed)
        if raw == w:
            raw_correct += 1

        out, _ = correct_word(observed, confs, f, dictionary_by_len, THRESHOLD)
        if out is not None:
            attempted += 1
        final = out if out is not None else raw
        if final == w:
            corrected_correct += 1

    return dict(
        raw_word_acc=100.0 * raw_correct / len(words),
        corrected_word_acc=100.0 * corrected_correct / len(words),
        letter_acc=100.0 * letter_correct / max(letter_total, 1),
        correction_rate=100.0 * attempted / len(words),
        n_half_a=len(idx_a),
        n_half_b=len(idx_b),
    )


# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------

def summarise(name, runs):
    keys = ["letter_acc", "raw_word_acc", "corrected_word_acc", "correction_rate"]
    out = {}
    for k in keys:
        vals = np.array([r[k] for r in runs], dtype=np.float64)
        out[k] = (vals.mean(), vals.std(ddof=1) if len(vals) > 1 else 0.0)
    print(f"\n  {name}")
    print("  " + "-" * 62)
    print(f"  {'metric':<34}{'mean':>10}{'std dev':>10}")
    print(f"  {'letter accuracy (held-out)':<34}{out['letter_acc'][0]:>9.2f}%{out['letter_acc'][1]:>9.2f}")
    print(f"  {'word accuracy, RAW':<34}{out['raw_word_acc'][0]:>9.2f}%{out['raw_word_acc'][1]:>9.2f}")
    print(f"  {'word accuracy, CORRECTED':<34}{out['corrected_word_acc'][0]:>9.2f}%{out['corrected_word_acc'][1]:>9.2f}")
    print(f"  {'correction applied to':<34}{out['correction_rate'][0]:>9.2f}%{out['correction_rate'][1]:>9.2f}")
    gain = out['corrected_word_acc'][0] - out['raw_word_acc'][0]
    print(f"  {'absolute gain from correction':<34}{gain:>9.2f}pp")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=N_SEEDS)
    ap.add_argument("--words", type=int, default=N_WORDS)
    ap.add_argument("--allow-missing", action="store_true",
                    help="proceed even if a TRAIN_CSVS entry matches no file")
    ap.add_argument("--mode", choices=["both", "fixed", "retrain"], default="both")
    ap.add_argument("--holdout", choices=sorted(ALL_DATASETS), default=None,
                    help="which dataset to hold out; the other two are used "
                         "for training. Default: signalphaset (the configured "
                         "HELDOUT_CSV). Use this for leave-one-dataset-out.")
    ap.add_argument("--mirror", action="store_true",
                    help="add mirrored copies of every training sample "
                         "(x coordinates negated), doubling the training set")
    ap.add_argument("--model", choices=["nn", "rf", "both"], default="nn",
                    help="nn = neural network (deployed); rf = random forest; "
                         "both = run each through identical evaluation code")
    args = ap.parse_args()

    print("=" * 66)
    print("  Word-level evaluation of the correction layer")
    print("=" * 66)
    print("  working directory:", os.getcwd())
    print("  script directory: ", SCRIPT_DIR)
    if len(SEARCH_DIRS) > 1:
        print("  (data files are looked for in both)")

    train_csvs, heldout_csv = TRAIN_CSVS, HELDOUT_CSV
    if args.holdout:
        heldout_csv = ALL_DATASETS[args.holdout]
        train_csvs = [v for k, v in sorted(ALL_DATASETS.items())
                      if k != args.holdout]
        print(f"\n  leave-one-out: holding out {args.holdout}")

    train_paths, train_missing = _resolve(train_csvs)
    if not train_paths:
        _die_with_listing("none of the training patterns matched a file: "
                          + ", ".join(train_csvs))
    if train_missing and not args.allow_missing:
        _die_with_listing(
            "these TRAIN_CSVS entries matched no file: "
            + ", ".join(train_missing)
            + "\n\nTraining on fewer datasets than intended silently changes every"
              "\nnumber this script prints. Fix the names, or re-run with"
              "\n--allow-missing if you really do mean to train on "
            + str(len(train_paths)) + " dataset(s) only.")
    if train_missing:
        print("\n  !! WARNING: training on " + str(len(train_paths)) +
              " dataset(s); no file matched: " + ", ".join(train_missing))
        print("  !! The results below are NOT the documented protocol.")
    heldout_path = _find_one(heldout_csv)
    if heldout_path is None:
        _die_with_listing(f"held-out CSV '{heldout_csv}' not found")

    print("\n  training on:")
    Xs, ys = [], []
    for p in train_paths:
        X, y = load_csv(p)
        print(f"    {os.path.basename(p):<40} {len(X):>7} rows")
        Xs.append(X)
        ys.append(y)
    Xtr = np.concatenate(Xs)
    ytr = np.concatenate(ys)

    if args.mirror:
        # A left-handed example is a right-handed one with x negated. The
        # landmark layout is [x0,y0,z0, x1,y1,z1, ...], so every third value
        # starting at 0 is an x coordinate.
        Xm = Xtr.copy()
        Xm[:, 0::3] *= -1.0
        Xtr = np.concatenate([Xtr, Xm])
        ytr = np.concatenate([ytr, ytr])
        print(f"\n  mirror augmentation: training set doubled to {len(Xtr)} rows")

    print("\n  held out (never trained on):")
    Xh, yh = load_csv(heldout_path)
    print(f"    {os.path.basename(heldout_path):<40} {len(Xh):>7} rows")

    missing = [LETTERS[c] for c in range(26) if not np.any(yh == c)]
    if missing:
        print(f"    NOTE: held-out set has no examples of: {', '.join(missing)}")

    words_pool = load_dictionary(DICTIONARY_FILE)
    words_pool = [w for w in words_pool if MIN_WORD_LEN <= len(w) <= MAX_WORD_LEN]
    by_len = {}
    for w in words_pool:
        by_len.setdefault(len(w), []).append(w)
    print(f"\n  dictionary: {len(words_pool)} words of length "
          f"{MIN_WORD_LEN}-{MAX_WORD_LEN} from {DICTIONARY_FILE}")
    print(f"  protocol:   {args.seeds} seeds x {args.words} words, "
          f"threshold {THRESHOLD} mean cost/letter")

    results = {}
    kinds = ["nn", "rf"] if args.model == "both" else [args.model]

    for kind in kinds:
      label = "NEURAL NETWORK" if kind == "nn" else "RANDOM FOREST"
      print("\n" + "=" * 66)
      print("  CLASSIFIER: " + label)
      print("=" * 66)

      if args.mode in ("both", "fixed"):
        print("\n" + "-" * 66)
        print("  MODE 1: FIXED classifier (trained once, seed 0)")
        print("  Seeds vary word sampling and image selection ONLY.")
        print("  The +/- below therefore does NOT include classifier variability.")
        print("-" * 66)
        print("  training once...", flush=True)
        model = train_model(Xtr, ytr, seed=0, kind=kind)
        runs = []
        for sd in range(args.seeds):
            r = run_seed(sd, model, Xh, yh, by_len, words_pool, args.words)
            runs.append(r)
            print(f"    seed {sd:<2} raw {r['raw_word_acc']:6.2f}%   "
                  f"corrected {r['corrected_word_acc']:6.2f}%", flush=True)
        print(f"\n    Half A (confusion): {runs[0]['n_half_a']} images   "
              f"Half B (evaluation): {runs[0]['n_half_b']} images   [disjoint]")
        results[kind + "_fixed"] = summarise(label + " / FIXED", runs)

      if args.mode in ("both", "retrain"):
        print("\n" + "-" * 66)
        print("  MODE 2: RETRAINED per seed")
        print("  Each seed retrains the classifier from a new initialisation,")
        print("  so the +/- DOES include classifier variability. This is the")
        print("  figure to quote if you want an honest error bar.")
        print("-" * 66)
        runs = []
        for sd in range(args.seeds):
            print(f"    seed {sd:<2} training...", end=" ", flush=True)
            model = train_model(Xtr, ytr, seed=sd, kind=kind)
            r = run_seed(sd, model, Xh, yh, by_len, words_pool, args.words)
            runs.append(r)
            print(f"raw {r['raw_word_acc']:6.2f}%   "
                  f"corrected {r['corrected_word_acc']:6.2f}%", flush=True)
        results[kind + "_retrain"] = summarise(label + " / RETRAINED", runs)

    out_dir = os.path.join(SCRIPT_DIR, OUT_DIR)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, ("eval_multiseed_%s_results.json" % args.model))
    with open(out_path, "w") as fh:
        json.dump({k: {m: list(v) for m, v in d.items()}
                   for k, d in results.items()}, fh, indent=2)

    print("\n" + "=" * 66)
    print("  Saved to " + out_path)
    print("  Send the whole output above to Claude to update Table 5.2.")
    print("=" * 66)


if __name__ == "__main__":
    main()
