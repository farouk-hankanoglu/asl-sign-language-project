"""
evaluate_word_level_accuracy_multiseed.py

Douglas's request: repeat the 300-word sampling with different random
seeds and report the mean and spread, rather than a single run - this
shows the 89.7% figure is stable, not a lucky draw.

Same clean split as before (Half A for confusion matrix, Half B for
word-level testing, fully independent - the circularity fix), but now
repeated across N different random seeds for the word-sampling step,
reporting mean, standard deviation, and a rough 95% confidence
interval.

Usage:
    python evaluate_word_level_accuracy_multiseed.py --kaggle landmarks_train.csv --aslhg landmarks_aslhg.csv --third landmarks_signalphaset.csv --dictionary_file words10k.txt --n_seeds 10
"""

import argparse
import random
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

LETTERS = sorted("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

FALLBACK_DICTIONARY = [
    "HELLO","WORLD","GOOD","MORNING","NIGHT","THANKS","PLEASE","SORRY",
    "HELP","WATER","FOOD","HOUSE","SCHOOL","FRIEND","FAMILY","LOVE",
]


def load_dictionary_file(path):
    with open(path) as f:
        words = [line.strip().upper() for line in f if line.strip()]
    return [w for w in words if w.isalpha() and 2 <= len(w) <= 9]


def load_and_filter(path, mirror=False):
    df = pd.read_csv(path)
    df = df[df["label"].isin(LETTERS)].copy()
    feature_cols = [c for c in df.columns if c != "label"]
    X = df[feature_cols].values.astype(np.float32)
    y = df["label"].values
    if mirror:
        X_mirrored = X.copy()
        X_mirrored[:, 0::3] *= -1
        X = np.concatenate([X, X_mirrored], axis=0)
        y = np.concatenate([y, y], axis=0)
    return X, y, feature_cols


def build_confusion_cost_table(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    cost_table = {}
    for i, true_letter in enumerate(classes):
        row_total = cm[i].sum()
        if row_total == 0:
            continue
        for j, pred_letter in enumerate(classes):
            if i == j:
                continue
            freq = cm[i][j] / row_total
            if freq > 0:
                cost_table[(pred_letter, true_letter)] = max(0.05, 1.0 - freq)
    return cost_table


def substitution_cost(predicted_letter, confidence, candidate_letter, cost_table):
    if predicted_letter == candidate_letter:
        return 0.0
    base = cost_table.get((predicted_letter, candidate_letter), 1.0)
    return base * confidence


def weighted_edit_distance(pred_letters, confidences, dict_word, cost_table):
    n, m = len(pred_letters), len(dict_word)
    INSERT_DELETE_COST = 0.9
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i * INSERT_DELETE_COST
    for j in range(m + 1):
        dp[0][j] = j * INSERT_DELETE_COST
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub_cost = substitution_cost(pred_letters[i-1], confidences[i-1], dict_word[j-1], cost_table)
            dp[i][j] = min(
                dp[i-1][j] + INSERT_DELETE_COST,
                dp[i][j-1] + INSERT_DELETE_COST,
                dp[i-1][j-1] + sub_cost,
            )
    return dp[n][m]


def autocorrect_word(pred_letters, confidences, cost_table, dictionary, max_cost_per_letter=0.5):
    raw_word = "".join(pred_letters)
    scored = [(w, weighted_edit_distance(pred_letters, confidences, w, cost_table)) for w in dictionary]
    scored.sort(key=lambda x: x[1])
    best_word, cost = scored[0]
    avg_cost = cost / max(1, len(best_word))
    if avg_cost > max_cost_per_letter:
        return raw_word
    return best_word


def run_one_seed(seed, dictionary, by_letter_indices_B, preds_B_letters, confidences_B,
                  y_B_raw, cost_table, n_simulated_words):
    rng = random.Random(seed)
    available_letters = {l for l, idxs in by_letter_indices_B.items() if len(idxs) > 0}
    words_pool = [w for w in dictionary if all(c in available_letters for c in w)]
    rng.shuffle(words_pool)
    if n_simulated_words and len(words_pool) < n_simulated_words:
        words_to_test = (words_pool * ((n_simulated_words // max(1, len(words_pool))) + 1))[:n_simulated_words]
    else:
        words_to_test = words_pool[:n_simulated_words] if n_simulated_words else words_pool

    raw_correct = 0
    corrected_correct = 0
    for word in words_to_test:
        pred_letters, conf_list = [], []
        for ch in word:
            idx_choice = rng.choice(list(by_letter_indices_B[ch]))
            pred_letters.append(preds_B_letters[idx_choice])
            conf_list.append(confidences_B[idx_choice])
        raw_word = "".join(pred_letters)
        if raw_word == word:
            raw_correct += 1
        corrected_word = autocorrect_word(pred_letters, conf_list, cost_table, dictionary)
        if corrected_word == word:
            corrected_correct += 1

    n = len(words_to_test)
    return raw_correct / n, corrected_correct / n


def main(kaggle_path, aslhg_path, third_path, n_simulated_words, dictionary_file, n_seeds):
    print("=" * 70)
    print("Training classifier on Kaggle + ASL-HG (combined, mirrored)...")
    X_kaggle, y_kaggle, feature_cols = load_and_filter(kaggle_path, mirror=True)
    X_aslhg, y_aslhg, _ = load_and_filter(aslhg_path, mirror=True)
    X_train = np.concatenate([X_kaggle, X_aslhg], axis=0)
    y_train_raw = np.concatenate([y_kaggle, y_aslhg], axis=0)

    le = LabelEncoder()
    le.fit(LETTERS)
    y_train = le.transform(y_train_raw)

    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    print(f"Trained on {len(X_train)} samples.")

    print("\nSplitting SignAlphaSet into two INDEPENDENT halves (fixed once, not re-split per seed)...")
    df_third = pd.read_csv(third_path)
    df_third = df_third[df_third["label"].isin(LETTERS)].copy()
    X_third_all = df_third[feature_cols].values.astype(np.float32)
    y_third_all_raw = df_third["label"].values

    idx = np.arange(len(df_third))
    idx_A, idx_B = train_test_split(idx, test_size=0.5, stratify=y_third_all_raw, random_state=42)
    print(f"Half A: {len(idx_A)} | Half B: {len(idx_B)} (this split is FIXED across all seeds below - "
          f"only the WORD SAMPLING varies per seed, not the train/test split itself)")

    X_A, y_A_raw = X_third_all[idx_A], y_third_all_raw[idx_A]
    probs_A = rf.predict_proba(X_A)
    preds_A_letters = le.inverse_transform(np.argmax(probs_A, axis=1))
    cost_table = build_confusion_cost_table(y_A_raw, preds_A_letters, LETTERS)

    X_B, y_B_raw = X_third_all[idx_B], y_third_all_raw[idx_B]
    probs_B = rf.predict_proba(X_B)
    preds_B_idx = np.argmax(probs_B, axis=1)
    confidences_B = np.max(probs_B, axis=1)
    preds_B_letters = le.inverse_transform(preds_B_idx)
    by_letter_indices_B = {letter: np.where(y_B_raw == letter)[0] for letter in LETTERS}

    letter_acc_B = accuracy_score(le.transform(y_B_raw), preds_B_idx)
    print(f"Letter-level accuracy on Half B (fixed, independent): {letter_acc_B:.4f}")

    dictionary = load_dictionary_file(dictionary_file) if dictionary_file else FALLBACK_DICTIONARY
    print(f"Dictionary size: {len(dictionary)} words")

    print(f"\n{'='*70}\nRunning {n_seeds} independent word-sampling seeds "
          f"({n_simulated_words} words each)...\n{'='*70}")

    raw_accuracies = []
    corrected_accuracies = []
    for seed in range(n_seeds):
        raw_acc, corrected_acc = run_one_seed(
            seed, dictionary, by_letter_indices_B, preds_B_letters, confidences_B,
            y_B_raw, cost_table, n_simulated_words
        )
        raw_accuracies.append(raw_acc)
        corrected_accuracies.append(corrected_acc)
        print(f"  Seed {seed}: raw={raw_acc:.4f}  corrected={corrected_acc:.4f}")

    raw_arr = np.array(raw_accuracies)
    corr_arr = np.array(corrected_accuracies)

    def summarize(arr, label):
        mean = arr.mean()
        std = arr.std(ddof=1) if len(arr) > 1 else 0.0
        ci95 = 1.96 * std / np.sqrt(len(arr)) if len(arr) > 1 else 0.0
        print(f"  {label}: mean={mean:.4f}  std={std:.4f}  "
              f"range=[{arr.min():.4f}, {arr.max():.4f}]  approx 95% CI of mean=+/-{ci95:.4f}")
        return mean, std

    print(f"\n{'='*70}\nSUMMARY ACROSS {n_seeds} SEEDS\n{'='*70}")
    summarize(raw_arr, "Word accuracy WITHOUT correction")
    summarize(corr_arr, "Word accuracy WITH correction   ")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kaggle", required=True)
    parser.add_argument("--aslhg", required=True)
    parser.add_argument("--third", required=True)
    parser.add_argument("--n_words", type=int, default=300)
    parser.add_argument("--dictionary_file", default=None)
    parser.add_argument("--n_seeds", type=int, default=10)
    args = parser.parse_args()
    main(args.kaggle, args.aslhg, args.third, args.n_words, args.dictionary_file, args.n_seeds)
