

import argparse
import random
from collections import Counter
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
    words = [w for w in words if w.isalpha() and 2 <= len(w) <= 9]
    return words


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
    return cost_table, cm


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


def main(kaggle_path, aslhg_path, third_path, n_simulated_words, dictionary_file):
    print("=" * 70)
    print("STEP 1: Train classifier on Kaggle + ASL-HG (combined, mirrored)")
    print("=" * 70)
    X_kaggle, y_kaggle, feature_cols = load_and_filter(kaggle_path, mirror=True)
    X_aslhg, y_aslhg, _ = load_and_filter(aslhg_path, mirror=True)
    X_train = np.concatenate([X_kaggle, X_aslhg], axis=0)
    y_train_raw = np.concatenate([y_kaggle, y_aslhg], axis=0)

    le = LabelEncoder()
    le.fit(LETTERS)
    y_train = le.transform(y_train_raw)

    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    print(f"Trained on {len(X_train)} samples (Kaggle+ASL-HG, mirror-augmented).")

    print("\n" + "=" * 70)
    print("STEP 2: Split SignAlphaSet into two INDEPENDENT halves")
    print("=" * 70)
    df_third = pd.read_csv(third_path)
    df_third = df_third[df_third["label"].isin(LETTERS)].copy()
    X_third_all = df_third[feature_cols].values.astype(np.float32)
    y_third_all_raw = df_third["label"].values

    idx = np.arange(len(df_third))
    idx_A, idx_B = train_test_split(
        idx, test_size=0.5, stratify=y_third_all_raw, random_state=42
    )
    print(f"Half A (confusion-matrix estimation): {len(idx_A)} samples")
    print(f"Half B (word-level correction test):  {len(idx_B)} samples")
    print("These are DISJOINT - no sample is used for both purposes.")

    # ---------- Half A: measure confusion matrix ONLY ----------
    X_A, y_A_raw = X_third_all[idx_A], y_third_all_raw[idx_A]
    probs_A = rf.predict_proba(X_A)
    preds_A_idx = np.argmax(probs_A, axis=1)
    preds_A_letters = le.inverse_transform(preds_A_idx)

    letter_acc_A = accuracy_score(le.transform(y_A_raw), preds_A_idx)
    print(f"\nLetter-level accuracy on Half A: {letter_acc_A:.4f}")

    cost_table, cm = build_confusion_cost_table(y_A_raw, preds_A_letters, LETTERS)
    top_confusions = sorted(cost_table.items(), key=lambda x: x[1])[:8]
    print("Top confusion pairs found on Half A (predicted -> true, cost):")
    for (pred_l, true_l), cost in top_confusions:
        print(f"  predicted '{pred_l}' actually '{true_l}': cost {cost:.2f}")

    # ---------- Half B: word-level correction test ONLY ----------
    X_B, y_B_raw = X_third_all[idx_B], y_third_all_raw[idx_B]
    probs_B = rf.predict_proba(X_B)
    preds_B_idx = np.argmax(probs_B, axis=1)
    confidences_B = np.max(probs_B, axis=1)
    preds_B_letters = le.inverse_transform(preds_B_idx)

    letter_acc_B = accuracy_score(le.transform(y_B_raw), preds_B_idx)
    print(f"\nLetter-level accuracy on Half B (independent test set): {letter_acc_B:.4f}")

    print("\n" + "=" * 70)
    print("STEP 3: Build word test set and run correction (using Half B only)")
    print("=" * 70)
    dictionary = load_dictionary_file(dictionary_file) if dictionary_file else FALLBACK_DICTIONARY
    print(f"Dictionary size: {len(dictionary)} words")

    by_letter_indices_B = {letter: np.where(y_B_raw == letter)[0] for letter in LETTERS}
    available_letters = {l for l, idxs in by_letter_indices_B.items() if len(idxs) > 0}
    print(f"Letters available in Half B: {sorted(available_letters)} "
          f"({len(available_letters)}/26)")

    words_to_test_pool = [w for w in dictionary if all(c in available_letters for c in w)]
    print(f"Dictionary words fully constructible from available letters: {len(words_to_test_pool)}")

    random.seed(42)
    random.shuffle(words_to_test_pool)
    if n_simulated_words and len(words_to_test_pool) < n_simulated_words:
        words_to_test = (words_to_test_pool * ((n_simulated_words // max(1, len(words_to_test_pool))) + 1))[:n_simulated_words]
    else:
        words_to_test = words_to_test_pool[:n_simulated_words] if n_simulated_words else words_to_test_pool

    length_dist = Counter(len(w) for w in words_to_test)
    print(f"\nMETHODOLOGY - word test set construction:")
    print(f"  Total simulated words tested: {len(words_to_test)}")
    print(f"  Construction method: for each target word, each letter is realized by "
          f"randomly sampling ONE real held-out image (from Half B) of that letter, "
          f"then running it through the trained classifier to get a predicted letter "
          f"and confidence score.")
    print(f"  Word-length distribution: {dict(sorted(length_dist.items()))}")
    print(f"  Dictionary coverage: {len(words_to_test_pool)}/{len(dictionary)} "
          f"dictionary words were constructible given available test letters "
          f"({len(words_to_test_pool)/len(dictionary)*100:.1f}%)")

    raw_correct = 0
    corrected_correct = 0
    letter_level_correct = 0
    letter_level_total = 0

    for word in words_to_test:
        pred_letters, conf_list = [], []
        for ch in word:
            idx_choice = random.choice(by_letter_indices_B[ch])
            pred_letters.append(preds_B_letters[idx_choice])
            conf_list.append(confidences_B[idx_choice])
            letter_level_total += 1
            if preds_B_letters[idx_choice] == ch:
                letter_level_correct += 1

        raw_word = "".join(pred_letters)
        if raw_word == word:
            raw_correct += 1

        corrected_word = autocorrect_word(pred_letters, conf_list, cost_table, dictionary)
        if corrected_word == word:
            corrected_correct += 1

    n = len(words_to_test)
    print("\n" + "=" * 70)
    print("FINAL RESULTS (Half A and Half B kept fully independent)")
    print("=" * 70)
    print(f"  Letter-level accuracy (Half B, independent):     {letter_level_correct/letter_level_total:.4f}")
    print(f"  Word-level accuracy WITHOUT autocorrection:      {raw_correct/n:.4f}")
    print(f"  Word-level accuracy WITH autocorrection:         {corrected_correct/n:.4f}")
    print(f"  Improvement from autocorrection: +{(corrected_correct-raw_correct)/n*100:.1f} percentage points")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kaggle", required=True)
    parser.add_argument("--aslhg", required=True)
    parser.add_argument("--third", required=True)
    parser.add_argument("--n_words", type=int, default=300)
    parser.add_argument("--dictionary_file", default=None)
    args = parser.parse_args()
    main(args.kaggle, args.aslhg, args.third, args.n_words, args.dictionary_file)
