"""
word_builder_demo.py

Demonstrates the "debounce" segmentation logic for turning a stream of
per-frame letter predictions into a clean spelled word.

The core idea:
  - The classifier predicts a letter on EVERY frame (30x per second).
  - Most of those predictions are noisy/wrong during hand transitions.
  - We only "commit" a letter once it has been the top prediction for
    STABLE_FRAMES consecutive frames.
  - After committing a letter, we require a brief "no stable sign" gap
    before the next letter can be committed, so "AAAA" doesn't collapse
    into one A only to somehow also avoid merging a genuine double
    letter like "LL" in "HELLO".

This script simulates a noisy frame-by-frame prediction stream (as if
it came from the real-time classifier) and shows the clean output.
"""

STABLE_FRAMES = 8      # ~0.27s at 30fps - how long a sign must hold to count
COOLDOWN_FRAMES = 4    # frames of "gap" required before the same letter can repeat


def debounce_stream(frame_predictions):
    """
    frame_predictions: list of per-frame predicted letters (strings),
                        including noisy/transitional junk.
    Returns: the committed word (string).
    """
    word = []
    current_candidate = None
    candidate_count = 0
    cooldown = 0
    last_committed = None

    for pred in frame_predictions:
        if cooldown > 0:
            cooldown -= 1

        if pred == current_candidate:
            candidate_count += 1
        else:
            current_candidate = pred
            candidate_count = 1

        # Commit only if: stable long enough, not in cooldown,
        # and (different from the last committed letter OR cooldown has cleared,
        # which allows genuine double letters like "LL")
        if candidate_count == STABLE_FRAMES and cooldown == 0:
            word.append(current_candidate)
            last_committed = current_candidate
            cooldown = COOLDOWN_FRAMES
            candidate_count = 0  # require fresh stability for a repeat letter

    return "".join(word)


def simulate_noisy_stream(true_word, frames_per_letter=15, transition_noise=4):
    """
    Builds a fake noisy per-frame prediction stream for a target word,
    inserting a few junk/transition frames between each real letter -
    mimicking what the live classifier would actually output.
    """
    import random
    random.seed(1)
    stream = []
    junk_letters = list("QXZ")  # stand-ins for "garbage transition" predictions

    for letter in true_word:
        stream += [letter] * frames_per_letter
        stream += [random.choice(junk_letters) for _ in range(transition_noise)]
    return stream


if __name__ == "__main__":
    for test_word in ["HELLO", "CAT", "GOOD"]:
        noisy_stream = simulate_noisy_stream(test_word)
        result = debounce_stream(noisy_stream)
        print(f"Target word: {test_word}")
        print(f"  Simulated noisy frame stream length: {len(noisy_stream)} frames")
        print(f"  Debounced output: {result}")
        print(f"  Correct: {result == test_word}")
        print()
