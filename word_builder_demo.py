

STABLE_FRAMES = 8      # ~0.27s at 30fps - how long a sign must hold to count
COOLDOWN_FRAMES = 4    # frames of "gap" required before the same letter can repeat


def debounce_stream(frame_predictions):
 
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
    junk_letters = list("QXZ")  #  "garbage transition" predictions

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
