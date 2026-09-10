# ASL Landmark Extraction + Training — Run This Locally

You have the Kaggle "ASL Alphabet" dataset unzipped, with `asl_alphabet_train`
and `asl_alphabet_test` folders. Everything below runs on YOUR machine —
no need to upload the image dataset anywhere.

## Folder structure you should have after unzipping

```
asl_alphabet_train/
    A/
        A1.jpg, A2.jpg, ...
    B/
        B1.jpg, ...
    ... (one folder per letter, plus "space", "del", "nothing")
asl_alphabet_test/
    A_test.jpg
    B_test.jpg
    ... (just a few test images, not organized by folder)
```

## Step 1 — Install Python packages (one-time setup)

Open a terminal / Command Prompt in the folder where you saved these
scripts, and run:

```
pip install mediapipe opencv-python scikit-learn tensorflow pandas numpy
```

This may take a few minutes. If you hit a "pip not found" error, try
`python -m pip install ...` instead.

## Step 2 — Extract landmarks from the TRAIN folder

```
python extract_landmarks.py --input "PATH_TO/asl_alphabet_train" --output landmarks_train.csv --max_per_class 300
```

Replace `PATH_TO` with wherever you unzipped it, e.g.
`C:\Users\Faruk\Downloads\asl_alphabet_train`.

**Important: `--max_per_class 300` on purpose.** The full dataset has
~3,000 images per letter — you don't need all of them, and processing
all ~87,000 images through MediaPipe would take a long time on a
laptop. 300 images per letter (about 7,200 total) is plenty to get a
real, meaningful baseline number for Friday. You can always re-run
later with more.

This will take roughly 10-25 minutes depending on your machine. It
will print progress as it finishes, and tell you how many images had
no detectable hand (some dataset images are low quality — that's normal,
don't worry about it).

**Output:** a file called `landmarks_train.csv` — this is small (a few
MB), NOT huge like the image folder. This is the file you should send
back to me.

## Step 3 — (Optional, if you have time) Extract a validation set

If you want the cross-dataset-style check without a second dataset yet,
you can split off some training images you didn't already use. Skip
this for now if time is tight — we can add it after Friday.

## Step 4 — Send me the CSV

Once `landmarks_train.csv` exists, upload just that CSV file back to
this chat (it'll be small — a few MB, not hundreds of MB). I'll run the
training script on it immediately and get you a real accuracy number
and confusion matrix.

## If something goes wrong

- **"No module named mediapipe"** → the pip install in Step 1 didn't
  finish. Re-run it and check for red error text.
- **Script runs but writes 0 rows** → double check the `--input` path
  points to the folder that directly CONTAINS the letter folders (A, B,
  C...), not one level above or below it.
- **It's very slow** → lower `--max_per_class` to 150 or 100. Fewer
  images per class still gives a valid (if slightly less precise)
  baseline number.
