# ASL Fingerspelling Recognition — Installable Mobile App (PWA)

This folder contains a complete Progressive Web App (PWA) version of the
ASL recognition demo. Once deployed, it can be **installed on Android**
directly from the browser — it gets its own icon, launches fullscreen
with no browser bars, and works offline after the first load.

## Files

- `index.html` — the app (camera, hand tracking, model, word builder, autocorrect, ASL reference guide)
- `manifest.json` — tells Android this is an installable app (name, icon, colours, fullscreen mode)
- `sw.js` — service worker; caches the app so it launches instantly and works offline
- `icon-192.png`, `icon-512.png` — app icons shown on the home screen
- `asl-chart.png` — ASL alphabet reference image (see below)

## asl-chart.png

`asl-chart.png` is the alphabet reference image shown above the written
handshape descriptions. It is included in this package because you sourced
it yourself from Wikimedia Commons:

https://commons.wikimedia.org/wiki/File:Asl_alphabet_gallaudet_ann.png

The attribution caption rendered under the image in the app reads
"Gallaudet ASL alphabet chart, public domain, via Wikimedia Commons". If you
ever swap in a different chart, update that caption in `index.html` to match
the new image's actual licence and source — the app is publicly deployed, so
an unlicensed chart is a real exposure, not a formality.

If the file is absent the app still works: the chart slot hides itself
automatically and the written handshape descriptions stand on their own.

## Deploy it (same as before)

1. Copy this whole folder's contents into your `asl-demo-deploy` folder,
   replacing what's there (keep the folder flat — all 5 files at the top level).
2. In PowerShell, from that folder:
   ```
   vercel --prod
   ```
3. Open the resulting HTTPS link on your Android phone in Chrome.

**After redeploying, force the phone to fetch the new page.** The service
worker cache name has been bumped to `asl-reader-v3`, which handles this
automatically in most cases, but if the performance panel does not appear:
Chrome menu → Settings → Site settings → All sites → your URL →
**Clear & reset**, then reload. If you installed the app to the home
screen, close it fully first.

## Install it on Android

Once the page loads on your phone, either:
- Tap the **"Install app"** button that appears next to the camera controls, OR
- Open Chrome's menu (three dots) and tap **"Add to Home screen"** / **"Install app"**

It will then appear in your app drawer like any other app, launch fullscreen,
and run without an address bar.

## About a real APK / Google Play

A PWA is a genuine installable Android app, but it is **not** an `.apk` file
and cannot be uploaded to the Google Play Store as-is. If you later need a
true APK (for Play Store submission or for the dissertation to claim a
"native Android application"), the realistic routes are:

1. **PWABuilder** (https://www.pwabuilder.com) — paste your deployed URL and it
   generates a signed Android package wrapping this PWA. This is the fastest
   route to a Play-Store-submittable artifact and requires no code changes.
2. **Capacitor** (https://capacitorjs.com) — wraps the same web app in a native
   Android shell you build locally with Android Studio. More control, more setup.
3. **Full native rebuild** (React Native / Kotlin + MediaPipe Tasks + TensorFlow
   Lite) — the most work by far; only worth it if native performance or
   native-only APIs become a requirement.

Note that Google Play submission also requires a developer account
(one-time fee), a privacy policy, store listing assets, and a review process —
plan for days, not hours, if that is the goal.

## For the dissertation

This PWA supports the "runs on an ordinary phone, no special hardware, no
server, no subscription" claim directly: it uses only the device camera, runs
the model in-browser on-device, and needs no backend. When writing the System
Design chapter, describe it accurately as a **mobile progressive web
application**, not a native Android application, unless you complete route 1
or 2 above.

### Collecting the on-device measurements

The app has an **On-device performance** section under the camera controls.
It is collapsed by default — click the heading to open it. It reports:

| Readout | What it is | How it is computed |
|---|---|---|
| Pipeline throughput | Inference cycles actually completed per second | cycles ÷ elapsed time over the last 120 cycles |
| Landmark extraction | Time inside `handLandmarker.detectForVideo()` | mean over the last 120 cycles |
| Classification | Time for normalisation + the forward pass | mean over the last 120 cycles |
| Total per frame (mean) | Landmark + classification | mean over the last 120 cycles |
| Total per frame (95th pct) | Tail latency | 95th percentile over the same window |
| Unthrottled capacity | Rate the device could sustain with the throttle removed | 1000 ÷ mean total |
| Classifier size | Parameter count and float32 footprint | computed from the embedded weights at load |

**Throughput is capped by design.** `DETECTION_INTERVAL_MS = 80` limits
inference to about 12.5 cycles per second, because a fingerspelling hold
lasts 1.4 s and running faster would only burn battery. So throughput will
read ≈12.5 fps on any reasonably modern phone. **Do not report that as the
device's limit** — report it as the configured operating rate, and report
unthrottled capacity separately as what the hardware can do. An examiner
will otherwise reasonably ask why a flagship phone manages only 12 fps.

Procedure:
1. Open the app on the phone you intend to name in the dissertation.
2. Start the camera, hold a hand in frame, and let it run for about 30 s so
   the rolling averages settle.
3. Tap **Copy measurements**. This puts a plain-text block on the clipboard
   containing every figure plus the user-agent string, screen size, camera
   resolution and core count. Paste it straight into your notes.
4. Note the phone model and browser version separately — the user-agent
   string identifies the phone, but state it in prose too.

**On model size, be specific about which number you mean.** There are three
different figures and they are not interchangeable:
- **Classifier parameters:** 7,034 (63×64 + 64 + 64×32 + 32 + 32×26 + 26).
- **Classifier weights as float32:** 28,136 bytes ≈ 27.5 KB. This is the
  figure to quote as the model size; it is what the weights occupy in memory.
- **`model_weights.json` on disk:** ≈145 KB. Larger only because JSON stores
  each float as decimal text. Quote this as the export file size if you
  mention it, not as the model size.
- **MediaPipe hand landmark model:** a separate ≈7.5 MB float16 file fetched
  from Google's CDN at load. It dominates the total download, and honesty
  requires stating it — the 27.5 KB figure is your contribution, not the
  whole system.
