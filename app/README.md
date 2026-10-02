# The deployed application

A progressive web application that performs ASL fingerspelling recognition
entirely in the browser on the user's own device. There is no server
component: camera frames, landmark extraction, classification and lexical
correction all happen locally, and no image leaves the device.

## Files

| File | Purpose |
|---|---|
| `index.html` | The complete application. Model weights and dictionary are embedded, which is why it is ~270 KB. |
| `manifest.json` | Declares the app installable: name, icons, colours, fullscreen display mode. |
| `sw.js` | Service worker. Network-first for the page so redeployments appear immediately; cache-first for static assets so the app launches offline. |
| `icon-192.png`, `icon-512.png` | Home-screen icons. |
| `asl-chart.png` | Alphabet reference image shown in the help panel. |

## Running it

Any static host will serve it; the files are self-contained and there is no
build step. HTTPS is required, because browsers only grant camera access on a
secure origin.

```
vercel --prod
```

After redeploying, the service worker may serve a cached page on devices that
already have it installed. The cache name is versioned (`asl-reader-v6`) and
is bumped whenever the page changes, which handles this in most cases. If a
device still shows the old version: Chrome menu → Settings → Site settings →
All sites → the deployed URL → **Clear & reset**, then reload. Close the
installed app fully first if it was added to the home screen.

## Installing on Android

Once the page loads, either tap **Install app** next to the camera controls,
or use Chrome's menu → **Add to Home screen**. The app then appears in the
launcher, opens fullscreen without browser chrome, and runs offline.

It is a progressive web application, not an `.apk`, and cannot be submitted to
the Play Store in this form. Wrapping it for store distribution — with
PWABuilder or Capacitor — is possible without changing any of the code here,
but was not undertaken.

## The alphabet reference image

`asl-chart.png` is the Gallaudet ASL alphabet chart, public domain, obtained
from Wikimedia Commons:

https://commons.wikimedia.org/wiki/File:Asl_alphabet_gallaudet_ann.png

The attribution caption is rendered beneath the image in the application. If
the chart is replaced, the caption in `index.html` must be updated to match
the new image's licence and source.

If the file is absent the application still works; the chart panel hides
itself and the written handshape descriptions stand alone.

## On-device performance panel

The application includes an **On-device performance** section beneath the
camera controls, collapsed by default. It exists so that the latency and
throughput figures reported in the dissertation can be reproduced by anyone
running the application.

| Readout | Definition |
|---|---|
| Pipeline throughput | Completed inference cycles per second, over the rolling window |
| Landmark extraction | Mean time inside `handLandmarker.detectForVideo()` |
| Classification | Mean time for normalisation plus the forward pass |
| Total per frame | Mean, median, 95th percentile and worst observed |
| Unthrottled capacity | 1000 ÷ mean total: the rate the device could sustain without the throttle |
| Classifier size | Parameter count and float32 footprint, computed from the embedded weights at load |

All statistics use a rolling window of the last 120 inference cycles, after
discarding the first 20 as warm-up. The panel reports whether the reading has
stabilised; **Copy measurements** places the full set on the clipboard
together with the user-agent string, screen size, camera resolution and core
count.

### Inference rate

Inference is throttled to one cycle per 80 ms. Because the throttle is checked
within animation-frame callbacks, which fire at the display refresh interval,
the effective rate settles near 10 cycles per second rather than the nominal
12.5. This is sufficient: the 1.4 s stability window that gates letter
commitment accumulates roughly fourteen independent predictions per letter, and
a higher rate would add no discriminative evidence while increasing power draw.

### Model size

Three distinct figures apply and are not interchangeable:

- **7,034 parameters** — the classifier's size, from 63×64 + 64 + 64×32 + 32 +
  32×26 + 26.
- **28,136 bytes (27.5 KB)** — those parameters as float32. This is the
  classifier's footprint.
- **~145 KB** — `model_weights.json` on disk. Larger only because JSON stores
  each float as decimal text.

Separately, the MediaPipe hand landmark model is an external dependency of
approximately 7.5 MB, fetched once from Google's CDN and cached thereafter. It
dominates the total download.
