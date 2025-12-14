# Talk-Show Emotion Analyzer

Pipeline for Sheila to quantify a target guest's emotions across many videos.

**Input:** a folder of videos, plus one reference image per guest (e.g., `refs/TimH.jpg`, `refs/FreekVonk.jpg`)  

**Process:** on each sampled frame → detect faces → verify which face matches the supplied reference (DeepFace default distance & threshold) → run a lightweight emotion model (EmotiEffLib) → save results  

**Output:** a single aggregated Excel workbook across all videos and debug images

---

## 1) Quick start (macOS Apple Silicon)

**Prereqs:** Python **3.10**

```bash
# clone the repo, then:
bash script/setup_macos_arm64.sh
source .venv/bin/activate
```

**Project structure:**

```
your-project/
  detect_emotion.py
  videos/          # put .mp4/.mov/.mkv/.avi here (e.g., TimH_arjenlubach.mp4, FreekV_1.mp4)
  refs/            # reference images (e.g., TimH.jpg, FreekV.jpg)
  scripts/
    setup_macos_arm64.sh
  requirements.txt
```

**Batch run:**

```bash
python detect_emotion.py --input-dir videos --refs-dir refs --frame-interval 40 --aggregate-out emotion_data_ALL.xlsx
```

- `--frame-interval 40` samples roughly every 1.6s on 24–25 fps footage (increase to go faster).

- Put one photo per guest in `refs/` (e.g., `TimH.jpg`). The tool picks the photo whose filename appears inside the video filename.  
  Examples: `TimH_arjenlubach.mp4` → `refs/TimH.jpg`, `FreekV_talk1.mp4` → `refs/FreekV.jpg`.

---

## 2) Args (short)

**`-i, --input-dir <DIR>`**  
Folder to scan for videos. Default: `videos`

**`--refs-dir <DIR>`**  
Folder of reference images. Auto-match by filename substring (e.g., `refs/TimH.jpg` matches `TimH_*.mp4`). Required.

**`--frame-interval <INT>`**  
Sample every Nth frame. Larger = faster, fewer samples. Default: `10`

**`--max-frames <INT>`**  
Hard cap per video. 0 = no cap. Default: `0`

**`--aggregate-out <PATH>`**  
Combined Excel across all processed videos. Default: `emotion_data_ALL.xlsx`

**`--pad <FLOAT>`**  
Padding ratio around the detected face for verification (e.g., 0.20 = 20%). Default: `0.20`

---

## 3) Output

**Aggregated Excel:** `emotion_data_ALL.xlsx` (two sheets)

**All Data** (one row per verified frame)
- `video_file`, `frame`, `timestamp_seconds`
- `emotion` (top-1), `emotion_prob` (softmax of logits, 0–1)
- `valence`, `arousal`
- `guest_distance`, `guest_threshold`

**Video Summary** (one row per video)
- `frames_analyzed`
- `valence_mean`, `valence_pos_mean` (mean of valence>0), `valence_neg_mean` (mean of valence<0)
- `arousal_mean`
- `top_emotion`
- `freq_*` columns = fraction of frames for each discrete emotion
- `emotion_prob_mean` = average top-1 probability

**Debug images:** saved under `emotion_debug/<video_name>/frame_*.jpg`, only when the guest was verified in that frame.

---

## 4) Reference images

- Place guest photos in `refs/` named after the guest (e.g., `TimH.jpg`, `FreekV.jpg`).
- A video uses the ref whose name appears in the video filename. If no match is found, that video is skipped.
- Use clean, frontal photos. A small padding is applied to the candidate crop to help DeepFace align.

---

## 5) Design choices & work process

### Emotion detection
The initial thought I have is figuring out differences in emotional intensity, and how Sheila would be able to measure different intensity levels. In the provided links, default outputs of the models are discrete emotions. I thought perhaps supplying Sheila with valence and arousal, through EmotiEff's `enet_b0_8_va_mtl`, would prove to be more useful since you get continuous values per frame, while still outputting discrete emotions.

### Face selection
The second problem that I thought about is how you could keep track of the guest face. Considering talkshows have different angles, cuts, and setups. Perhaps in one talkshow there is a live audience, or there are more than 1 host. The solution that I came up with is to provide a reference image for the target host. I believe Sheila has this information of the guest list that she would like to evaluate.

### Sampling
I would think that the software should sample per N number of frames, with some flexibility on the sampling rate for Sheila. I also think Sheila needs to verify the sample and the output of the emotion detection model for that particular sample, which led me to include a debug image for every sample. I also include the summary of possibly useful stats, such as average positive valence, average negative valence, arousal average, etc.

---

## 6) Challenges

- **Inaccurate face selection in edge cases:** profile views, tiny faces, look-alikes, inserts and banners that show faces out of context. Perhaps, in this tool's case, we still need to occasion verification on the debugged image to make sure the correct face is verified and analysed.

- **Emotion predictions:** valence/arousal and discrete labels do not always match. Accuracy of emotion detected can be somewhat poor

---

## 7) Model references 

- **MTCNN** (face detection) — arXiv: https://arxiv.org/abs/1604.02878

- **VGG-Face** (ID verification backbone, via DeepFace) — BMVC 2015: http://www.robots.ox.ac.uk/~vgg/publications/2015/Parkhi15/ • DeepFace: https://github.com/serengil/deepface

- **EmotiEffLib `enet_b0_8_va_mtl`** (discrete emotions + valence/arousal) — https://github.com/sb-ai-lab/EmotiEffLib


