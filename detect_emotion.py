#!/usr/bin/env python3
import os
import cv2
import numpy as np
import pandas as pd

from facenet_pytorch import MTCNN
from emotiefflib.facial_analysis import EmotiEffLibRecognizer, get_model_list
from deepface import DeepFace


# -----------------------------
# Face detection (high threshold) - MTCNN is passed in (built once)
# -----------------------------
def recognize_faces(frame_rgb, mtcnn, conf_thresh=0.98):
    """
    Detect faces with MTCNN (instance provided).
    Returns (boxes, confidences) where boxes are [x1,y1,x2,y2] ints, ordered, clipped.
    """
    boxes, probs = mtcnn.detect(frame_rgb, landmarks=False)

    out_boxes, out_confs = [], []
    if boxes is not None:
        H, W = frame_rgb.shape[:2]
        for b, p in zip(boxes, probs):
            if p is None or float(p) <= conf_thresh:
                continue
            x1, y1, x2, y2 = [int(v) for v in b]
            if x2 < x1: x1, x2 = x2, x1
            if y2 < y1: y1, y2 = y2, y1
            x1 = max(0, min(x1, W - 1)); x2 = max(0, min(x2, W - 1))
            y1 = max(0, min(y1, H - 1)); y2 = max(0, min(y2, H - 1))
            if (x2 - x1) > 2 and (y2 - y1) > 2:
                out_boxes.append([x1, y1, x2, y2])
                out_confs.append(float(p))
    return out_boxes, out_confs


# -----------------------------
# Emotion analysis + probability (top-1 softmax from logits)
# -----------------------------
def analyze_face_emotion(face_rgb, fer, frame_count):
    """
    Return emotion_label, valence, arousal, emotion_prob (top-1 softmax from logits).
    """
    try:
        emotion_result, scores = fer.predict_emotions(face_rgb, logits=True)
        emotion_label = emotion_result[0] if emotion_result else "unknown"

        # last two columns are valence / arousal in EmotiEffLib's multi-head
        valence = float(scores[0, -2])
        arousal = float(scores[0, -1])

        # first 8 are emotion logits -> softmax to get a confidence-like probability
        logits = np.array(scores[0, :8], dtype=np.float32)
        logits = logits - logits.max()  # numerical stability
        probs = np.exp(logits) / (np.exp(logits).sum() + 1e-12)
        emotion_prob = float(probs.max())

        return emotion_label, valence, arousal, emotion_prob
    except Exception as e:
        print(f"Error analyzing face on frame {frame_count}: {e}")
        return "error", 0.0, 0.0, 0.0


# -----------------------------
# Reference helpers
# -----------------------------
def _index_guest_refs(refs_dir):
    """
    Build a small index: key = lowercase stem (no extension), value = BGR image.
    refs/TimH.jpg -> key 'timh'
    """
    ref_map = {}
    if not refs_dir or not os.path.isdir(refs_dir):
        return ref_map
    for n in os.listdir(refs_dir):
        stem, ext = os.path.splitext(n)
        if ext.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        img = cv2.imread(os.path.join(refs_dir, n))  # BGR
        if img is None:
            continue
        ref_map[stem.lower()] = img
    return ref_map


def _find_ref_for_video(video_basename, ref_map):
    """
    Find the best matching ref by substring: longest key contained in the video filename (case-insensitive).
    Returns (ref_img_bgr, ref_key) or (None, None).
    """
    name = os.path.splitext(video_basename)[0].lower()
    matches = [k for k in ref_map.keys() if k in name]
    if not matches:
        return None, None
    matches.sort(key=len, reverse=True)
    if len(matches) > 1 and len(matches[0]) == len(matches[1]):
        print(f"⚠️ Ambiguous refs for '{video_basename}': {matches}. Using '{matches[0]}'.")
    return ref_map[matches[0]], matches[0]


# -----------------------------
# Per-video summary
# -----------------------------
def _video_summary_from_df(df_one_video):
    """
    One-row summary per video, including avg valence (overall, positive-only, negative-only),
    avg arousal, top emotion, emotion mix, and mean top-1 probability.
    """
    import numpy as np
    import pandas as pd

    if df_one_video.empty:
        return pd.DataFrame([{"video_file": None, "frames_analyzed": 0}])

    video = df_one_video["video_file"].iloc[0]
    N = len(df_one_video)

    # intensity
    val = df_one_video["valence"].astype(float).to_numpy()
    aro = df_one_video["arousal"].astype(float).to_numpy()

    pos = val[val > 0]
    neg = val[val < 0]

    valence_mean = float(val.mean()) if N else 0.0
    valence_pos_mean = float(pos.mean()) if pos.size else np.nan
    valence_neg_mean = float(neg.mean()) if neg.size else np.nan
    arousal_mean = float(aro.mean()) if N else 0.0

    # discrete emotions
    emo_counts = df_one_video["emotion"].value_counts(normalize=True, dropna=False)
    EMOS = ["Neutral", "Happiness", "Sadness", "Surprise", "Fear", "Disgust", "Anger", "Contempt", "error", "unknown"]
    emo_cols = {f"freq_{e}": round(float(emo_counts.get(e, 0.0)), 4) for e in EMOS}
    top_emo = emo_counts.idxmax() if not emo_counts.empty else ""

    # confidence (softmax top-1)
    prob_mean = float(df_one_video["emotion_prob"].mean()) if "emotion_prob" in df_one_video else np.nan

    row = {
        "video_file": video,
        "frames_analyzed": N,
        "valence_mean": round(valence_mean, 4),
        "valence_pos_mean": round(valence_pos_mean, 4) if not np.isnan(valence_pos_mean) else "",
        "valence_neg_mean": round(valence_neg_mean, 4) if not np.isnan(valence_neg_mean) else "",
        "arousal_mean": round(arousal_mean, 4),
        "top_emotion": top_emo,
        **emo_cols,
        "emotion_prob_mean": round(prob_mean, 4) if not np.isnan(prob_mean) else "",
    }
    return pd.DataFrame([row])


def _save_excel_with_video_summary(df, excel_filename):
    """
    Per-video Excel: 'All Data'+ 'Video Summary'.
    """
    cols_keep = [c for c in df.columns if c in {
        "video_file", "frame", "timestamp_seconds",
        "emotion", "emotion_prob",
        "valence", "arousal",
        "guest_distance", "guest_threshold"
    }]
    df_out = df[cols_keep].copy()

    vs = _video_summary_from_df(df)

    with pd.ExcelWriter(excel_filename, engine="openpyxl") as writer:
        df_out.to_excel(writer, sheet_name="All Data", index=False)
        vs.to_excel(writer, sheet_name="Video Summary", index=False)


def _save_aggregated_excel(df_all, output_filename):
    """
    Aggregated Excel across all processed videos:
    - 'All Data' (minimal)
    - 'Video Summary' (one row per video, union of emotions)
    """
    cols_keep = [c for c in df_all.columns if c in {
        "video_file", "frame", "timestamp_seconds",
        "emotion", "emotion_prob",
        "valence", "arousal",
        "guest_distance", "guest_threshold"
    }]
    df_out = df_all[cols_keep].copy()

    # Build per-video summaries, then align freq_* columns by filling missing with 0
    summaries = []
    for _, g in df_all.groupby("video_file"):
        summaries.append(_video_summary_from_df(g))
    if summaries:
        vs_all = pd.concat(summaries, ignore_index=True)
        freq_cols = [c for c in vs_all.columns if c.startswith("freq_")]
        if freq_cols:
            vs_all[freq_cols] = vs_all[freq_cols].fillna(0.0)
    else:
        vs_all = pd.DataFrame()

    with pd.ExcelWriter(output_filename, engine="openpyxl") as writer:
        df_out.to_excel(writer, sheet_name="All Data", index=False)
        if not vs_all.empty:
            vs_all.to_excel(writer, sheet_name="Video Summary", index=False)


# -----------------------------
# Core processing (guest-only; saves a debug image for every processed frame)
# -----------------------------
def process_video_with_ref(video_path,
                           guest_ref_bgr,
                           mtcnn,
                           fer,
                           pad_ratio=0.20,
                           frame_interval=10,
                           max_frames=0,
                           debug_dir="emotion_debug"):
    """
    Reference-only pipeline:
      - Detect all faces in a sampled frame (MTCNN).
      - For each face, pad the crop (20%) and DeepFace.verify against the reference
        with DeepFace's internal detect+align (model=VGG-Face, backend=opencv).
      - Keep ONLY the best verified match in that frame; run emotion on the tight crop.
      - Save an annotated debug image for every processed frame (even if no match).
    """
    if guest_ref_bgr is None or getattr(guest_ref_bgr, "size", 0) == 0:
        print("❌ No valid reference image. Skipping.")
        return pd.DataFrame()

    # Hard-coded DeepFace defaults (no CLI switches)
    deepface_model_name = "VGG-Face"
    deepface_backend = "opencv"

    os.makedirs(debug_dir, exist_ok=True)
    vbase = os.path.splitext(os.path.basename(video_path))[0]
    out_frames_dir = os.path.join(debug_dir, vbase)
    os.makedirs(out_frames_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ ERROR: Could not open video: {video_path}")
        return pd.DataFrame()

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    print(f"\n📹 Video: {os.path.basename(video_path)}")
    if fps > 0:
        print(f"   FPS: {fps:.1f}, Frames: {total_frames}, Duration: {total_frames / fps:.1f}s")
    else:
        print(f"   Frames: {total_frames}")

    excel_data = []
    frame_idx = 0
    processed_frames = 0

    while True:
        if max_frames > 0 and frame_idx >= max_frames:
            break

        ret, frame_bgr = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            boxes_list, _ = recognize_faces(frame_rgb, mtcnn, conf_thresh=0.98)

            debug = frame_bgr.copy()
            H, W = frame_rgb.shape[:2]

            best_idx, best_distance, best_threshold = -1, None, None

            for i, (x1, y1, x2, y2) in enumerate(boxes_list):
                w, h = x2 - x1, y2 - y1
                if w <= 2 or h <= 2:
                    continue

                # padding so DeepFace's internal detector can find & align the face
                pad = int(pad_ratio * max(w, h))
                xx1 = max(0, x1 - pad); yy1 = max(0, y1 - pad)
                xx2 = min(W - 1, x2 + pad); yy2 = min(H - 1, y2 + pad)
                crop_bgr = frame_bgr[yy1:yy2, xx1:xx2]

                try:
                    res = DeepFace.verify(
                        img1_path=crop_bgr,
                        img2_path=guest_ref_bgr,
                        model_name=deepface_model_name,
                        detector_backend=deepface_backend,
                        align=True,
                        enforce_detection=False,
                    )
                    if res.get("verified", False):
                        dist = res.get("distance", None)
                        thr = res.get("threshold", None)
                        if best_distance is None or (dist is not None and dist < best_distance):
                            best_idx, best_distance, best_threshold = i, dist, thr
                except Exception:
                    # keep going if a candidate fails
                    pass

            # compute timestamp (used for both cases)
            timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            timestamp_sec = timestamp_ms / 1000.0 if timestamp_ms > 0 else (frame_idx / fps if fps else 0.0)

            if best_idx != -1:
                # emotion on the tight crop (no pad)
                x1, y1, x2, y2 = boxes_list[best_idx]
                face_rgb = frame_rgb[y1:y2, x1:x2]
                emotion_label, valence, arousal, emotion_prob = analyze_face_emotion(
                    face_rgb, fer, frame_idx
                )

                # record row
                row = {
                    "video_file": os.path.basename(video_path),
                    "frame": frame_idx,
                    "timestamp_seconds": round(timestamp_sec, 2),
                    "emotion": emotion_label,
                    "emotion_prob": round(emotion_prob, 4),
                    "valence": round(valence, 4),
                    "arousal": round(arousal, 4),
                    "guest_distance": round(best_distance, 4) if isinstance(best_distance, (int, float)) else None,
                    "guest_threshold": round(best_threshold, 4) if isinstance(best_threshold, (int, float)) else None,
                }
                excel_data.append(row)

                # draw all candidates faint
                for j, (bx1, by1, bx2, by2) in enumerate(boxes_list):
                    color = (128, 128, 128)
                    cv2.rectangle(debug, (bx1, by1), (bx2, by2), color, 1)
                    cv2.putText(debug, f"cand#{j}", (bx1, max(20, by1 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

                # highlight chosen
                cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 255, 0), 2)
                d_str = f"{best_distance:.2f}" if isinstance(best_distance, (int, float)) else "NA"
                t_str = f"{best_threshold:.2f}" if isinstance(best_threshold, (int, float)) else "NA"
                label = f"{emotion_label} | V:{valence:.2f} A:{arousal:.2f} | d:{d_str} t:{t_str}"
                cv2.putText(debug, label, (x1, min(H-10, y2 + 18)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
            else:
                # no verified match: draw candidates and note no match
                for j, (bx1, by1, bx2, by2) in enumerate(boxes_list):
                    color = (0, 0, 255)
                    cv2.rectangle(debug, (bx1, by1), (bx2, by2), color, 1)
                    cv2.putText(debug, f"cand#{j}", (bx1, max(20, by1 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
                cv2.putText(debug, "no verified match", (10, H - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

            # common overlay + save debug image for every processed frame
            cv2.putText(debug, f"t={timestamp_sec:.2f}s", (10, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            out_path = os.path.join(out_frames_dir, f"frame_{frame_idx:06d}.jpg")
            cv2.imwrite(out_path, debug)
            processed_frames += 1

        frame_idx += 1
        if frame_idx % 1000 == 0:
            print(f"  Progress: {frame_idx} frames, processed {processed_frames}")

    cap.release()

    if not excel_data:
        print("\n❌ No data collected (no verified matches)")
        return pd.DataFrame()

    return pd.DataFrame(excel_data)


# -----------------------------
# Batch helpers / CLI
# -----------------------------
def _list_videos_in_dir(input_dir):
    VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".m4v"}
    files = []
    for name in os.listdir(input_dir):
        path = os.path.join(input_dir, name)
        if os.path.isfile(path) and os.path.splitext(name.lower())[1] in VIDEO_EXTS:
            files.append(os.path.abspath(path))
    return sorted(files)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Reference-only talk-show emotion analysis (guest-focused).")
    parser.add_argument("--video", "-v", default="",
                        help="Path to a single video file. If omitted, processes all videos in --input-dir.")
    parser.add_argument("--input-dir", "-i", default="videos",
                        help="Directory containing videos to process (default: videos).")
    parser.add_argument("--refs-dir", default="",
                        help="Directory of reference images (e.g., refs/TimH.jpg, refs/FreekV.jpg). Auto-mapped by filename substring.")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Maximum frames to process (0=all).")
    parser.add_argument("--frame-interval", type=int, default=10,
                        help="Process every Nth frame (default: 10).")
    parser.add_argument("--aggregate-out", default="emotion_data_ALL.xlsx",
                        help="Path for aggregated Excel across all processed videos.")
    parser.add_argument("--pad", type=float, default=0.20,
                        help="Padding ratio around face crop for verification (default: 0.20).")
    args = parser.parse_args()

    # Build detector ONCE
    mtcnn = MTCNN(keep_all=True, post_process=False, device="cpu")

    # Emotion model (once)
    device = "cpu"
    model_name = get_model_list()[3]  # enet_b0_8_va_mtl
    print(f"🤖 Using emotion model: {model_name}")
    fer = EmotiEffLibRecognizer(engine="torch", model_name=model_name, device=device)

    # Index refs (required for this reference-only flow)
    ref_map = _index_guest_refs(args.refs_dir) if args.refs_dir else {}
    if not ref_map:
        print("❌ No references found. Provide --refs-dir with images named after guests (e.g., TimH.jpg).")
        return
    else:
        print(f"🔎 Loaded {len(ref_map)} reference image(s) from '{args.refs_dir}': {list(ref_map.keys())}")

    # Single video
    if args.video:
        if not os.path.exists(args.video):
            print(f"❌ Video file not found: {args.video}")
            return

        vbase = os.path.basename(args.video)
        guest_ref_bgr, ref_key = _find_ref_for_video(vbase, ref_map)
        if guest_ref_bgr is None or getattr(guest_ref_bgr, "size", 0) == 0:
            print(f"❌ No matching reference for '{vbase}'. This script requires a ref.")
            return
        print(f"🎯 Matched ref '{ref_key}' for video '{vbase}'")

        df = process_video_with_ref(
            video_path=args.video,
            guest_ref_bgr=guest_ref_bgr,
            mtcnn=mtcnn,
            fer=fer,
            pad_ratio=args.pad,
            frame_interval=args.frame_interval,
            max_frames=args.max_frames,
            debug_dir="emotion_debug",
        )
        if not df.empty:
            # Per-video Excel (All Data + Video Summary)
            base_name = os.path.splitext(os.path.basename(args.video))[0]
            excel_filename = f"emotion_data_{base_name}.xlsx"
            _save_excel_with_video_summary(df, excel_filename)
            print(f"\n💾 Per-video Excel saved: {excel_filename}  (rows: {len(df)})")

            # Aggregate (single video) for convenience
            _save_aggregated_excel(df, args.aggregate_out)
            print(f"📚 Aggregated Excel (single video) saved to: {args.aggregate_out}")
        return

    # Batch directory
    if not os.path.exists(args.input_dir):
        os.makedirs(args.input_dir, exist_ok=True)
        print(f"❌ Input directory '{args.input_dir}' not found. Created it for you.")
        print("Add video files (.mp4, .mov, .mkv, .avi, .m4v) and rerun.")
        return

    videos = _list_videos_in_dir(args.input_dir)
    if not videos:
        print(f"❌ No videos found in '{args.input_dir}'.")
        return

    print(f"🔎 Found {len(videos)} video(s) in '{args.input_dir}'.")
    all_dfs = []
    for idx, vp in enumerate(videos, 1):
        vbase = os.path.basename(vp)
        guest_ref_bgr, ref_key = _find_ref_for_video(vbase, ref_map)
        if guest_ref_bgr is None or getattr(guest_ref_bgr, "size", 0) == 0:
            print(f"\n=== [{idx}/{len(videos)}] {vbase} → skipped (no matching ref) ===")
            continue
        print(f"\n=== [{idx}/{len(videos)}] {vbase} → ref '{ref_key}' ===")

        try:
            df = process_video_with_ref(
                video_path=vp,
                guest_ref_bgr=guest_ref_bgr,
                mtcnn=mtcnn,
                fer=fer,
                pad_ratio=args.pad,
                frame_interval=args.frame_interval,
                max_frames=args.max_frames,
                debug_dir="emotion_debug",
            )
            if df is not None and not df.empty:
                # Per-video Excel (All Data + Video Summary)
                base_name = os.path.splitext(os.path.basename(vp))[0]
                excel_filename = f"emotion_data_{base_name}.xlsx"
                _save_excel_with_video_summary(df, excel_filename)
                print(f"💾 Saved: {excel_filename} (rows: {len(df)})")

                all_dfs.append(df)
        except Exception as e:
            print(f"⚠️ Error while processing '{vp}': {e}")

    if all_dfs:
        df_all = pd.concat(all_dfs, ignore_index=True)
        _save_aggregated_excel(df_all, args.aggregate_out)
        print(f"\n✅ Batch complete. Aggregated Excel saved to: {args.aggregate_out}")
        print(f"   Total rows aggregated: {len(df_all)}")
    else:
        print("\n⚠️ No data to aggregate (no verified matches across videos).")


if __name__ == "__main__":
    # pip install deepface opencv-python-headless pandas openpyxl pillow facenet-pytorch torch torchvision emotiefflib timm
    main()
