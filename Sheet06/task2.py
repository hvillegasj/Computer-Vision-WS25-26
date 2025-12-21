import os
import glob
import cv2
import numpy as np
from task1 import MOG

mog = None

def get_foreground_masks(frame):
    global mog
    
    if mog is None:
        h , w = frame.shape[:2]
        mog = MOG(
             height=h,
            width=w,
            number_of_gaussians=3,
            background_thresh=0.5,
            lr=0.01
        )
    
    BG_pivot = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)

    # Update GMM parameters + get foreground mask
    fg_mask = mog.updateParam(frame, BG_pivot)

    # fg_mask: 0 = background, 255 = foreground
    return fg_mask

def boxes_from_mask(mask, min_area=800, max_area_ratio=0.15, min_aspect=1.2, max_aspect=6.0):
    """
    Convert mask to person-like boxes.

    max_area_ratio: ignore blobs bigger than this fraction of the image
                    (removes the huge 'whole image' foreground region).
    """
    H, W = mask.shape[:2]
    max_area = int(max_area_ratio * H * W)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h

        # too small -> noise
        if area < min_area:
            continue

        # too big -> likely the whole background region got marked
        if area > max_area:
            continue

        if w <= 0:
            continue

        ar = h / float(w)
        if ar < min_aspect or ar > max_aspect:
            continue

        boxes.append((x, y, w, h))

    return boxes

def box_center(box):
    x, y, w, h = box
    return (x + w/2.0, y + h/2.0)

def center_dist(a, b):
    ax, ay = box_center(a)
    bx, by = box_center(b)
    return ((ax - bx)**2 + (ay - by)**2) ** 0.5

def match_by_distance(tracks, det_boxes, dist_thresh=40):
    matches = []
    used_dets = set()

    for t_idx, t in enumerate(tracks):
        best_d = None
        best_j = None
        for j, dbox in enumerate(det_boxes):
            if j in used_dets:
                continue
            d = center_dist(t.box, dbox)
            if best_d is None or d < best_d:
                best_d = d
                best_j = j

        if best_j is not None and best_d <= dist_thresh:
            matches.append((t_idx, best_j))
            used_dets.add(best_j)

    unmatched_dets = [j for j in range(len(det_boxes)) if j not in used_dets]
    unmatched_tracks = [i for i in range(len(tracks)) if i not in {m[0] for m in matches}]

    return matches, unmatched_tracks, unmatched_dets

class Track:
    """Simpel tracking of a person 
    id : uniquie track ID
    box : last known bounding box
    last_seen : last frame index where it appeared
    hits : how many frames it was detected 
    
    """
    
    def __init__(self, track_id, box, frame_idx):
        self.id = track_id
        self.box = box
        self.last_seen = frame_idx
        self.hits = 1

def match_detections_to_track(tracks, det_boxes, iou_threshold=0.2):
    """
    Greedy matching based on IoU . Compute IoU between every track box and detection box and assign best pairs if IoU >= threshold. 
    Unmatches detections start new tracks
    """
    
    assigned_tracks = set()
    assigned_dets = set()
    matches = []

    # Make a list of all candidate matches with their IoU scores
    candidates = []
    for t_idx, t in enumerate(tracks):
        for d_idx, d in enumerate(det_boxes):
            candidates.append((iou(t.box, d), t_idx, d_idx))

    # Sort by best IoU first
    candidates.sort(reverse=True, key=lambda x: x[0])

    # Greedily assign matches
    for score, t_idx, d_idx in candidates:
        if score < iou_threshold:
            break
        if t_idx in assigned_tracks or d_idx in assigned_dets:
            continue
        assigned_tracks.add(t_idx)
        assigned_dets.add(d_idx)
        matches.append((t_idx, d_idx))

    unmatched_tracks = [i for i in range(len(tracks)) if i not in assigned_tracks]
    unmatched_dets = [i for i in range(len(det_boxes)) if i not in assigned_dets]

    return matches, unmatched_tracks, unmatched_dets

def count_people_in_frames(frames):
    tracks = []
    next_id = 1
    confirmed_ids = set()
    
    warmup = 3
    # Hyper parameters (can be tuned)
    min_area = 600
    min_aspect = 1.3
    max_aspect = 5.5

    max_missing = 3   # if not seen for >3 frames ; drop it
    min_hits = 3    # must be seen at least thrice to count (reduces false positives)

    for f_idx, frame in enumerate(frames):
        # 1) foreground mask from background subtraction
        fg = get_foreground_masks(frame)
        fg = (fg > 0).astype(np.uint8) * 255

        # invert if mostly white
        if np.mean(fg) > 127:
            fg = 255 - fg
        
        if f_idx < warmup:
            continue
        

        # 3) get person-like boxes
        det_boxes = boxes_from_mask(fg, min_area=min_area, max_area_ratio=0.08, min_aspect=min_aspect, max_aspect=max_aspect)

        #print(f"[DEBUG] frame {f_idx}: mean={np.mean(fg):.1f}, boxes={len(det_boxes)}")
        
        # 4) match detections to existing tracks
        matches, _, unmatched_dets = match_by_distance(tracks, det_boxes, dist_thresh=50)
        
        # update matched tracks
        for t_idx, d_idx in matches:
            tracks[t_idx].box = det_boxes[d_idx]
            tracks[t_idx].last_seen = f_idx
            tracks[t_idx].hits += 1

            if tracks[t_idx].hits >= min_hits:
                confirmed_ids.add(tracks[t_idx].id)

        # new tracks
        for d_idx in unmatched_dets:
            tracks.append(Track(next_id, det_boxes[d_idx], f_idx))
            confirmed_ids.add(next_id)  # because min_hits=1 for now
            next_id += 1

        # keep tracks alive longer
        tracks = [t for t in tracks if (f_idx - t.last_seen) <= max_missing]
        
    return len(confirmed_ids)

def load_frames(img_dir = "imgs"):
    files = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    
    if len(files) == 0:
        raise FileNotFoundError("No images found inside folder")
    
    frames = []
    for f in files:
        img = cv2.imread(f)
        if img is None:
            print(f"[WARN] Could not read the image : {f}")
            continue
        
        frames.append(img)
    
    return frames

def main():
    
    img_dir = "imgs"
    
    # Load frames
    frames = load_frames(img_dir)
    
    # Count unique people
    n_people = count_people_in_frames(frames)
    
    print(f"The number of people in frames : {n_people}")
    
if __name__ == "__main__":
    main()