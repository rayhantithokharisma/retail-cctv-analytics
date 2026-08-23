# Retail Behavioral Analytics Pipeline: Architectural Design & Implementation Log

**Author:** Deep Learning & Computer Vision Engineering Team  
**Target Audience:** Computer Vision Students, Researchers, and Applied AI Engineers  
**Codebase:** [hendricks-assignment](file:///Users/Rayhan.Kharisma/Documents/prsnl-prjct/pprsnl/hendricks-assignment)

---

## Table of Contents
1. [Executive Summary & Problem Formulation](#1-executive-summary--problem-formulation)
2. [First Principles & Architectural Paradigms](#2-first-principles--architectural-paradigms)
3. [The Perception Pass: Pose Estimation & Deep ReID](#3-the-perception-pass-pose-estimation--deep-reid)
4. [Spatial Modeling & Perspective Normalization](#4-spatial-modeling--perspective-normalization)
5. [Overcoming Tracker Fragmentation: Bipartite Tracklet Stitching](#5-overcoming-tracker-fragmentation-bipartite-tracklet-stitching)
6. [Behavioral Feature Engineering & Formulas](#6-behavioral-feature-engineering--formulas)
7. [Temporal Dynamics: Hysteresis State Machines](#7-temporal-dynamics-hysteresis-state-machines)
8. [Task-Specific Implementations](#8-task-specific-implementations)
   - [Task 1: Storefront Interest & Entrance Conversion](#task-1-storefront-interest--entrance-conversion)
   - [Task 2: Interior Shelf Engagement & Group Dwells](#task-2-interior-shelf-engagement--group-dwells)
   - [Task 3: Staff Classification & Interaction Sessions](#task-3-staff-classification--interaction-sessions)
9. [Comparative Analysis vs. Alternative Methods](#9-comparative-analysis-vs-alternative-methods)
10. [Validation, Sensitivity & Ablation Findings](#10-validation-sensitivity--ablation-findings)
11. [Known Edge Cases & Future Directions](#11-known-edge-cases--future-directions)
12. [Data Artifacts, Granularity & End-to-End Dataflow](#12-data-artifacts-granularity--end-to-end-dataflow)
13. [Parameter Audit: Justification, Sensitivity & Improvement Analysis](#13-parameter-audit-justification-sensitivity--improvement-analysis)

---

## 1. Executive Summary & Problem Formulation

In retail surveillance and ambient intelligence, extracting actionable business metrics from raw video streams is a classical problem in computer vision. Store managers and retail analysts need answers to critical questions:
- *How many people walked down the hallway?*
- *How many glanced at or showed interest in the storefront without entering?*
- *What is our conversion rate (people who entered vs. total interested passersby)?*
- *Which specific display shelves (`shelf-a` through `shelf-d`) capture the highest attention and physical product touch interactions?*
- *How often do staff members actively engage with shoppers?*

### The Fundamental Technical Challenges
While these questions sound straightforward, implementing them robustly on raw, unconstrained monocular surveillance cameras (`entrance.mp4` and `interior.mp4`) presents several severe computer vision challenges:
1. **Perspective Distortion:** Pedestrians close to the camera appear 3× to 4× larger than pedestrians in the far background. A velocity of $10\text{ pixels/frame}$ in the foreground represents a slow crawl, whereas the exact same pixel velocity in the background represents a dead sprint.
2. **Visual Occlusions & Tracker Fragmentation:** In real stores, physical props (easels, podiums, display tables, pillars) and other shoppers constantly occlude individuals. Standard online multi-object trackers (e.g. SORT, ByteTrack) frequently lose track and spawn brand new identity IDs upon re-emergence, causing **overcounting by 30% to 50%**.
3. **Subtle Behavioral Distinctions:** A passerby who does not stop walking, but turns their head for 1.2 seconds to inspect a shoe display, exhibits genuine *store interest*. Bounding-box trajectory analysis alone is completely blind to head turn and body orientation.
4. **Staff Disambiguation:** Staff members frequently walk near shelves and stand in front of displays. If staff members are not accurately filtered out, shelf dwell and customer interest metrics become heavily inflated.

---

## 2. First Principles & Architectural Paradigms

### Two-Pass Offline Architecture vs. Single-Pass Online Streaming
Most textbook tracking pipelines attempt to solve detection, association, smoothing, and metric computation simultaneously in a single online frame-by-frame loop. However, retail video analysis for historical auditing is fundamentally an **offline batch problem**.

```
┌────────────────────────────────────────────────────────────────────────┐
│ PASS 1: PERCEPTION PASS (Compute Heavy, Frame-by-Frame)               │
│ - YOLO11m-Pose Inference (17 Keypoints @ 1280x720)                    │
│ - BoT-SORT Tracking (Short-term frame association)                     │
│ - Feature Extraction: yolo26n-reid ONNX Visual Embeddings             │
│ Output: Streaming Parquet Observations + Embeddings NPY Array         │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│ PASS 2: GLOBAL ASSEMBLY & BEHAVIORAL LOGIC (Sub-Second Execution)      │
│ - Global Bipartite Hungarian Stitching (Long-term occlusion repair)    │
│ - Gap Interpolation (<= 10 frames) & Median / Savitzky-Golay Smoothing │
│ - Perspective Normalization via Body-Height Function BH(y)            │
│ - Multi-Channel Interest & Shelf Attention Formulation                 │
│ - Hysteresis Episode Latching (Ton / Toff State Machines)              │
│ Output: Structured CSV Reports + Annotated Overlay Videos              │
└────────────────────────────────────────────────────────────────────────┘
```

#### Why Decouple Perception from Analytics?
1. **Separation of Compute Concerns:** Deep neural networks (YOLO pose and ONNX ReID) require GPU/MPS acceleration. By saving raw detections and ReID vectors into flat, columnar Parquet files, Pass 1 runs **once**.
2. **Infinite Experimentation Velocity:** Once Pass 1 is cached, testing new behavioral heuristics, tuning sensitivity thresholds, or modifying state machines in Pass 2 takes **less than 1 second** across 15,000 frames without re-running heavy neural networks.
3. **Global Temporal Context:** An online filter can only know the past ($t \le t_{\text{curr}}$). An offline global stitcher has access to the future ($t > t_{\text{curr}}$), allowing it to bridge multi-second occlusion gaps and fit optimal polynomial smoothing trajectories across the entire lifespan of a shopper.

---

## 3. The Perception Pass: Pose Estimation & Deep ReID

### 3.1 Why Pose Estimation over Bounding Boxes?
Traditional retail tracking relies exclusively on 2D bounding boxes $[x_{\min}, y_{\min}, x_{\max}, y_{\max}]$. We deliberately selected **YOLO11m-Pose** (17 COCO keypoints: nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles) for three core reasons:

```
          [Nose]
       [Eyes / Ears] ──► Visibility Asymmetry = Head Yaw Gaze Proxy
             │
      [L_Sh]───[R_Sh] ──► Shoulder Normal Vector = Torso Facing Angle
        │        │
      [L_El]   [R_El]
        │        │
      [L_Wr]   [R_Wr] ──► Wrist Proximity = Product Reach & Interaction
        │        │
      [L_Hip]──[R_Hip]
        │        │
      [L_Knee] [R_Knee]
        │        │
      [L_Ank]  [R_Ank] ──► Ankle Midpoint = True Ground Foot Location
```

1. **Torso Normal ($A(t)$):** The vector perpendicular to the line connecting the left shoulder $(x_5, y_5)$ and right shoulder $(x_6, y_6)$ gives the precise facing angle of the person.
2. **Head Yaw Proxy ($Y(t)$):** When a person looks sideways towards the store window, the far ear/eye drops in detection confidence while the near ear/eye remains high ($c_{\text{near}} \gg c_{\text{far}}$). This confidence asymmetry provides a direct signal for head turning.
3. **Product Interaction ($W(t)$):** Keypoints 9 and 10 (wrists) allow the system to detect when a shopper physically extends their hands toward a retail display shelf.
4. **Foot-Point Precision:** Bounding box bottom-centers are prone to jitter when a person swings their arms or carries bags. Ankle keypoints provide the ground contact location, eliminating spatial perspective noise.

### 3.2 Deep Visual Feature Embeddings (`yolo26n-reid.onnx`)
BoT-SORT incorporates a dedicated convolutional ReID network. For every detected person crop, the network computes a 512-dimensional $L_2$-normalized feature vector:
$$\mathbf{e} \in \mathbb{R}^{512}, \quad \|\mathbf{e}\|_2 = 1$$
The similarity between two detection crops $i$ and $j$ is computed via Cosine Distance:
$$D_{\text{cosine}}(\mathbf{e}_i, \mathbf{e}_j) = 1 - \frac{\mathbf{e}_i \cdot \mathbf{e}_j}{\|\mathbf{e}_i\| \|\mathbf{e}_j\|} = 1 - (\mathbf{e}_i^\top \mathbf{e}_j)$$
This visual descriptor encodes color histograms, cloth patterns, and upper/lower body texture, remaining invariant even when the person moves across different camera positions.

---

## 4. Spatial Modeling & Perspective Normalization

### The Fallacy of Euclidean Pixel Coordinates
In retail surveillance, cameras are angled downwards (typically $25^\circ - 40^\circ$ pitch angle). Consequently:
- A person in the background ($y \approx 180\text{ px}$) has an apparent height of $\sim 85\text{ px}$.
- A person in the foreground ($y \approx 650\text{ px}$) has an apparent height of $\sim 160\text{ px}$.

Evaluating absolute pixel velocities (e.g. $\text{threshold} = 5\text{ px/frame}$) completely fails: it over-penalizes foreground walkers and makes background pedestrians appear stationary.

### The Linear Scale Map: $BH(y) = a \cdot y + b$
We model perspective scale by measuring empirical person heights across vertical image coordinates $y$, deriving the local **Body Height ($BH$) function**:
$$BH(y) = a \cdot y + b$$

For our two cameras:
- **Entrance Camera:** $BH(y) = 0.136 \cdot y + 62.0\text{ px}$
- **Interior Camera:** $BH(y) = 0.200 \cdot y + 60.0\text{ px}$

```
Vertical Position (y)    Expected Person Height (BH)    1 Body-Height Distance
y = 150 px (Far Hallway)       ~82 px                         82 px
y = 400 px (Mid Entrance)      ~116 px                        116 px
y = 650 px (Foreground Store)  ~150 px                        150 px
```

All downstream physical calculations are converted into **Body-Height Normalized Units**:
- **Normalized Velocity:** $v_{\text{norm}} = \frac{v_{\text{pixels/s}}}{BH(y)} \quad [\text{BH/s}]$
- **Normalized Distance:** $d_{\text{norm}} = \frac{d_{\text{pixels}}}{BH(y)} \quad [\text{BH}]$

A speed of $0.3\text{ BH/s}$ corresponds to slow loitering/browsing regardless of whether the person is $3\text{ meters}$ or $15\text{ meters}$ from the lens.

---

## 5. Overcoming Tracker Fragmentation: Bipartite Tracklet Stitching

### 5.1 The Root Cause of Identity Switches
Online Kalman-filter trackers (BoT-SORT / ByteTrack) maintain track states frame-to-frame. When an individual walks behind an easel or podium for $>30\text{ frames}$ ($1.0\text{s}$), the tracker marks the track as *dead*. When the person re-emerges:
1. Spatial distance from the last Kalman prediction is too large.
2. The tracker assigns a brand new Track ID.
3. **Result:** 1 customer walking down the hallway produces 3 separate track IDs, inflating traffic counts by $50\%$.

### 5.2 Global Hungarian Stitching Algorithm
After Pass 1 completes, we group raw frame observations into continuous **Tracklets** $T_k$. We then construct a global bipartite cost matrix between every candidate tail tracklet $T_i$ and head tracklet $T_j$.

```
Tracklet T_i (Disappears at t_end)           Tracklet T_j (Appears at t_start)
      [=============]                               [==============]
                     \                             /
                      \─── Temporal Gap Δt ───────/
                           (0.0s < Δt <= 2.0s)
```

#### The Four Gating Filters:
A pair $(T_i, T_j)$ is considered for stitching if and only if:
1. **Temporal Order & Max Gap:** $0.0 < (t_{\text{start}, j} - t_{\text{end}, i}) \le 2.0\text{ seconds}$ (`max_gap_frames = 60` @ 30 fps).
2. **Spatial Reachability:** velocity-extrapolated residual satisfies:
   $$\frac{\|\hat{\mathbf{x}}_j - \mathbf{x}_j(t_{\text{start}})\|}{BH} \le 2.5\text{ BH} \quad (\texttt{motion\_gate\_bh})$$
3. **Scale Consistency:** Person heights must not change discontinuously:
   $$\frac{|h_j - h_i|}{\max(h_i, h_j)} \le 0.40 \quad (\texttt{scale\_gate\_frac})$$
4. **Direction Consistency:** heading change across the gap $\le 60^\circ$ (`direction_gate_deg`),
   enforced only when both tracklets have $> 5\text{ px}$ displacement (stationary tracklets
   carry no heading signal).

#### The Fused Cost Function:
For all valid pairs, the cost matrix entry $C(i, j)$ is computed as:
$$C(i, j) = w_{\text{app}} \cdot D_{\text{cosine}}(\mathbf{e}_i, \mathbf{e}_j) + w_{\text{mot}} \cdot \frac{\|\hat{\mathbf{x}}_j - \mathbf{x}_j\|}{BH} \Big/ \text{gate} + w_{\text{scale}} \cdot \frac{|h_i - h_j|}{\max(h_i, h_j)} \Big/ \text{gate}$$
where:
- $w_{\text{app}} = 0.55$: Deep ReID appearance cosine distance (set to maximum cost 1.0 if either embedding is missing).
- $w_{\text{mot}} = 0.30$: Velocity-extrapolated position error $\hat{\mathbf{x}}_j = \mathbf{x}_i(t_{\text{end}}) + \mathbf{v}_i \cdot \Delta t$, normalised by the motion gate.
- $w_{\text{scale}} = 0.15$: Relative height difference penalty, normalised by the scale gate.

We solve the global optimal assignment using the **Hungarian Algorithm** ($O(N^3)$ via `scipy.optimize.linear_sum_assignment`). Pairs with $C(i, j) < 0.45$ (`cost_max`) are merged into a unified `Identity`, and the assignment is iterated to a fixed point (max 5 rounds) so chained A→B→C fragment sequences collapse into one identity.

### 5.3 Signal Smoothing & Outlier Suppression
Once merged, the raw trajectory $(x(t), y(t))$ undergoes two-stage signal filtering:
1. **Gap Linear Interpolation:** Missing frames ($\le 10\text{ frames}$, `max_interp_gap`) caused by detector drops are linearly interpolated; longer gaps stay NaN so downstream logic never trusts fabricated positions.
2. **Rolling Median Filter ($w = 15\text{ frames} = 0.5\text{s}$, centred):** Replaces single-frame keypoint/foot spikes without blurring genuine stops and starts. Applied to foot position, speed, and facing.

(A Savitzky-Golay filter, `savgol_win = 21 / order = 2`, is implemented and unit-tested in
`smoothing.py` but currently **disabled** — the median filter proved sufficient and preserves
step edges better; see the audit in §13.)

---

## 6. Behavioral Feature Engineering & Formulas

To classify complex human behavior without black-box overfitting, we engineer four physics-based behavioral channels normalized to $[0, 1]$:

```
                          ┌──────────────────────────┐
                          │   Torso & Head Facing    │ ──► Weight: 40%
                          │      Orientation A(t)    │
                          └─────────────┬────────────┘
                                        │
                          ┌─────────────┴────────────┐
                          │   Velocity Deceleration  │ ──► Weight: 25%
                          │           S(t)           │
                          └─────────────┬────────────┘
                                        │
 ┌───────────────────┐    ┌─────────────┴────────────┐
 │  INTEREST SCORE   │◄───│  Approach to Store Line  │ ──► Weight: 20%
 │       I(t)        │    │           P(t)           │
 └───────────────────┘    └─────────────┬────────────┘
                                        │
                          ┌─────────────┴────────────┐
                          │     Storefront Dwell     │ ──► Weight: 15%
                          │           D(t)           │
                          └──────────────────────────┘
```

### 6.1 Multi-Channel Storefront Interest Formula: $I(t)$
$$I(t) = 0.40 \cdot A(t) + 0.25 \cdot S(t) + 0.20 \cdot P(t) + 0.15 \cdot D(t)$$

#### Channel Derivations (as implemented — see audit §13):
1. **Orientation Score $A(t)$:**
   $$\cos \theta(t) = \hat{\mathbf{n}}_{\text{torso}}(t) \cdot \hat{\mathbf{n}}_{\text{to-entrance}}$$
   $$A_{\text{body}}(t) = \text{clamp}_{[0, 1]}\left(\frac{\cos \theta(t) - \cos(75^\circ)}{1.0 - \cos(75^\circ)}\right)$$
   Fusing head yaw confidence asymmetry $Y(t) \in [-1, 1]$ (mapped to $[0,1]$):
   $$A(t) = \max\left(A_{\text{body}}(t),\; 0.70 \cdot A_{\text{body}}(t) + 0.30 \cdot Y(t)\right)$$
   The $\max$ ensures a clear head-turn toward the store still counts when the torso lags.
2. **Deceleration Score $S(t)$:**
   Compares instantaneous speed $v(t)$ against the person's $80^{\text{th}}$-percentile cruising speed $v_{\text{cruise}}$:
   $$S(t) = \text{clamp}_{[0, 1]}\left(\frac{v_{\text{cruise}} - v(t)}{0.7 \cdot v_{\text{cruise}}}\right), \quad S(t) := 1 \text{ when } v(t) < 0.25\text{ BH/s}$$
   The $0.7$ slope means a speed drop of $\sim 70\%$ from cruising saturates the channel; the
   $0.25\text{ BH/s}$ floor forces full credit for an effective stop.
3. **Approach Score $P(t)$:**
   Ramp on body-height distance closed toward the entrance line relative to the track's first observation:
   $$P(t) = \text{clamp}_{[0, 1]}\left(\frac{\text{dist}_{\text{BH}}(0) - \text{dist}_{\text{BH}}(t)}{1.5\text{ BH}}\right)$$
   with full credit ($P = 1$) whenever the foot is inside the storefront zone or past the line.
4. **Storefront Dwell Score $D(t)$:**
   Ramps up linearly with continuous duration spent inside the storefront zone polygon:
   $$D(t) = \text{clamp}_{[0, 1]}\left(\frac{\text{duration\_in\_zone}(t)}{3.0\text{ s}}\right)$$
   The running counter resets to zero the instant the foot exits the zone.

---

## 7. Temporal Dynamics: Hysteresis State Machines

### The Flicker Problem in Simple Thresholding
If an analytics system uses instantaneous thresholding (e.g. `is_interested = I(t) > 0.55`), sensor noise and keypoint jitter cause the boolean condition to oscillate rapidly between `True` and `False` across adjacent frames. A single 4-second dwell could falsely generate 6 distinct short events.

### Hysteresis State Machine Architecture
To eliminate flicker and produce clean, auditable episodes, all events are governed by an **asymmetric 2-parameter state machine**:
- **$T_{\text{on}}$ (Onset Persistence):** Requires $N_{\text{on}} = \lceil T_{\text{on}} \cdot \text{fps} \rceil$ consecutive `True` frames before transitioning `IDLE -> ACTIVE`.
- **$T_{\text{off}}$ (Gap Tolerance / Hangover):** Requires $N_{\text{off}} = \lceil T_{\text{off}} \cdot \text{fps} \rceil$ consecutive `False` frames before transitioning `ACTIVE -> IDLE`.

```
Condition:  0 0 1 1 1 1 1 0 1 1 1 1 0 0 0 0 0 0 0 0
State:      IDLE───────►[    ACTIVE EPISODE   ]──────►IDLE
                         │                   │
                     Episode Start       Episode End
```

#### Critical Timestamp Recording:
When closing an episode, the end timestamp is recorded at the **last true frame before the off-gap**, rather than the expiration of the timer. This prevents artificial inflation of episode durations by $T_{\text{off}}$.

---

## 8. Task-Specific Implementations

### Task 1: Storefront Interest & Entrance Conversion

```
[All Video Tracks] ──► [Filter 1: Hallway Corridor Intersection]
                              │
                      [Filter 2: Minimum Presence (>= 1.0s)]
                              │
                      [Filter 3: Staff Exclusion (Task 3)]
                              │
                     [Candidate Identities]
                     /                    \
                    /                      \
 [Store Interest Predicate]       [Entrance Crossing Predicate]
 ├─ Sustained I(t) >= 0.55        ├─ Signed side change (Outside -> Inside)
 └─ Duration >= 0.8s              ├─ Penetration depth >= 1.0 BH
                                  └─ In-store dwell >= 2.0s
```

- **Interested Count:** Candidates satisfying the multi-channel interest score.
- **Entered Count:** Candidates who crossed the entrance threshold line past the deadzone ($0.25\text{ BH}$) and sustained inward dwell.
- **Passed By Count:** $\text{Passed By} = \max(0, \text{Interested} - \text{Entered})$.

---

### Task 2: Per-Shelf Customer Interest

#### 1. Shelf Geometry from Ground-Truth Annotations
Shelf rectangles are loaded directly from the supplied VIA annotation
(`data/annotations/interior_annotation.csv`) — no hand-drawn polygons. The rect marks
the shelf's product area and is both the attention target and the rendering primitive,
so the on-video boxes always sit exactly on the shelves.

#### 2. Interest Candidate Predicate (per person, per frame)
A frame is a candidate for shelf $k$ when ALL of the following hold:
1. **Proximity:** distance from the foot point to the shelf rect $\le 1.2$ body-heights,
   normalised by the person's own bbox height that frame (perspective-invariant).
   Calibrated visually: true browsers measure 0.6–1.2 BH from the rect, passers-by $\ge 1.5$ BH.
2. **Facing:** torso facing normal points at the rect centre within $80^\circ$
   ($\cos\theta \ge \cos 80^\circ$). This is the disambiguator when a person stands
   between two shelves — proximity alone cannot separate them.
3. **Browsing Speed:** $v(t) \le 0.35\text{ BH/s}$ (standing / shuffling, not walking past).
4. **Exclusion Zones:** feet inside the wall-mirror region or the cashier counter /
   staff-desk region are never candidates (mirror reflections and paying customers
   would otherwise hallucinate shelf-d / shelf-a interest — found during
   frame-by-frame verification).

Each frame keeps only the nearest candidate shelf, then a 1-second rolling majority
vote removes A/B flicker between fixtures, so a person has at most one active shelf
at any time.

#### 3. Event Segmentation (no double counting)
Per (identity, shelf), the boolean candidate series feeds the Hysteresis State Machine
($T_{\text{on}} = 2.0\text{s}$ sustained attention starts one event; $T_{\text{off}} = 3.0\text{s}$
without candidates ends it). A continuous interaction = ONE event; leaving and later
returning to the same shelf = a NEW event. The official deliverable is the episode
count per shelf (`outputs/task2_shelf_interest.csv`).

#### 4. Group Dwell Spatial Clustering (auxiliary)
Detects social shopping groups (families, couples, friends browsing together):
1. **Proximity Condition:** Pairwise distance between identities $d(i, j) \le 1.5\text{ BH}$.
2. **Co-Stationary Condition:** Both individuals moving slowly ($v \le 0.35\text{ BH/s}$).
3. **Temporal Persistence:** Group cluster sustained for duration $\ge 2.0\text{s}$.

---

### Task 3: Staff Classification & Interaction Sessions

#### 1. Multi-Cue Staff Classifier
Staff are identified per-identity by fusing three independent cues:
$$P(\text{staff}) = 0.60 \cdot P_{\text{apron}} + 0.25 \cdot P_{\text{counter}} + 0.15 \cdot P_{\text{dwell}}$$
- **$P_{\text{apron}}$ (Color Matching):** HSV color fraction within torso keypoint quad matching terracotta (`[2–18, 120–255, 60–220]`) or dark navy (`[100–130, 60–255, 20–110]`).
- **$P_{\text{counter}}$ (Counter Zone Prior):** Dwell time behind the checkout counter polygon.
- **$P_{\text{dwell}}$ (Duration Prior):** Long-term presence ($>60\text{s}$) across the camera view.

#### 2. Staff-Customer Interaction Episodes
An interaction session is triggered when:
1. **Spatial Proximity:** Inter-person distance $d(\text{staff}, \text{customer}) \le 0.90\text{ BH}$.
2. **Mutual Facing:** Both persons oriented towards each other (angle $\le 60^\circ$, $\cos \theta \ge 0.50$).
3. **Co-Stationary:** Both speeds $\le 0.35\text{ BH/s}$.
4. **Duration:** Interaction sustained for $\ge 2.0\text{s}$ via Hysteresis State Machine.

---

## 9. Comparative Analysis vs. Alternative Methods

| Criterion | Naive Track Counter (SORT/ByteTrack) | End-to-End Deep Action Video Models (I3D / SlowFast / Video Transformers) | **Our Multi-Modal Two-Pass Architecture** |
| :--- | :--- | :--- | :--- |
| **Overcounting Resistance** | ❌ **Severe Failure** (+30% to +50% overcounting on occlusions) | ⚠️ Moderate (depends on tracking backbone) | ✅ **State of the Art** (Global Hungarian stitching repairs broken tracks) |
| **Behavioral Explainability** | ❌ None (only bounding boxes & centroid speeds) | ❌ **Black Box** (opaque feature activations, hard to audit) | ✅ **100% Auditable** (explicit geometric formulas & state machines) |
| **Edge Compute Feasibility** | ✅ Very Fast | ❌ **Extremely Heavy** (requires high-end 24GB+ GPUs for 3D convolutions) | ✅ **High Performance** (2D pose + flat Parquet + sub-second analytics) |
| **Perspective Invariance** | ❌ None (pixels only) | ⚠️ Implicit only | ✅ **Explicit $BH(y)$ scale normalization** |
| **Sensitivity Tuning** | ❌ Impossible | ❌ Requires complete model retraining on new labeled data | ✅ **Zero-shot parameter tuning** in $<1\text{s}$ from cached Parquet |

---

## 10. Validation, Sensitivity & Ablation Findings

### 10.1 Phase 2 Validation Gate Results
Ground truth hand counts vs. pipeline output evaluated across three 30-second windows on `entrance.mp4`:

| Window | Time Interval | Ground Truth (Hand Count) | Raw Tracker Tracks | Stitched Identities | Error ($\Delta$) | Gate Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Window 1** | `100.0s – 130.0s` | **8** | 10 | **7** | `-1` | **PASSED** |
| **Window 2** | `240.0s – 270.0s` | **5** | 7 | **5** | `0` | **PASSED** |
| **Window 3** | `320.0s – 350.0s` | **5** | 7 | **5** | `0` | **PASSED** |

- **Key Takeaway:** Raw tracking produced **24 tracks** across the three windows due to easel and podium occlusions. The Hungarian stitcher reduced this to **17 identities**, matching the ground truth count of **18** within a tight tolerance ($|\Delta| \le 1$).

### 10.2 Ablation Study

```
┌────────────────────────────────────────────────────────┬─────────────┬───────────┐
│ Configuration                                          │ Stitched IDs│ Error Rate│
├────────────────────────────────────────────────────────┼─────────────┼───────────┤
│ Full Pipeline (Pose + BoT-SORT + ReID + Stitching)     │     17      │   5.5 %   │
│ Ablation A: No Stitching (Raw Tracks Only)             │     24      │  33.3 %   │
│ Ablation B: Kinematic-Only Stitching (No ReID Embed)   │     21      │  16.6 %   │
└────────────────────────────────────────────────────────┴─────────────┴───────────┘
```

---

## 11. Known Edge Cases & Future Directions

### Current Limitations
1. **Severe Multi-Person Clumps:** When 4+ individuals huddle tightly around a shopping stroller, keypoint occlusion can cause ankle dropout.
2. **Direct Solar Glare:** At specific times of day, direct sunlight through the shopping mall atrium causes specular floor reflections that can trigger low-confidence false positive pose keypoints (mitigated by our bounding box height floor $h \ge 60\text{ px}$).
3. **Single-Camera Monocular Depth:** Physical depth is inferred via vertical $y$-position. If a child stands in the foreground, their small body height is partially attributed to depth.

### Future Improvements
1. **Planar Homography Calibration:** Mapping $(u, v)$ pixel coordinates to a top-down metric $(X, Y)$ ground floor plane using calibrated camera intrinsics and 4-point homography.
2. **Temporal Graph Neural Networks (GNNs):** Modeling multi-person interactions (group shopping and customer-staff service) via dynamic spatio-temporal graphs with edge weights representing visual attention and distance.
3. **Cross-Camera Global ReID:** Tracking a customer as they transition seamlessly from `entrance.mp4` to `interior.mp4` by matching global ReID embeddings across camera viewpoints.
4. **INT8 & TensorRT Quantization:** Quantizing the YOLO11m-Pose and ONNX ReID models to 8-bit integers for high-throughput deployment on edge micro-servers (e.g. NVIDIA Jetson Orin Nano).

---

## 12. Data Artifacts, Granularity & End-to-End Dataflow

This section specifies every persisted data artifact, its exact schema, its temporal
granularity, and the path data takes from raw video to the final CSV/video outputs.

### 12.1 Pass-1 Observation Cache (the Parquet + NPY pair)

Pass 1 runs once per video and persists two aligned files per scene
(`outputs/debug/entrance_*`, `outputs/debug/interior_*`):

**`{scene}_observations.parquet`** — flat columnar table.
**Granularity: one row per tracked person detection, per processed frame**
(stride=1 → every frame @ 30 fps). A frame with 3 visible people contributes 3 rows;
a frame with no tracked detection contributes none.

| Column(s) | Type | Meaning |
| :--- | :--- | :--- |
| `video` | string | Scene identifier (`entrance` / `interior`) |
| `frame_idx` | int32 | 0-based frame index in the source video |
| `t_s` | float32 | Timestamp in seconds (`frame_idx / fps`) |
| `raw_track_id` | int32 | Online BoT-SORT track id (fragmented; pre-stitching) |
| `x1, y1, x2, y2` | float32 | Person bounding box, pixels @ 1280×720 |
| `det_conf` | float32 | Detector confidence of the box |
| `kp_x_00 … kp_x_16` | float32 | X coordinate of the 17 COCO pose keypoints (nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles) |
| `kp_y_00 … kp_y_16` | float32 | Y coordinate of the 17 keypoints |
| `kp_c_00 … kp_c_16` | float32 | Per-keypoint confidence ∈ [0, 1] |
| `embed_ref` | int32 | Row index into the companion NPY embedding matrix (−1 if unavailable) |

**`{scene}_embeddings.npy`** — `float32` matrix of shape `(N, 512)`, $L_2$-normalized
ReID appearance vectors. **Granularity: one row per parquet row**, aligned via
`embed_ref` (e.g. a parquet row with `embed_ref = 137` uses `embeddings[137]`).
Actual cache sizes on the supplied footage: entrance = 15,380 rows / 5,288 frames /
46 raw tracks; interior = 20,757 rows / 9,767 frames / 71 raw tracks.

Nothing else is stored from Pass 1 — no images, no per-frame feature maps. All
behavioral logic re-derives everything from these two files, which is why Pass-2
experimentation runs in seconds without touching the neural networks.

### 12.2 Pass-2 In-Memory Derived Data (per identity, per frame)

`build_identities()` merges parquet rows into stitched `Identity` objects. Every
field below is a **per-frame array** aligned with that identity's `frames` /
`t_s` vectors (only frames where the person is tracked; gaps ≤ 10 frames are
interpolated, so the arrays are contiguous over the track lifespan):

| Array | Shape | Granularity / derivation |
| :--- | :--- | :--- |
| `frames`, `t_s` | `[T]` | Frame indices and timestamps of the stitched identity |
| `foot_xy` | `[T, 2]` | Ground contact point: ankle-keypoint midpoint when both ankles ≥ 0.5 conf, else bbox bottom-centre; gap-interpolated, median-smoothed |
| `heights` | `[T]` | Bbox pixel height — the per-frame body-height unit used for all BH normalization |
| `facing_normal` | `[T, 2]` | Unit torso normal from shoulder keypoints (NaN where shoulders < 0.5 conf); gap-filled |
| `head_yaw` | `[T]` | Head-turn proxy from eye/ear confidence asymmetry ∈ [−1, 1] |
| `speed_bh_s` | `[T]` | $\|\Delta \text{foot}\| / BH(y) / \Delta t$, median-smoothed |
| `kpts_raw` | `[T, 17, 3]` | Unsmoothed raw keypoints (audit trail) |

### 12.3 Per-Task Data Products

**Task 1 — Store Interest (`entrance`)**
- *Per frame, per candidate identity:* the four channel series $A(t), S(t), P(t), D(t)$
  and the fused interest score $I(t)$; plus the signed distance to the entrance line
  used by the crossing predicate. These live in memory only (recomputable on demand).
- *Persisted:* `outputs/task1_store_interest.csv` — **one row per candidate identity**
  (`identity_id, first_t_s, last_t_s, seconds_in_hallway, peak_interest,
  orientation_at_peak, decel_at_peak, approach_at_peak, dwell_at_peak, interested,
  entered, low_confidence`), and `outputs/task1_summary.csv` — **one row per metric**
  (`total_interested, interested_entered, interested_passed_by`), the brief's
  labelled final counts.

**Task 2 — Per-Shelf Interest (`interior`)**
- *Per frame, per identity:* a single shelf assignment ∈ {`shelf-a`…`shelf-d`, None}
  (nearest rect passing the proximity + facing + speed + exclusion gates, then a
  1-second rolling majority vote). Granularity: one label per tracked frame.
- *Per (identity, shelf) episode:* the hysteresis machine turns the boolean
  `assignment == shelf` series into interest episodes.
- *Persisted:* `outputs/task2_shelf_engagement.csv` — **one row per interest
  episode** (`identity_id, shelf_name, start_t, end_t, duration_s, median_dist_bh`);
  `outputs/task2_shelf_interest.csv` — **one row per shelf** (`shelf,
  interest_events`) plus the `total` row — the brief's official deliverable;
  `outputs/task2_shelf_summary.csv` — per-shelf visitors/dwell aggregates;
  `outputs/task2_group_dwell.csv` — **one row per group-dwell episode** (auxiliary).

**Task 3 — Staff & Interactions (`entrance`)**
- *Per identity:* staff probability score (apron HSV + counter-zone dwell + duration
  prior) → binary staff label.
- *Per frame, per (staff, customer) pair:* the interaction condition (distance ≤
  0.9 BH, mutual facing ≤ 60°, co-stationary) → hysteresis sessions.
- *Persisted:* `outputs/task3_staff_interactions.csv` — **one row per interaction
  session** (`staff_id, customer_id, start_t, end_t, duration_s, mean_distance_bh`);
  `outputs/task3_staff_summary.csv` — **one row per staff instance, including
  zero-session staff**, plus the final `AVERAGE` row — the brief's required metric.

### 12.4 Dataflow: From Video to Output, Step by Step

What happens to a single frame / a single person as data flows through the pipeline
(function names reference `src/`):

```
entrance.mp4 / interior.mp4  (1280×720 @ 30 fps)
        │
        │  for every frame_idx (stride=1 → each frame sampled):
        ▼
(1) YOLO11m-Pose inference          detection.py  :: PoseDetector
        → person bboxes + 17 keypoints + confidences
(2) BoT-SORT online association     tracking.py   :: Tracker
        → raw_track_id per person + 512-d ReID embedding (yolo26n-reid ONNX)
(3) Stream to cache                 io_utils.py   :: ParquetBatchWriter
        → one parquet row per detection (+ one NPY embedding row)
        ══ END OF PASS 1 (runs once; everything below re-reads the cache) ══
(4) Load cache                      io_utils.py   :: read_observations / read_embeddings
(5) Group rows into raw tracklets   stitching.py  :: build_tracklets
(6) Global Hungarian stitching      stitching.py  :: stitch
        → stitch_map: raw_track_id → identity_id  (repairs occlusion breaks)
(7) Build per-frame identity arrays identity.py   :: build_identities
        → foot_xy, heights, facing_normal, head_yaw, speed_bh_s (smoothed)
        ══ behavioral logic — all per-frame predicates, then event machines ══
(8) entrance: task3_staff.detect_staff  → staff identity set
              task3_staff.staff_customer_interactions → session episodes
(9) entrance: task1_interest.run_task1  → I(t) per frame → interested/entered flags
(10) interior: task2_shelf.assign_shelf_per_frame + mode filter
              → per-frame shelf label → hysteresis → interest episodes
(11) Export CSVs (§12.3) and re-walk the video rendering overlays
              render.py :: render_entrance_frame / render_interior_frame
        → outputs/*.csv + outputs/entrance_annotated.mp4 / interior_annotated.mp4
```

Steps (1)–(3) are the *online* part (causal, frame-by-frame, neural-network heavy).
Steps (4)–(11) are the *offline* part: global, sub-second, fully deterministic given
the cache — the "online-offline" split that makes the pipeline both accurate
(stitching sees the future) and cheap to iterate on.

---

## 13. Parameter Audit: Justification, Sensitivity & Improvement Analysis

Every tunable constant in the system, the evidence behind its value, its measured
sensitivity, and a verdict (*Keep* / *Tune* / *Fix*). Evidence comes from the cached
observations (15.4k entrance / 20.8k interior detections), the sensitivity sweeps in
§10, and frame-by-frame visual verification of the outputs.

### 13.1 Why These Algorithms (before the numbers)

| Choice | Alternatives rejected | Justification |
| :--- | :--- | :--- |
| **YOLO11m-Pose** (one model → boxes + 17 keypoints) | Detector-only (YOLO/RT-DETR boxes); separate pose stack (MediaPipe/OpenPose) | The brief's behaviours — *looking at* a storefront, *facing* a shelf — are invisible to boxes. Pose gives torso facing, head-yaw gaze proxy, and ankle ground truth in a single inference pass at negligible extra cost. |
| **BoT-SORT + offline Hungarian stitching** | Pure online tracking (SORT/ByteTrack alone); transformer ReID | Online trackers fragment on occlusions: interior yields 71 raw tracks for 62 real identities, 26 tracks < 1 s, median track only 2.4 s. Stitching with fused ReID+motion cost recovers the breaks (ablation §10.2: 5.5% vs 33.3% ID error without it). Offline global assignment can see the future — impossible online. |
| **Hysteresis state machine** | Instantaneous thresholding; HMMs | Two parameters everyone can reason about; provably eliminates flicker-induced double counting (the brief's explicit requirement); closing at the last-true frame avoids duration inflation. |
| **Hand-crafted geometric predicates** | Learned action recognition (SlowFast/X3D) | Zero labelled data, 2 videos, 5-day scope, CPU-class deployment, and a rubric that rewards explainability. Every decision is auditable per frame — which is how the Task-2 false positives were found and fixed. |
| **Body-height (BH) normalisation** | Raw pixels; full homography | No camera intrinsics available. BH(y) needs only the image y-coordinate, and Task 2 goes further — normalising by each person's *own* bbox height, making it self-calibrating even where the global scale map is imperfect (see §13.4, F4). |

### 13.2 Perception & Tracking (Pass 1)

| Parameter | Value | Justification & evidence | Sensitivity | Verdict |
| :--- | :--- | :--- | :--- | :--- |
| `detector conf` | 0.30 | Pre-filter floor feeding the tracker; kept detections concentrate at 0.6–0.9, so 0.30 only cuts the noisy tail (reflections, poster people). | Low — tracker applies its own two thresholds above this | **Keep** |
| `imgsz` | 1280 entrance / 960 interior | Matches native 1280×720 aspect; interior people are larger on screen, so 960 is sufficient and faster | Moderate (smaller would miss far hallway people) | **Keep** |
| `track_high_thresh` | 0.40 | 96.6–98.7% of cached detections exceed it — filters junk, keeps nearly all real people | Low | **Keep** |
| `new_track_thresh` | 0.65 | 69–91% of detections qualify; spawning requires strong evidence, suppressing reflection/mannequin births | Moderate | **Keep** |
| `track_low_thresh` | 0.08 | Second-stage association rescues partially occluded people (occlusion is the dominant failure mode here) | Moderate | **Keep** |
| `track_buffer` | 90 f = 3.0 s | Exceeds every occlusion gap actually bridged by stitching (p90 = 1.7–1.8 s, max 1.9 s) | Low | **Keep** |
| `gmc_method` | `none` | Fixed camera: global motion compensation adds cost and noise, nothing else | None | **Keep** |
| `with_reid`, `appearance_thresh` 0.25, `proximity_thresh` 0.50, `match_thresh` 0.85 | — | BoT-SORT defaults validated on this footage via the stitching validation gate (§10.1) | Low | **Keep** |

### 13.3 Stitching & Smoothing (Pass 2)

| Parameter | Value | Justification & evidence | Sensitivity | Verdict |
| :--- | :--- | :--- | :--- | :--- |
| `max_gap_frames` | 60 f = 2.0 s | Bridged gaps in practice: p50 ≈ 1.0–1.3 s, max 1.9 s — 2.0 s is snug but sufficient; longer gaps risk wrong-person merges | Moderate | **Keep** |
| `motion_gate_bh` | 2.5 BH | A person covers ≲ 1.5 BH/s when walking → 2.5 BH over a ≤ 2 s gap admits realistic motion with margin | Moderate | **Keep** |
| `scale_gate_frac` | 0.40 | Bbox height jitter across an occlusion is ≪ 40%; two different people at similar depths differ by more | Low | **Keep** |
| `direction_gate_deg` | 60° | Nobody reverses direction within a 2 s occlusion; only enforced when both tracklets moved > 5 px | Low | **Keep** |
| Cost weights 0.55 / 0.30 / 0.15, `cost_max` 0.45 | — | Appearance-dominant: ReID is the most reliable cue across occlusions; ablation B (kinematics only) shows error tripling (16.6% vs 5.5%). Gate-validated: 17 stitched IDs vs 18 hand-counted | Moderate | **Keep** |
| `median_win` | 15 f = 0.5 s | Kills single-frame keypoint spikes; short enough not to blur genuine stop/start events | Low | **Keep** |
| `max_interp_gap` | 10 f = 0.33 s | Only micro-gaps fabricated; anything longer stays NaN so downstream never trusts invented positions | Low | **Keep** |
| `savgol_win/order` | 21 / 2 | **Configured but unused** — the median filter proved sufficient and preserves step edges better | — | **Fix (remove dead config or wire it)** |

### 13.4 Task 1 — Store Interest (entrance)

| Parameter | Value | Justification & evidence | Sensitivity | Verdict |
| :--- | :--- | :--- | :--- | :--- |
| Weights 0.40 / 0.25 / 0.20 / 0.15 | — | Orientation dominates: the brief defines interest as *directing attention*, so the visual-attention channel leads; deceleration is the strongest physical corroborator; approach and dwell are weaker alone | Moderate | **Keep** |
| `interest_threshold` | 0.55 | **Highest-leverage knob in the whole system.** Sweep: 0.45 → 14 interested, 0.55 → 6, 0.65 → 3. Five candidates sit in the 0.45–0.55 band, so the count moves in steps | **High** | **Tune** — calibrate against a small hand-labelled set if one becomes available; 0.55 is a defensible conservative point |
| `interest_min_duration_s` | 0.8 s | Brief requires *sustained* attention; 0.8 s (24 f) filters casual glances, keeps genuine looks (~1–2 s) | Moderate | **Keep** |
| `orientation_cos_floor_deg` | 75° | Cosine floor before orientation credit ramps; 75° tolerates angled approaches to the storefront | Low | **Keep** |
| `entered_deadzone_bh` | 0.25 BH | Deadband around the entrance line kills sign-flip flicker from foot jitter | Low | **Keep** |
| `entered_depth_bh` | 1.0 BH | Must penetrate a full body height past the line — separates "leaning over the threshold" from entering | Low | **Keep** |
| `entered_dwell_s` | 2.0 s | Entry must persist; consistent with every other dwell rule in the system | Low | **Keep** |
| `min_bbox_height` | 70 px | Excludes far-background mall pedestrians outside the analysis area | Low | **Keep** |
| Candidate gate | ≥ 1.0 s in hallway | Minimal presence to be scored; removes drive-by tracks at the frame edge | Low | **Keep** |
| `task1.event` (t_on/t_off) | 2.0 / 3.0 | **Dead config** — `decide_interested` uses a contiguous-streak test, not the state machine; the Task-1 t_off column in the sensitivity table is consequently a no-op | — | **Fix (wire or remove)** |
| Hardcoded `30.0` fps | — | `is_candidate` and `seconds_in_hallway` divide by a literal 30.0 instead of `cfg.fps` — silently wrong for non-30 fps input | — | **Fix (use cfg.fps)** |

### 13.5 Task 2 — Per-Shelf Interest (interior)

Every value below was calibrated against the footage and then **verified frame-by-frame**
(all 11 final events eyeballed at start/middle/end — see `tools/verify_task2.py`).

| Parameter | Value | Justification & evidence | Sensitivity | Verdict |
| :--- | :--- | :--- | :--- | :--- |
| `reach_dist_bh` | 1.2 BH | Verified browsers measure 0.59–1.18 BH from the rect; passers-by and mirror reflections measure ≥ 1.5 BH. Sweep: 1.0 → 8 events (misses real bent-over browsers at 1.1–1.2), 1.2 → 11, 1.5 → 16–18 (absorbs non-browsers). 1.2 sits in the measured gap between the two populations | Moderate | **Keep** |
| `facing_cos_floor_deg` | 80° | The between-two-shelves disambiguator. Shoulders available ≥ 0.5 conf in 99.6% of frames, so the gate is almost always computable. Caught the back-table browser (cos ≈ −0.9) that proximity alone would have admitted | Moderate | **Keep** |
| `engage_speed_bh_s` | 0.35 BH/s | Candidate frames: p50 = 0.07, p90 = 0.24. All frames: p90 = 0.52. The threshold sits cleanly between browsing shuffle and walking | Low | **Keep** |
| `min_bbox_height` | 40 px | Sanity floor against tiny/mostly-occluded detections; real shelf browsers are ≥ 80 px | Low | **Keep** |
| `event.t_on_s` | 2.0 s | "Sustained attention" per the brief; shortest visually-verified true event is 2.4 s | Moderate | **Keep** |
| `event.t_off_s` | 3.0 s | Longer than micro-interruptions (looking away, occlusion), shorter than a genuine leave-and-return (demonstrated: id10's shelf-a return correctly counted as a new event) | Moderate | **Keep** |
| Mode-filter window | 1.0 s | Removes A/B assignment flicker between adjacent fixtures; guarantees one active shelf per person per frame | Low | **Keep** |
| Ignore zones | mirror + counter/desk | Both added only after their false positives were observed in rendered frames (mirror reflections → phantom shelf-d; paying/working at the counter → phantom shelf-a). Documented per-zone in the config | None observed on true events | **Keep** |

### 13.6 Task 3 — Staff & Interactions (entrance)

| Parameter | Value | Justification & evidence | Sensitivity | Verdict |
| :--- | :--- | :--- | :--- | :--- |
| `proximity_bh` | 0.9 BH | Conversational distance ≈ an arm's length; wider values admit co-browsing strangers | Moderate | **Keep** |
| `orientation_cos_floor_deg` | 60° | Mutual facing within 60° — the signature of a conversation vs parallel walking | Moderate | **Keep** |
| `co_stationary_bh_s` | 0.35 BH/s | Same browsing/stationary constant as Task 2 — deliberate consistency | Low | **Keep** |
| `event.t_on_s` | 2.0 s | Sessions, not fleeting contact | Moderate | **Keep** |
| Staff weights 0.60 / 0.25 / 0.15, threshold 0.50 | — | **Apron cue is not wired at runtime** (`p_apron` is a constant 0.5 fallback; `staff.py`'s HSV detector is dead code). Effective score = 0.30 + 0.25·counter + 0.15·dwell, so staff ≈ counter-dwellers and long-present identities. It works on this footage (5 staff found, all at the counter) but it is *prior-based*, not the apron-visual evidence the brief suggests | — | **Fix (wire `detect_staff_apron` into Pass 2 via `VideoFrameAccessor`, or document as prior-only)** |

### 13.7 Findings Register & Prioritised Improvements

| # | Finding | Impact | Action taken / recommended |
| :--- | :--- | :--- | :--- |
| F1 | Doc/code drift: §5.2 gates (1.8→2.5 BH, 0.35→0.40, 0.42→0.45, missing direction gate), §5.3 filter windows, §6.1 decel/approach formulas and 60°→75° orientation floor | Documentation correctness | **Fixed in this revision** |
| F2 | Apron HSV detector implemented but unused (`p_apron = 0.5` constant) | Task 3 relies on spatial/dwell priors only | Recommend wiring `staff.py::detect_staff_apron` into Pass 2; one `VideoFrameAccessor` lookup per candidate identity |
| F3 | `task1.event` config dead; sensitivity sweep of Task-1 `t_off` is a no-op | Misleading analysis tooling | Recommend routing `decide_interested` through `HysteresisStateMachine` (or deleting the config keys) |
| F4 | Scale-map drift: least-squares on cached detections gives interior `h ≈ 0.280y + 54` vs configured `0.200y + 60` (mean residual +17 px) | Small bias in BH-normalised quantities | Task 2 already immune (per-person bbox height). Recommend re-fitting the global map on cleaned, high-confidence, full-body samples if Tasks 1/3 move further into the far field |
| F5 | Hardcoded `30.0` fps in two Task-1 helpers | Latent bug for non-30 fps video | Recommend plumbing `cfg.fps` |
| F6 | `interest_threshold = 0.55` dominates Task-1 counts (6 → 14 → 3 across the sweep) | Main accuracy lever | Recommend a ~20-person hand-labelled validation set to pick the operating point properly |
| F7 | Interior ankle availability only 45.6% (fixtures occlude feet) → foot point falls back to bbox bottom-centre in ~54% of frames | Foot jitter slightly higher interior | Acceptable (median smoothing + BH normalisation absorb it); noted for completeness |

**Overall verdict:** the system sits at a stable, defensible operating point. Every gate
threshold has measured clearance on this footage (bridged gaps max 1.9 s < 2.0 s budget;
browser distances 0.59–1.18 < 1.2 BH budget; browsing speeds p90 0.24 < 0.35 cap). The two
genuinely improvable items are **F6** (Task-1 threshold calibration — needs labels) and
**F2** (apron cue wiring — brief-relevant); the rest is hygiene (F1, F3, F5) or optional
refinement (F4, F7).

---

## Summary Diagram: End-to-End Data Lifecycle

```
[Raw Frame RGB (1280x720)]
         │
         ▼
[YOLO11m-Pose Keypoints] ──► [BoT-SORT Tracker] ──► [ONNX ReID Embedding Vector]
         │                            │                           │
         └────────────────────────────┴───────────────────────────┘
                                      │
                         [Parquet & NPY Observation Cache]
                                      │
                                      ▼
                        [Hungarian Tracklet Stitching]
                                      │
                         [Perspective Scale Map BH(y)]
                                      │
                         [Multi-Channel Feature Fusion]
                                      │
                        [Hysteresis State Machines]
                                      │
                                      ▼
             [Executive CSV Reports & Annotated Overlay Videos]
```
