# DAS Signal Processing — Submarine Cable Detection

**Reference:** Lin et al. (2025), *Journal of Lightwave Technology*  
**Dataset:** PubDAS — Zhoushan Archipelago experiment  
**Source:** 50 kJ plasma pulse, primary frequency 100–400 Hz

---

## 1. Business Understanding

### What is DAS?

Distributed Acoustic Sensing (DAS) converts a standard optical fibre into a dense array of
acoustic sensors by measuring Rayleigh backscatter along the cable.  Each channel records the
axial strain rate at a fixed position along the fibre, giving a 2-D space–time dataset.

---

## 1. Business Understanding

### 1.1 Water Pipeline Monitoring

Water utilities operate extensive buried pipeline networks that are difficult and costly
to inspect. Undetected leaks cause Non-Revenue Water (NRW) losses that can exceed 20–30%
of total supply in ageing networks, while unauthorised intrusions and third-party damage
pose safety and service-continuity risks.

**Objective:** Detect and localise leaks and intrusions in real time, with sufficient
spatial precision (<10 m) to guide targeted repair without unnecessary excavation.

**Why DAS:** A single fibre installed inside or alongside the pipe acts as a continuous
array of thousands of acoustic sensors. Leak orifices generate broadband turbulent noise
(~100 Hz – several kHz); intrusion events produce characteristic mechanical impacts.
DAS captures both continuously, without on-site intervention.

**Key challenges:**
- Distinguishing leak signatures from pump harmonics, valve transients, and
  third-party traffic noise in a shared frequency band.
- Achieving 5–10 m localisation accuracy over pipeline runs of tens of kilometres,
  where signal attenuation and variable pipe material affect propagation speed.
- Maintaining reliable detection under low-flow or intermittent-pressure conditions
  where leak acoustic output is weak.

---

### 1.2 Submarine Cable Monitoring (Offshore Wind)

Offshore wind farms depend on subsea export and inter-array power cables to deliver
electricity to shore. Cable failure causes prolonged and expensive downtime; repair
campaigns require specialised vessels and can take weeks. Cables face burial exposure,
mechanical fatigue at Cable Protection Systems (CPS), third-party anchor strikes, and
electrical faults — all difficult to detect by periodic vessel-based inspection.

**Objective:** Provide continuous, real-time health monitoring of subsea power cables —
detecting exposure, CPS abrasion, impact events, and electrical anomalies — to enable
proactive O&M decisions and reduce unplanned downtime.

**Why DAS:** The optical fibre already embedded in the cable's structure is repurposed
as a dense acoustic and strain sensor array spanning the full cable length (up to 70 km
per interrogator). No offshore vessels or subsea intervention are required for monitoring.

**Key challenges:**
- Separating target signals (CPS scraping, anchor impact, cable movement) from
  high-amplitude ocean ambient noise, shipping traffic, and biological sources
  such as whale vocalisations.
- Distinguishing acoustic wave arrivals (propagating through the water column at
  ~1480 m/s) from guided elastic waves travelling along the cable structure at
  higher apparent velocities — critical for correct source localisation.
- Processing continuously generated, high-volume spatio-temporal data streams
  efficiently enough for real-time alerting on edge or cloud infrastructure.

## 2. Data Understanding

### Source Publication

> J. Lin, Q. Wang, W. Zhang, J. Shu, L. Zhang and Q. Tang,
> "Estimation of Submarine Cable Location Using Optical-Fiber Distributed Acoustic Sensing
> Combined With Ship-Borne Sound Sources,"
> *Journal of Lightwave Technology*, vol. 43, no. 18, pp. 8917–8926, 15 Sept. 2025.
> doi: [10.1109/JLT.2025.3588069](https://doi.org/10.1109/JLT.2025.3588069)

The dataset is publicly available.


### Dataset Parameters

| Parameter | Value |
|-----------|-------|
| Sampling rate `fs` | 1000 Hz |
| Channel spacing `dx` | 4.09 m |
| Gauge length | 10 m |
| Aperture used | 1.0 – 3.0 km |
| Number of channels | 490 |
| Record length | 4.95 s |
| Bandpass applied | 1 – 148 Hz |
| Water sound speed `c_water` | 1480 m/s |
| Number of events | 10 (event_id 1–10) |

### File Format

Raw data are stored as per-channel binary **SAC** files (`Channel_XXXX.sac`).  
Each file contains a 70-float + 40-int header followed by 32-bit float waveform samples.

### Visualisation

The raw vs bandpass-filtered waterfall (time × distance) shows the hyperbolic arrival
pattern of the plasma-pulse wavefront, Scholte interface waves (slow, dispersive), and
persistent background noise.

![Raw vs filtered waterfall](plots/raw_vs_filtered.png)

---

## 3. Feature Engineering — Single-Channel Features

Seven features are extracted in a **sliding window** (window = 0.5 s, hop = 0.25 s) applied
to the bandpass-filtered strain-rate time series of the peak channel.

| Feature | Domain | Description |
|---------|--------|-------------|
| **RMS** | Time | Root-mean-square amplitude — overall energy |
| **Peak** | Time | Maximum absolute amplitude |
| **Crest Factor** | Time | Peak / RMS — impulsiveness indicator |
| **Kurtosis** | Time | Fourth statistical moment — transient sharpness |
| **Band energy Lo** | Frequency | Welch PSD mean, 1–50 Hz |
| **Band energy Mid** | Frequency | Welch PSD mean, 50–100 Hz |
| **Band energy Hi** | Frequency | Welch PSD mean, 100–148 Hz |
| **WPD sub-bands** | Frequency | Wavelet packet decomposition (db4, level 3, 62 Hz / band) |

A combined **anomaly score** is formed by z-scoring all 7 features and computing their
L2 norm: `score = ‖z‖₂ = √(Σ zᵢ²)`.  Windows exceeding a threshold (default 2.5) define
the anomalous segment `t_window_anomaly` used in all subsequent analyses.

![Single-channel features](plots/features_ch539.png)

---

## 4. Event Detection

### Current Method — RMS Sliding Window + L2 Anomaly Score

1. A 0.5 s sliding window scans the full record; the window with maximum RMS energy defines
   the event time.
2. The 7-feature L2 z-score (Section 3) identifies the precise anomalous segment and
   outputs `t_window_anomaly`.

![Event detection](plots/peak_amplitude.png)

![Anomaly detection score](plots/anomaly_detection.png)


### Suggested Extension — Machine Learning Anomaly Detection

The 7-feature vector (or the WPD sub-band energies) forms a compact representation suitable
for supervised or unsupervised ML:

- **Unsupervised:** Isolation Forest / Autoencoder on feature sequences
- **Supervised:** Random Forest / XGBoost trained on labelled event windows
- **Sequence models:** LSTM / Transformer on the sliding-window feature matrix to capture
  temporal evolution of the anomaly

The `t_window_anomaly` output of the current method can serve as a weak label for
bootstrapping supervised approaches.


---

## 5. Source Localisation

### f-k Separation

A 2-D FFT is applied over the full (time × space) dataset.  A cosine-tapered mask
separates:
- **Acoustic component** — apparent velocity ≥ c_water (1480 m/s)
- **Scholte component** — apparent velocity 200–1480 m/s


![f-k spectrum](plots/fk_spectrum.png)

![Acoustic and Scholte filtered waterfalls](plots/fk_strain.png)

![Single-channel f-k decomposition](plots/fk_single_ch539.png)

![STFT after f-k separation](plots/stft_fk_ch539.png)

### Hyperbola Fit

A point source at depth `z0` below the cable at along-cable position `x0` produces a
hyperbolic travel-time curve:

```
t(x) = t0 + √((x − x0)² + z0²) / c_water
```

**Procedure:**

1. The peak channel (highest amplitude in the anomalous segment) provides a template
   waveform.
2. Cross-correlation (CC) of every channel against the template yields per-channel
   arrival times.  Only picks with normalised CC ≥ 0.8 are retained.
3. `scipy.optimize.curve_fit` fits the three free parameters `(x0, z0, t0)` to the
   CC picks.  The initial estimate for `z0` is derived automatically from pick geometry:
   `z0 ≈ median(Δx² / (2 · c · Δt))`.

**Output:** source position `(x0, z0)` and excitation time `t0`.

**Result for Event 1:**

| Parameter | Value | Description |
|-----------|-------|-------------|
| `x0` | 2.08 km | Source position along cable |
| `z0` | 230 m | Source-to-cable perpendicular distance |
| `t0` | 0.49 s | Source excitation time |

![Hyperbola fit — source localisation](plots/hyperbola_fit.png)

---

## 6. Beamforming

### Short-Time Plane-Wave Beamforming

Applied to the acoustic-filtered component within the anomalous segment.  For each
short time window (50 ms, hop 10 ms) and each candidate slowness `p`:

```
B(p, t) = |Σₙ Xₙ(f) · exp(−j 2π f p xₙ)|²
```

The **slowness–time image** reveals:
- *When* energy arrives (time axis)
- *From which direction* (slowness axis, converted to incidence angle θ = arcsin(p · c))
- Multiple simultaneous arrivals appear as distinct blobs

![Slowness–time beamforming image](plots/beamforming_anomaly.png)

### Focused DAS

Using the source position `(x0, z0)` from the hyperbola fit, spherical-wave delays are
computed for each channel:

```
τₙ = √((xₙ − x0)² + z0²) / c_water
```

Frequency-domain delay-and-sum with CC-quality-weighted channel selection (N = 22 channels)
produces a beamformed time-domain signal with improved coherence.

> **Note on limitations:** For near-field point sources observed by DAS, the beamformed SNR
> gain is limited by spatially correlated noise and the cos²(θ) DAS directional sensitivity.
> The peak single channel may outperform the array average in this geometry.

![Focused DAS — time domain and Welch PSD](plots/focused_das.png)

The final output combines:

- **Time-domain:** focused DAS waveform showing the pulse shape
- **Frequency-domain:** Welch PSD of the focused signal

The PSD shows a dominant peak near **100 Hz**, consistent with the plasma-pulse source
spectrum reported in the paper.
---


## 7. Classification of Acoustic Events

DAS monitoring serves two distinct operational contexts — submarine cable surveillance
and water pipeline monitoring — each with different signal and noise characteristics
that shape the choice of classification approach.

---

### 7.1 Signal and Noise Characteristics by Application

#### Submarine Cable Monitoring

Target events arrive as **transient, broadband impulses** propagating through the water
column at ~1480 m/s, producing a hyperbolic travel-time signature across the array.
Distinct event types include anchor strikes (high-energy, short duration), CPS abrasion
(repetitive narrow-band scraping correlated with tidal/current periodicity), cable
exposure (gradual low-frequency strain increase), and electrical partial discharge
(high-frequency burst, >1 kHz).

Background noise is **spatially and temporally non-stationary**: shipping traffic
contributes tonal propeller harmonics (5–50 Hz) with characteristic linear moveout in
the f-k domain; ocean swell generates low-frequency (< 1 Hz) correlated noise across
all channels; biological sources such as whale vocalisations produce structured
narrowband calls. Crucially, guided elastic waves travelling along the cable structure
at speeds well above c_water appear as steep-slope components in f-k space and must be
separated from water-borne arrivals before classification.

#### Water Pipeline Monitoring

Target signals are **continuous, stationary broadband noise** rather than discrete
impulses. Leak orifices generate turbulent flow noise spanning ~100 Hz to several kHz,
with spectral shape dependent on pressure differential and orifice geometry. Intrusion
events (drilling, excavation) produce impulsive mechanical transients with dominant
energy below 500 Hz.

Background interference is **deterministic and periodic**: pump harmonics appear as
narrow spectral lines at fundamental and overtone frequencies; valve actuation produces
short-lived broadband bursts; water hammer generates decaying oscillatory transients.
Unlike the subsea case, signal propagation is confined to the pipe wall (elastic guided
waves at 1000–5000 m/s depending on material), so f-k separation by water-column
velocity is not applicable — discrimination relies on spectral and temporal features
rather than apparent velocity.

---

### 7.2 Feature-Based Classification Framework

The 7-feature vector (Section 3) together with source-localisation outputs `(x0, z0)`
and beamforming slowness estimates form a **classification feature vector**:

| Feature group | Examples | Most relevant for |
|---|---|---|
| Energy features | RMS, Peak, band energies | Both |
| Impulsiveness | Crest Factor, Kurtosis | Cable impacts, pipeline intrusion |
| Spectral shape | WPD sub-band ratios, PSD peak frequency | Leak vs pump discrimination |
| Temporal stationarity | Short-time RMS variance, autocorrelation decay | Leak (stationary) vs impact (transient) |
| Spatial | Source distance `z0`, along-cable position `x0` | Cable localisation |
| Directional | Dominant slowness, f-k velocity | Cable event typing |

---

### 7.3 Recommended ML Approaches

#### For submarine cable events (transient, multi-class)

Events are sparse and labelled examples are scarce, favouring methods robust to small
datasets:

- **Random Forest / Gradient Boosting** on the hand-crafted feature vector: interpretable,
  resistant to overfitting on small labelled sets, and directly auditable by domain
  experts — important for operational deployment where false alarms are costly.
- **1-D CNN on the f-k-filtered spectrogram**: captures the hyperbolic moveout pattern
  that encodes both event type and source geometry; suitable when a moderate labelled
  dataset is available across multiple deployments.
- **One-class SVM or Isolation Forest** for anomaly detection in the absence of labelled
  event examples: trained on background-only data and triggered when a new observation
  falls outside the learned noise manifold.

#### For pipeline leak detection (continuous signal, binary or multi-class)

The continuous nature of leak signals favours methods that exploit temporal context:

- **Sliding-window feature classification (Random Forest / XGBoost)**: the 7-feature
  vector computed in overlapping windows captures the sustained elevated RMS and spectral
  shift associated with active leaks; temporal smoothing of classifier output reduces
  false positives from transient pump noise.
- **Autoencoder on short-time spectrograms**: trained on normal pipeline noise; elevated
  reconstruction error flags anomalous acoustic activity without requiring labelled leak
  examples — practical given the rarity of real leak events in training data.
- **Physics-informed feature engineering**: expressing band energy ratios relative to
  known pump harmonic frequencies explicitly encodes domain knowledge and reduces the
  burden on the classifier to learn these invariances from data.

#### Shared consideration: interpretability over complexity

In both applications the cost of a missed detection (cable failure, pipe burst) is high,
and operators need to trust and audit alerts. A well-engineered feature vector fed to a
gradient-boosted tree — with SHAP values for per-alert explanation — is therefore
preferable to a black-box deep model unless labelled data is abundant. Deep approaches
are best reserved for sub-tasks where their advantage is clear, such as spectrogram-based
event typing in the submarine cable case.



---

## Repository Structure

```
distributed-acoustic-sensing/
├── DAS_Signal_Processing.ipynb   # Main analysis notebook
├── utils/
│   └── plot_style.py             # Colour palettes and figure sizes
├── PubDAS_data/                  # Raw SAC files (not included)
│   └── DAS data for submarine cable detection/1/
├── plots/                        # Saved figures (auto-generated)
└── README.md
```

---

## Requirements

See `requirements.txt`.  Install with:

```bash
pip install -r requirements.txt
# Optional wavelet support:
conda install -c conda-forge pywavelets
```

---
