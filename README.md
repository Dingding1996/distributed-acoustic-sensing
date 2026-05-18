# DAS Signal Processing — Submarine Cable Detection

**Reference:** Lin et al. (2025), *Journal of Lightwave Technology*  
**Dataset:** PubDAS — Zhoushan Archipelago experiment  
**Source:** 50 kJ plasma pulse, primary frequency 100–400 Hz

---

## 1. Business Understanding

> *(To be completed — describe the operational context, stakeholders, and objectives.)*

---

## 2. Data Understanding

### Source Publication

> J. Lin, Q. Wang, W. Zhang, J. Shu, L. Zhang and Q. Tang,
> "Estimation of Submarine Cable Location Using Optical-Fiber Distributed Acoustic Sensing
> Combined With Ship-Borne Sound Sources,"
> *Journal of Lightwave Technology*, vol. 43, no. 18, pp. 8917–8926, 15 Sept. 2025.
> doi: [10.1109/JLT.2025.3588069](https://doi.org/10.1109/JLT.2025.3588069)

The dataset is publicly available as part of the **PubDAS** repository.

### What is DAS?

Distributed Acoustic Sensing (DAS) converts a standard optical fibre into a dense array of
acoustic sensors by measuring Rayleigh backscatter along the cable.  Each channel records the
axial strain rate at a fixed position along the fibre, giving a 2-D space–time dataset.

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

### Suggested Extension — Machine Learning Anomaly Detection

The 7-feature vector (or the WPD sub-band energies) forms a compact representation suitable
for supervised or unsupervised ML:

- **Unsupervised:** Isolation Forest, Local Outlier Factor, Autoencoder on feature sequences
- **Supervised:** Random Forest / XGBoost trained on labelled event windows
- **Sequence models:** LSTM / Transformer on the sliding-window feature matrix to capture
  temporal evolution of the anomaly

The `t_window_anomaly` output of the current method can serve as a weak label for
bootstrapping supervised approaches.

![Event detection](plots/peak_amplitude.png)

![Anomaly detection score](plots/anomaly_detection.png)

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

---

## 7. Classification of Acoustic Events

The final output combines:

- **Time-domain:** focused DAS waveform showing the pulse shape, coda, and decay
- **Frequency-domain:** Welch PSD of the focused signal (bar chart, 1–148 Hz)

The PSD shows a dominant peak near **100 Hz**, consistent with the plasma-pulse source
spectrum reported in the paper.

### Feature-Based Classification Framework

The 7-feature vector (Section 3) together with source-localisation outputs `(x0, z0)`
and beamforming slowness estimates can form a **classification feature vector**:

| Feature group | Examples |
|---------------|---------|
| Energy features | RMS, Peak, band energies |
| Impulsiveness | Crest Factor, Kurtosis |
| Spectral shape | WPD sub-band ratios, PSD peak frequency |
| Spatial | Source distance `z0`, along-cable position `x0` |
| Directional | Dominant slowness from beamforming |

Possible classifiers: **Random Forest**, **SVM**, **1-D CNN** on the raw spectrogram.  
With multi-event labelling (events 1–10), leave-one-out cross-validation provides an
unbiased accuracy estimate even with the small sample size.



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
