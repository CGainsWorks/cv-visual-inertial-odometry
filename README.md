# cv-visual-inertial-odometry

> **Status: planned. This repository is a brief, not an implementation.**
> Nothing here is built yet. The sections below define what goes in it.

A visual-inertial odometry system built from the ground up: feature tracking front end, IMU preintegration, optimisation-based back end and loop-closure detection, benchmarked against public datasets with absolute and relative trajectory error.

---

## What this covers

CV entries this repository is the evidence for:

- Visual-Inertial Odometry

---

## Requirements

### Hardware

- None required if using public benchmarks
- Optional physical target: global-shutter camera, IMU, rigid mount, motion-capture ground truth

### Software

- Feature tracking (KLT or learned features)
- IMU preintegration
- Optimisation back end (Ceres or GTSAM)
- Loop-closure detection (DBoW2 or similar)
- evo for trajectory evaluation
- EuRoC MAV and TUM VI datasets

### Report and documentation deliverables

- Camera and IMU extrinsic and temporal calibration
- Absolute and relative trajectory error against ground truth
- Loop-closure effect analysis
- Drift rate per metre travelled
- Computational profile

---

## Inputs

- Synchronised image stream
- IMU measurements
- Calibration parameters
- Ground-truth trajectory for evaluation

## Outputs

- Estimated 6-DoF trajectory
- Sparse landmark map
- Loop-closure events
- ATE and RPE plots and tables
- TUM-format trajectory files

---

## Intended structure

```
src/
  frontend/               feature tracking, keyframe selection
  imu/                    preintegration
  backend/                optimisation, marginalisation
  loop_closure/
  calibration/            camera-IMU extrinsic and temporal
  io/                     dataset loaders (EuRoC, TUM VI)
evaluation/               evo wrappers, ATE/RPE scripts
results/                  trajectory files and plots
docs/
```

Plus the standard baseline: `README.md`, `LICENSE`, `.gitignore`, `.gitattributes`
(Git LFS rules for weights, bags and media), `.github/workflows/ci.yml`, `docs/`,
`scripts/setup.sh`, `scripts/run_demo.sh` and `tests/`.

---

## Build checklist

- [ ] EuRoC dataset loader
- [ ] Feature tracking front end
- [ ] IMU preintegration
- [ ] Visual-only odometry baseline
- [ ] Optimisation back end fusing visual and inertial
- [ ] Camera-IMU extrinsic and temporal calibration
- [ ] Loop-closure detection and correction
- [ ] ATE and RPE evaluation on EuRoC sequences
- [ ] Drift-rate analysis with and without loop closure
- [ ] Runtime profiling

---

## Definition of done

This repository is finished when:

- [ ] The whole thing runs from one command on a clean machine.
- [ ] The README opens with a diagram or a GIF, not a wall of text.
- [ ] Every number quoted in the README is reproducible from a script in the repo.
- [ ] Limitations are stated honestly in their own section.
- [ ] A licence is declared.
- [ ] Large binaries live in Releases or LFS, not in the Git history.
- [ ] The repository URL is added to the matching CV bullet point.

---

## Starter prompt

Paste this into your coding agent of choice to begin.

<details>
<summary>Expand</summary>

```
You are helping me build a portfolio project from scratch. Read this brief and then
propose a concrete implementation plan before writing any code.

PROJECT: cv-visual-inertial-odometry
A visual-inertial odometry system built from the ground up: feature tracking front end, IMU preintegration, optimisation-based back end and loop-closure detection, benchmarked against public datasets with absolute and relative trajectory error.

TARGET STACK
- Feature tracking (KLT or learned features)
- IMU preintegration
- Optimisation back end (Ceres or GTSAM)
- Loop-closure detection (DBoW2 or similar)
- evo for trajectory evaluation
- EuRoC MAV and TUM VI datasets

HARDWARE CONTEXT
- None required if using public benchmarks
- Optional physical target: global-shutter camera, IMU, rigid mount, motion-capture ground truth

THE SYSTEM MUST TAKE THESE INPUTS
- Synchronised image stream
- IMU measurements
- Calibration parameters
- Ground-truth trajectory for evaluation

AND PRODUCE THESE OUTPUTS
- Estimated 6-DoF trajectory
- Sparse landmark map
- Loop-closure events
- ATE and RPE plots and tables
- TUM-format trajectory files

INTENDED REPOSITORY LAYOUT
src/
  frontend/               feature tracking, keyframe selection
  imu/                    preintegration
  backend/                optimisation, marginalisation
  loop_closure/
  calibration/            camera-IMU extrinsic and temporal
  io/                     dataset loaders (EuRoC, TUM VI)
evaluation/               evo wrappers, ATE/RPE scripts
results/                  trajectory files and plots
docs/

CONSTRAINTS
- Everything must run from a single command: `docker compose up` or `make demo`.
- No placeholder or stub code presented as working. If something is not implemented, say so.
- Every quantitative claim in the README must be reproducible from a script in this repo.
- Use only public or synthetic data. Record the licence of every dataset used.


START BY
1. Confirming or challenging the proposed repository layout.
2. Listing the specific libraries and versions you intend to pin.
3. Identifying the single riskiest technical unknown and how to de-risk it first.
Then work through the milestones in the README build checklist, one at a time.

```

</details>

---

## Notes

- Do not backdate commits to match CV dates. Publishing older work today is normal and expected.
- Record dataset licences before training on anything. Several common research datasets are
  non-commercial only.
- Flip this repository to public once there is something in it worth showing.
