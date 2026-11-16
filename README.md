# Selecting and Transmitting ECG Information Under a Limited Wireless Budget

**Puyang Zhao · Jingxi Zhao · Huiyun Zhang**

Research code accompanying the article accepted for publication in the *IEEE Journal of Biomedical and Health Informatics*.

We study which ECG lead representations to transmit and how to encode them under a fixed wireless budget. The code implements the shared per-lead encoder, record-specific lead selection, ranked repetition, SSCC, and single-lead and joint JSCC.

![Schematic ECG selection and transmission workflow](assets/overview.gif)

## Quick start

Download [PTB-XL 1.0.3](https://physionet.org/content/ptb-xl/1.0.3/) and place or link it at `code/pipeline/data/ptbxl/`. The external dataset is [ECG-Arrhythmia 1.0.0](https://physionet.org/content/ecg-arrhythmia/1.0.0/).

```bash
cd code/pipeline
conda env create -f environment.yml
conda activate ecg_semcomm
python -m pip install --no-deps wfdb==4.3.1
export PYTHONNOUSERSITE=1

python scripts/train_baseline.py --config configs/baseline.yaml --run-name baseline_run1
python scripts/precompute_embeddings.py
python scripts/train_phase2a.py --method proposed --seed 0 --config configs/phase2a.yaml --run-name phase2a_proposed_seed0
```

The transmission training entry points are `scripts/train_phase2b_pilot.py` for ranked repetition and `revision_audit/comm/train_comm.py` for the coding comparisons. The coding comparisons use the pretrained embeddings and the selector checkpoint for the requested seed. Run either script with `--help` for its arguments. Model and channel implementations are in `src/`, with training settings in `configs/`.

## Citation

P. Zhao, J. Zhao, and H. Zhang, “Selecting and Transmitting ECG Information Under a Limited Wireless Budget,” *IEEE Journal of Biomedical and Health Informatics*, accepted for publication.

Code is licensed under the [MIT License](LICENSE).
