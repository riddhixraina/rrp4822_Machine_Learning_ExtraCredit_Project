# Lakh MIDI Scaling Study

End-to-end pipeline for training and evaluating Transformer and LSTM language models on a Lakh MIDI subset, including data conversion, scaling experiments, and sample generation.

## Environment
- Python 3.10+ recommended
- Create a virtual environment, then install dependencies:
  - `pip install -r requirements.txt`

## Data preparation
1) Prepare a manageable Lakh MIDI subset ZIP (e.g., 10k–30k MIDI files). The full dataset is too large for this script.  
2) Open `LakhMIDI.py` and set `Cfg.LMD_ZIP_PATH` to the local path of your subset ZIP.  
   - If you already have extracted MIDI files, set `Cfg.USE_LOCAL_MIDI=True` and point `Cfg.LOCAL_MIDI_ROOT` to that directory.  
3) The script will write intermediate data to `data_lakh/midi` and `data_lakh/abc` by default. Adjust `Cfg.DATA_ROOT`, `Cfg.MIDI_ROOT`, and `Cfg.ABC_ROOT` if desired.

## Running the pipeline
Run the full pipeline (download/extract -> convert -> tokenize -> train -> evaluate -> sample):
```
python LakhMIDI.py
```
Key knobs (edit in `Cfg`):
- `MAX_MIDI_FILES`: limit how many MIDI files to convert.
- `TRAIN_TOKENS_TARGET`: total tokens to stream for one epoch per model.
- `BATCH_TOKENS`, `SEQ_LEN`: adjust if you encounter GPU OOM.
- `SAVE_DIR`: where checkpoints, plots, and samples are saved (default `./runs_lakh`).

## Outputs
- Model checkpoints: `runs_lakh/{gpt_*,lstm_*}.pt`
- Scaling plots: `runs_lakh/*_scaling.png` and `runs_lakh/combined_scaling.png`
- Training curves and metrics: `runs_lakh/scaling_results.json`, `runs_lakh/training_curves.json`
- Samples and MIDI conversions: `runs_lakh/best_samples/`

## Notebook
- `Machine_Learning_Project.ipynb` mirrors the pipeline in notebook form (data download, preprocessing, training loops, and progress widgets).  
- Open in Jupyter/Colab, ensure `pip install -r requirements.txt`, and re-run cells from the top.  
- If widgets fail to render, trust the notebook and restart the kernel; widget state is stored under notebook metadata.

### Notebook datasets
- Default notebook run uses the **TheSession ABC dataset** (pulled from the public mirror) for training/eval.
- A Lakh MIDI path is also scaffolded in code, but full Lakh scaling runs were **not completed** here due to resource limits. Use `LakhMIDI.py` with a manageable subset and adequate GPU if you want to reproduce those experiments.

### Report alignment
- The accompanying report describes the Session-based training/evaluation performed in the notebook and notes the Lakh MIDI attempt that was abandoned because of resource constraints.

## Notes
- GPU strongly recommended; script will fall back to CPU if CUDA is unavailable.
- PrettyMIDI conversion may skip corrupted files; see console logs for any failures.
