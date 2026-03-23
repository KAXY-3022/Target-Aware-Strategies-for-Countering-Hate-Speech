# Beyond Generic Responses: Target-Aware Strategies for Countering Hate Speech

Code for training, running, and evaluating **target-aware counterspeech generation** models against hateful or sexist input. The repository includes utilities for:

- training causal and seq2seq generators,
- running inference on custom evaluation sets,
- scoring generations with automated metrics and classifiers,
- training auxiliary topic / counterspeech classifiers.

This repo is the code implementation for "[Beyond Generic Responses: Target-Aware Strategies for Countering Hate Speech (LREC 2026)](https://something)".

## What this repository contains

The Python package lives under `src/lm_against_hate/` and is organized into a few main areas:

- `config/`: central paths, training arguments, inference settings, and evaluation model configuration.
- `utilities/`: model loading, data loading, batching, cleanup, and helper functions.
- `scripts/`: runnable entry points for training, inference, evaluation, and analysis workflows.
- `evaluation/`: automated evaluation pipeline and metrics.
- `inference/`: prediction and post-processing utilities.

## Repository status and assumptions

This repository is currently structured as a research codebase rather than a polished end-user package. A few assumptions are baked into the code:

- data files are expected in local `data/` subfolders configured in `src/lm_against_hate/config/config.py`,
- locally saved models are expected under `models/`,
- predictions are written to `predictions/`,
- some workflows expect a `credentials.json` file in the repository root,
- several scripts are meant to be edited before running to choose models or datasets.

If you want a quick start, read the setup section first and then use the script-specific instructions below.

## Project layout

```text
.
├── README.md
├── requirements.txt
├── setup.py
├── installation
└── src/
    └── lm_against_hate/
        ├── config/
        ├── evaluation/
        ├── inference/
        ├── notebooks/
        ├── scripts/
        └── utilities/
```

## Prerequisites

- Python 3.9+
- A recent PyTorch installation
- Optional but recommended: an NVIDIA GPU for training and larger-model inference
- Enough disk space for Hugging Face models, local checkpoints, and generated outputs

Some dependencies in `requirements.txt` are GPU-specific or heavyweight, including:

- `torch`, `torchvision`, `torchaudio`
- `bitsandbytes`
- `flash-attn`
- `xformers`
- `transformers`, `datasets`, `peft`

If your environment does not support CUDA, install a CPU-compatible subset instead of blindly installing every package.

## Installation

### Option 1: minimal editable install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
```

This uses `setup.py`, which reads dependencies from `requirements.txt`.

### Option 2: manual install for GPU environments

The repository also includes an `installation` helper file with example commands for CUDA 12.4 environments:

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install transformers datasets bertopic nltk tweet-preprocessor peft tf-keras
pip install pynvml adjustText seaborn datamapplot bitsandbytes
pip install flash-attn --no-build-isolation
pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu124
```

Use this path only if the packages match your driver, CUDA, and compiler setup.

## Required local files and folders

Before running the scripts, create the directories expected by the code:

```bash
mkdir -p data/Custom data/Custom_New models predictions evaluation_results logs
```

### `credentials.json`

Some scripts require API and Hugging Face credentials. Create a root-level `credentials.json` such as:

```json
{
  "HF_TOKEN": "your_huggingface_token",
  "Perspective_API": "your_perspective_api_key"
}
```

At minimum:

- `generator_training.py` expects `HF_TOKEN`,
- evaluation utilities may expect `Perspective_API`.

## Expected datasets

The code references these datasets in the README and scripts:

- **CONAN**
- **Reddit / Gab benchmark data**
- **CrowdCounter**
- **EDOS / sexism-specific data**

The config defaults point to files such as:

- `data/Custom/CONAN_train.csv`
- `data/Custom/CONAN_val.csv`
- `data/Custom/CONAN_test.csv`
- `data/Custom/T8-S10.csv`
- `data/Custom/EDOS_sexist.csv`
- `data/Custom_New/Classifier_train.csv`
- `data/Custom_New/Classifier_val.csv`
- `data/Custom_New/Classifier_test.csv`

Make sure your CSV column names match what the dataloaders expect, especially:

- `Hate_Speech`
- `Counter_Speech`
- `Target`

For classifier workflows, the dataset may also need per-category columns such as:

- `MIGRANTS`
- `POC`
- `LGBT+`
- `MUSLIMS`
- `WOMEN`
- `JEWS`
- `other`
- `DISABLED`

## Main workflows

### 1) Train a generator

Edit `src/lm_against_hate/scripts/generator_training.py` and set the values in the `main(...)` call at the bottom for the model you want to train.

Then run:

```bash
PYTHONPATH=src python src/lm_against_hate/scripts/generator_training.py
```

Common model choices noted in the script include:

- `openai-community/gpt2-medium`
- `openai-community/gpt2-xl`
- `facebook/bart-large`
- `google/flan-t5-large`
- `google/flan-t5-xl`
- `google/flan-t5-xxl`
- `meta-llama/Llama-3.2-1B-Instruct`
- `meta-llama/Llama-3.2-3B-Instruct`
- `mistralai/Mistral-7B-Instruct-v0.3`

### Notes

- Training parameters are defined centrally in `src/lm_against_hate/config/config.py`.
- Local checkpoints are written under `models/`.
- Category-aware training is controlled through the `category` flag.

### 2) Run inference

Edit:

- `src/lm_against_hate/config/inf_config.py` to choose which locally available models should run,
- `src/lm_against_hate/scripts/inference.py` to choose datasets.

Then run:

```bash
PYTHONPATH=src python src/lm_against_hate/scripts/inference.py
```

Outputs are saved under:

```text
predictions/<dataset_name>/
```

Supported built-in dataset shortcuts in the current code are:

- `Base`
- `Small`
- `Sexism`

These map to files inside `data/Custom/`.

### 3) Evaluate generated predictions

After inference creates CSV outputs, run:

```bash
PYTHONPATH=src python src/lm_against_hate/scripts/evaluation.py
```

The script scans `predictions/<dataset>/` for CSV files and writes evaluation outputs under:

```text
evaluation_results/<dataset>/
```

### 4) Train auxiliary classifiers

Available classifier-related scripts include:

- `topic_classifier_training.py`
- `topic_classifier_test.py`
- `counter_argument_classifier_training.py`
- `counter_argument_classifier_test.py`

Run them the same way:

```bash
PYTHONPATH=src python src/lm_against_hate/scripts/topic_classifier_training.py
```

Adjust configuration or hard-coded options in each script before running.

### 5) Topic modeling and analysis

Additional research utilities include:

- `BERTopic_training.py`
- `topic_modeling_BERTopic.py`
- `response_length.py`
- `judgeLM_formatter.py`
- `judgelm_scoring.py`
- `Knowledge_Embedding.py`
- `Data_filtering.py`

These are analysis-oriented scripts and may require manual path or credential setup.

## Consistency notes from the current codebase

During a basic code consistency pass, the following repository conventions were confirmed:

- package imports assume `PYTHONPATH=src` when running scripts directly,
- config values are centralized but many scripts still rely on manual editing,
- training / inference / evaluation outputs are expected in separate local folders rather than being auto-provisioned,
- credentials are loaded from a root-level JSON file instead of environment variables.

If you intend to productionize the project, good next steps would be:

1. add a proper CLI,
2. move credentials to environment variables,
3. validate required directories at startup,
4. add small fixture datasets and automated tests,
5. split optional GPU dependencies from core requirements.

## Referenced datasets and related work

- Reddit and Gab benchmark data: [A Benchmark Dataset for Learning to Intervene in Online Hate Speech](https://github.com/jing-qian/A-Benchmark-Dataset-for-Learning-to-Intervene-in-Online-Hate-Speech/tree/master/data)
- CONAN: [COunter NArratives through Nichesourcing](https://github.com/marcoguerini/CONAN)
- CrowdCounter: [CrowdCounter benchmark](https://github.com/hate-alert/crowdcounter)
- EDOS: [Explainable Detection of Online Sexism (EDOS)](https://github.com/rewire-online/edos)

- [Generate, Prune, Select: A Pipeline for Counterspeech Generation against Online Hate Speech](https://github.com/WanzhengZhu/GPS)
- [CounterGeDi: A controllable approach to generate polite, detoxified and emotional counterspeech](https://github.com/hate-alert/CounterGEDI)

## Troubleshooting

### Import errors when running scripts

Use:

```bash
PYTHONPATH=src python path/to/script.py
```

### Model not found locally

Some code paths first try loading from local `models/` folders and only then download from Hugging Face.

### `flash-attn` or `bitsandbytes` installation failures

Those packages are environment-specific. Remove them from your install plan if your machine does not support them.

### Missing data columns

Check your CSV schema against the required columns listed above. Most scripts assume exact column names.

## License

See `COPYING.txt`.
