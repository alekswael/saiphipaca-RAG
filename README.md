# David and Goliath: Domain-specific instruction fine-tuning of a lightweight LLM (phi-1.5) on synthetic data for use in RAG applications

This repository contains the code for instruction tuning `microsoft/phi-1_5` on the Alpaca dataset and on the Synthetic Academic Dataset (SAI) to create two models: `alekswael/phipaca` and `alekswael/saiphipaca`. It also includes the code for generating the SAI dataset, Alpaca subset and a benchmark for testing the models performance on a Retrieval Augmented Generation (RAG) task.

#### Abstract from [Wael](https://github.com/alekswael) & [Baskakovs](https://github.com/sashapustota) (2023):
_The development of large language models (LLMs) like OpenAI's GPT-4 has yielded incredible results for NLP tasks. Despite their impressive capabilities, the scale of these models poses challenges in terms of computational resources and accessibility. This study addresses the issue of scale in LLMs by exploring the viability of smaller language models (SLMs) in achieving comparable performance to their larger counterparts. In this context, we instruction-tune a small 1.3B parameter model (Microsoft’s phi-1.5) using QLoRA, on two datasets; 1) using a subset of the Alpaca dataset for task-independent instruction-following abilities, and 2) using a generated Synthetic Academic Instruct dataset for task-dependent performance. The fine-tuned models, named phipaca and saiphipaca, are then evaluated against the base phi-1.5 model in a Retrieval Augmented Generation (RAG) task. Our evaluation, based on cosine similarity with outputs from a state-of-the-art model (gpt-3.5-turbo) and an inspection of model outputs, reveals that while the base model performs well, the fine-tuned models exhibit limitations, such as false information and hallucinations, suggesting a need for refinement in training methodologies._

## Project Structure

- `data/`: Contains the datasets used for training the models. It includes the SAI dataset, Alpaca dataset, and benchmark data.
- `models/`: Contains the saved model checkpoints. It includes the phipaca checkpoint (alpaca_peft) and the saiphipaca checkpoint (synthetic_peft).
- `results/`: Contains the results of the model training and evaluation. It includes chunk histograms, cosine similarity plots, and CSV files with results for different tasks.
- `src/`: Contains the source code for the project. It includes code for benchmark scores, data generation, and model training.
- `run_tests.sh`: A shell script to run the RAG-benchmarks.
- `requirements.txt`: Contains the Python dependencies required for this project.

## Installation

Be sure to have Python installed on your system before proceeding.

To create a Python virtual environment and install the required packages, open a bash terminal and run the following script (with the repo folder as root):

```
bash setup.sh
```

## Usage

### Data generation

To generate the SAI dataset, run the following script:

```
bash data_generation.sh
```

- This script runs the scripts in the `data_generation` folder in the correct order.

### Model training

The training of the models was done using Google Colab, utilising a Nvidia Tesla T4 GPU. The Jupyter Notebooks are available in the `model_training` folder. Be sure to modify these scripts to include your own paths.

### RAG benchmarking

To test the models capabilities for RAG, run the following script:

```
bash run_tests.sh
```

- This bash script performs all RAG benchmark tests, results can be found in the `results` folder. It creates `cosine_similarity.png`, two bar plots of the models' performance on the single PDF- and joint PDF task.

## License

The project is licensed under the MIT license.

## Full repository structure

```
├── data
│   ├── alpaca_dataset
│   │   ├── dataset_test.json
│   │   └── dataset_train.json
│   ├── benchmark_data
│   │   ├── all_questions_data
│   │   │   ├── benchmark_data.csv
│   │   │   └── benchmark_data.xlsx
│   │   ├── joint_paper
│   │   │   ├── joint_paper_data.csv
│   │   │   ├── joint_paper_data.xlsx
│   │   │   └── joint_paper.pdf
│   │   └── single_paper
│   │       ├── single_paper_data.csv
│   │       ├── single_paper_data.xlsx
│   │       └── single_paper.pdf
│   └── SAI_dataset
│       ├── dataset_test.json
│       ├── dataset_train.json
│       └── SAI_dataset_0312.csv
├── LICENSE
├── models
│   ├── alpaca_peft
│   │   ├── checkpoints
│   │   │   └── checkpoint-1000
│   │   │       ├── adapter_config.json
│   │   │       ├── adapter_model.safetensors
│   │   │       ├── optimizer.pt
│   │   │       ├── README.md
│   │   │       ├── rng_state.pth
│   │   │       ├── scheduler.pt
│   │   │       ├── trainer_state.json
│   │   │       └── training_args.bin
│   │   ├── loss_epochs.png
│   │   └── loss_steps.png
│   └── synthetic_peft
│       ├── checkpoints
│       │   └── checkpoint-267
│       │       ├── adapter_config.json
│       │       ├── adapter_model.safetensors
│       │       ├── optimizer.pt
│       │       ├── README.md
│       │       ├── rng_state.pth
│       │       ├── scheduler.pt
│       │       ├── trainer_state.json
│       │       └── training_args.bin
│       ├── loss_epochs.png
│       └── loss_steps.png
├── README.md
├── requirements.txt
├── results
│   ├── chunk_histograms
│   │   ├── joint_paper_histogram_phi-1_5.png
│   │   ├── joint_paper_histogram_phipaca.png
│   │   ├── joint_paper_histogram_saiphipaca.png
│   │   ├── single_paper_histogram_phi-1_5.png
│   │   ├── single_paper_histogram_phipaca.png
│   │   └── single_paper_histogram_saiphipaca.png
│   ├── cosine_similarity
│   │   ├── cosine_similarity.png
│   │   ├── joint_paper_results_phi-1_5.csv
│   │   ├── joint_paper_results_phipaca.csv
│   │   ├── joint_paper_results_saiphipaca.csv
│   │   ├── single_paper_results_phi-1_5.csv
│   │   ├── single_paper_results_phipaca.csv
│   │   └── single_paper_results_saiphipaca.csv
│   ├── joint_paper_results_phi-1_5.csv
│   ├── joint_paper_results_phipaca.csv
│   ├── joint_paper_results_saiphipaca.csv
│   ├── single_paper_results_phi-1_5.csv
│   ├── single_paper_results_phipaca.csv
│   └── single_paper_results_saiphipaca.csv
├── run_tests.sh
└── src
    ├── benchmark_scores
    │   ├── cosine_similarity_benchmark_data.py
    │   ├── RAG_test_GPT.py
    │   ├── RAG_test.py
    │   └── results_plots.py
    ├── data_generation
    │   ├── cosine_similarity_training_data.py
    │   ├── data_fix.py
    │   ├── synthetic_data_generator.py
    │   └── synthetic_data_prep.py
    └── model_training
        ├── phipaca_train.ipynb
        └── saiphipaca_train.ipynb
```
