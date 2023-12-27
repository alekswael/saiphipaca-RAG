# David and Goliath: Instruction fine-tuning a small LLM (phi-1.5) on synthetic data for use in RAG applications

This repository contains the code for instruction tuning ```microsoft/phi-1_5``` on the Alpaca dataset and on the Synthetic Academic Dataset (SAI) to create two models: ```alekswael/phipaca``` and ```alekswael/saiphipaca```.

#### Abstract from the paper:

## Installation

```
pip install -r requirements.txt
```

## Project Structure

```
- `data/`: Contains the datasets used for training the models. It includes the SAI dataset, Alpaca dataset, and benchmark data.
- `models/`: Contains the trained models. It includes the Alpaca PEFT model and the Synthetic PEFT model.
- `results/`: Contains the results of the model training and evaluation. It includes histograms, cosine similarity plots, and CSV files with results for different tasks.
- `src/`: Contains the source code for the project. It includes code for benchmark scores, data generation, and model training.
- `run_tests.sh`: A shell script to run tests.
- `requirements.txt`: Contains the Python dependencies required for this project.
```

## Usage

The data-preprocessing and training of the models was done using Google Colab, utilising a Nvidia Tesla T4 GPU.

To generate data, train the models and run the RAG benchmarks, follow these steps:

### Data generation:
synthetic_data_prep.py
synthetic_data_generator.py
data_fix.py
cosine_similarity_training_data.py

```
bash run_tests.sh
```

This bash script performs all RAG benchmark tests, results can be found in the ```results``` folder.

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
