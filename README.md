

## Project Overview
This project demonstrates the fine-tuning of the **Gemma 3** model (1 Billion parameters, Keras, English version, v2) using a specialized **Telecoms Operations Q & A** dataset (`ops_eng_trunc.csv`). The goal is to adapt the pre-trained model to answer IT operations and telecommunications troubleshooting questions.

## Model Used
- **Model**: Gemma 3 (1B parameters, Keras, English version, v2)
- **Source**: `keras/gemma3/Keras/gemma3_1b/3` (downloaded via KaggleHub)
- **Fine-tuning Method**: LoRA (Low Rank Adaptation) is applied to efficiently adapt the model to the target domain by adjusting a small subset of its parameters.

## Dataset
- **Name**: Telecoms Operations Q & A dataset (`ops_eng_trunc.csv`)
- **Source**: `oculeusptyltd/llm-for-it-operation` (downloaded via KaggleHub)
- **Content**: Contains pairs of IT operations and telecommunications troubleshooting questions and answers.
- **Preparation**: The dataset is preprocessed to limit the number of entries for demonstration purposes (currently set to `data_size = 500`) and formatted for training.

## Setup and Dependencies
To run this notebook, you need to install the following Python packages and configure the Keras backend:

1.  **Install Keras-NLP and Keras**: 
    ```bash
    !pip install -q -U keras-nlp
    !pip install -q -U keras>=3
    ```

2.  **Configure Keras Backend**: This notebook is configured to use the JAX backend for Keras 3. You can also choose TensorFlow or PyTorch. The `XLA_PYTHON_CLIENT_MEM_FRACTION` is set to optimize memory usage with JAX.
    ```python
    import os
    os.environ["KERAS_BACKEND"] = "jax" # Options: "jax", "tensorflow", "torch"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.90"
    os.environ["JAX_PLATFORMS"] = ""
    ```

## Kaggle Data and Model Sources
This notebook relies on data and models hosted on Kaggle. To access these, you will need to:
1.  **Kaggle Account**: Ensure you have a Kaggle account.
2.  **Kaggle API Credentials**: Follow the instructions to log in to KaggleHub. This typically involves setting up your Kaggle API token.
    ```python
    import kagglehub
    kagglehub.login()
    ```
3.  **Download Sources**: The notebook automatically downloads the dataset and pre-trained Gemma 3 model using `kagglehub.dataset_download` and `kagglehub.model_download`.

## How to Run the Notebook
1.  **Environment Setup**: Ensure you are in a Python environment with the required packages installed (as described in 'Setup and Dependencies'). A Google Colab environment is recommended.
2.  **Kaggle Login**: Run the `kagglehub.login()` cell to authenticate and download necessary data/models.
3.  **Execute Cells Sequentially**: Run all cells in the notebook from top to bottom.
    - **Data Loading**: The `ops_eng_trunc.csv` dataset is loaded and prepared.
    - **Model Loading**: The `Gemma3CausalLM` model is loaded from its preset.
    - **Fine-tuning**: LoRA is enabled, and the model is fine-tuned on the prepared Q&A dataset for a specified number of epochs (e.g., `epochs = 3`).
    - **Inference**: Examples of model inference are shown both before and after fine-tuning to demonstrate the impact of LoRA.
    - **Visualization**: Training loss and accuracy are plotted.
    - **Model Saving**: The fine-tuned model is saved to a local preset directory.

## Customization
- **`cfg.data_size`**: Adjust this parameter to use a larger or smaller portion of the dataset for training. A smaller size (e.g., 500) is used for quick demonstration.
- **`cfg.epochs`**: Modify the number of training epochs to control the extent of fine-tuning.
- **`cfg.lora_rank`**: Adjust the LoRA rank to experiment with different levels of parameter adaptation.
- **Keras Backend**: Change `os.environ["KERAS_BACKEND"]` to `
