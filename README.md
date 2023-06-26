# DeepCover: Advancing RNN Test Coverage and Online Error Prediction using State Machine Extraction

## Abstract
Recurrent neural networks (RNNs) have emerged as powerful tools for processing sequential data in various fields, including natural language processing and speech recognition. However, the lack of explainability in RNN models has limited their interpretability, posing challenges in understanding their internal workings. To address this issue, this paper proposes a methodology for extracting a state machine (SM) from an RNN-based model to provide insights into its internal function. The proposed SM extraction algorithm was assessed using four newly proposed metrics: Purity, Richness, Goodness, and Scale. The proposed methodology along with its assessment metrics contribute to increasing explainability in RNN models by providing a clear representation of their internal decision making process through the extracted SM. In addition to improving the explainability of RNNs, the extracted SM can be used to advance testing and and monitoring of the primary model (PM). \FF{To improve RNNs testing, we introduce six model coverage criteria on the SM for evaluating test suites aimed at analyzing the PM. The coverage criteria give the developer a metric for evaluating the effectiveness of the test suite. We also propose a tree-based model  to predict the error probability of the PM for each input based on the extracted SM. We evaluated our method on MNIST dataset and Mini Speech Commands dataset achieving an area under the curve (AUC) of the receiver operating characteristic (ROC) chart exceeding 80%.

# DeepCover: Advancing RNN Test Coverage and Online Error Prediction using State Machine Extraction

This repository contains the code and data required to replicate the experiments presented in the research paper "DeepCover: Advancing RNN Test Coverage and Online Error Prediction using State Machine Extraction". The main goal of our research is to provide an explainable model based on state machines (SM) extracted from recurrent neural networks (RNNs). This repository includes the necessary tools to extract SMs, define coverage criteria, and predict errors online.

## Requirements

To run the code in this repository, please ensure that you have installed the following:

- Python 3.6 or higher
- TensorFlow 2.3.0 or higher
- Keras 2.4.3 or higher
- Scikit-learn 0.23.2 or higher
- NumPy 1.19.2 or higher
- Matplotlib 3.3.2 or higher
- Pandas 1.1.3 or higher

## Repository Contents

The repository is organized as follows:

- `experiments.ipynb`: Jupyter notebook containing the code for reproducing the experiments in the paper.
- `weights/`: Folder containing pre-trained weights for GRU, LSTM, and S-RNN models.
- `kmeans/`: Folder containing K-Means model files for clustering the states in the SM extraction process.
- `data/`: Folder containing the datasets used in the experiments (MNIST and Mini Speech Commands).

## Experiments

The experiments in this study can be divided into the following categories:

1. State Machine Extraction and Evaluation: Evaluating the quality of state machines extracted from RNN-based models trained on the MNIST and Mini Speech Commands datasets. The results are compared with the DeepStellar method.
2. Coverage Criteria Statistical Test: A statistical test is performed to validate the efficacy of the proposed coverage criteria in measuring the completeness of testing in relation to the extracted state machines.
3. Online Error Prediction: Train a tree-based model to predict potential errors in the RNN-based model using input sequence datasets, features extracted from the RNN model's state traces based on the extracted state machine, and its state space.

To reproduce the results of these experiments, simply follow the instructions provided in the `experiments.ipynb` notebook.

## Steps to Reproduce Results

1. Clone the GitHub repository to your local machine.
2. Ensure that all necessary packages and dependencies are installed.
3. Open and run the `experiments.ipynb` notebook in your preferred environment.
4. The notebook contains detailed instructions and explanations for each step of the experiment.

Note: The pre-trained model weights and K-Means models are provided in the `weights/` and `kmeans/` directories. These can be used directly in the experiments, allowing for easy replication of the results. The datasets are available in the `data/` folder.

## Troubleshooting

If you encounter any issues or have questions regarding the code or methodology, please feel free to open an issue on the GitHub repository, and we will do our best to address it promptly.

## Citation

Please cite our paper if you use our method, code, or data in your work:

https://dx.doi.org/10.2139/ssrn.4382943

## License

This repository is licensed under the [MIT License](LICENSE).
