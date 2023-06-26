# DeepCover
## Advancing RNN Test Coverage and Online Error Prediction using State Machine Extraction
Paper Citation: https://dx.doi.org/10.2139/ssrn.4382943



## Abstract
Recurrent neural networks (RNNs) have emerged as powerful tools for processing sequential data in various fields, including natural language processing and speech recognition. However, the lack of explainability in RNN models has limited their interpretability, posing challenges in understanding their internal workings. To address this issue, this paper proposes a methodology for extracting a state machine (SM) from an RNN-based model to provide insights into its internal function. The proposed SM extraction algorithm was assessed using four newly proposed metrics: Purity, Richness, Goodness, and Scale. The proposed methodology along with its assessment metrics contribute to increasing explainability in RNN models by providing a clear representation of their internal decision making process through the extracted SM. In addition to improving the explainability of RNNs, the extracted SM can be used to advance testing and and monitoring of the primary model (PM). \FF{To improve RNNs testing, we introduce six model coverage criteria on the SM for evaluating test suites aimed at analyzing the PM. The coverage criteria give the developer a metric for evaluating the effectiveness of the test suite. We also propose a tree-based model  to predict the error probability of the PM for each input based on the extracted SM. We evaluated our method on MNIST dataset and Mini Speech Commands dataset achieving an area under the curve (AUC) of the receiver operating characteristic (ROC) chart exceeding 80%.

