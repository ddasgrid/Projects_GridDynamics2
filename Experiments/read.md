## base_model.ipynb

**Objective:** Establish a strong performance baseline using 100% of the dataset by testing a simple fusion mechanism and progressively unfreezing the transformer layers.

## Core Architecture Blueprint

This identical base pipeline was used across all three training sessions in this file:
* Image Encoder: Google's Vanilla Vision Transformer (ViT).
* Text Encoder: Bidirectional Encoder Representations from Transformers (BERT).
* Fusion Head: Concatenation followed by a Feed Forward Neural Network (FFNN).
* Hyperparameters: Hidden Dimensions: 512, Dropout Rates: 0.1, Number of Layers (Depth): 2.
* Classifier Head: Standard Fully Connected / Linear Layer (nn.Linear).

## Experimental Runs & Insights

**1. Fully Frozen Encoders**
* Setup: Trained on 100% of the data with all layers of transformers frozen.
* Result: Reached a final validation loss of 0.9016 and a validation accuracy of approximately 58%. The model was saved to saved_model_acc_58.0.pth.
* What it suggests: While pre-trained embeddings hold strong general knowledge, forcing them through a simple concatenation mechanism without allowing them to learn task-specific visual entailment features creates a severe performance bottleneck.

**2. Light Fine-Tuning**
* Setup: Trained on 100% of the data with the last 2 layers unfrozen.
* Result: Reached a validation accuracy of 70.69%. The model was saved to best_model_acc_70.7.pth.
* What it suggests: Granting the image and text encoders just a slight degree of flexibility to adjust their internal representations to the data yields a massive 12%+ jump in performance.

**3. Deep Fine-Tuning**
* Setup: Trained on 100% of the data with the last 6 layers of the transformer unfrozen.
* Result: Achieved a validation accuracy of 73.73%. The model was saved to best_model_acc_73.7.pth.
* What it suggests: Unfreezing half of the network allows for deep, task-specific alignment between the visual and textual modalities. It proves that a highly simple architecture can achieve competitive scores if the powerful pre-trained encoders are permitted to do the heavy lifting.

## Fusion Head Optimization (Experiment_1.ipynb to Experiment_4.ipynb)

**Objective:** Conduct a rapid hyperparameter and architecture search to determine the optimal Fusion Mechanism. To iterate quickly and isolate the fusion head's learning capacity, a 20% subset of the data was used with all transformer layers completely frozen. 

## Core Architecture Blueprint

The following base pipeline was used across all five experiments:
* Image Encoder: Google's Vanilla Vision Transformer (ViT) (Frozen).
* Text Encoder: Bidirectional Encoder Representations from Transformers (BERT) (Frozen).
* Fusion Head Hidden Dimensions: 512.
* Classifier Head: Standard Fully Connected / Linear Layer (nn.Linear).


## Experimental Runs & Insights

**1. Element-wise Multiplication (Experiment_2.ipynb)**
* Setup: Depth 2, Dropout Rate 0.1.
* Result: Reached a validation accuracy of 44.22%.
* What it suggests: Simple multiplicative merging of embeddings struggles to capture complex cross-modal relationships when the base encoders cannot adapt.

**2. Element-wise Addition (Experiment_3.ipynb)**
* Setup: Depth 2, Dropout Rate 0.1.
* Result: Reached a validation accuracy of 48.32%.
* What it suggests: Marginally better than multiplication, but still severely underperforms. Additive merging lacks the expressive power needed to find logical contradictions or entailments between image and text.

**3. Concatenation (Experiment_4.ipynb)**
* Setup: Depth 2, Dropout Rate 0.1.
* Result: Reached a strong validation accuracy of 57.65%.
* What it suggests: Appending the raw embeddings together and passing them through a Feed Forward Neural Network performs significantly better than simple mathematical merging, acting as a very strong baseline.

**4. Attention Fusion (Experiments_1.ipynb)**
* Setup: Tested at Depth 2 (Dropout 0.1) yielding 57.74% accuracy, and Depth 1 (Dropout 0.3) yielding 56.53% accuracy.
* Result: Attention Fusion achieved the highest overall accuracy (57.74%), narrowly beating Concatenation. 
* What it suggests: Cross-attention is the most effective and theoretically sound fusion method for this pipeline. By allowing the text to actively query the image representations, the model finds better alignments between modalities, making it the chosen fusion architecture for the final model build.

## Experiment_1.ipynb

**Objective:** With Attention Fusion selected as the optimal merging strategy, this experiment focuses on fine-tuning the hyperparameters of the Fusion Head (depth, hidden dimensions, and dropout rate) to maximize validation accuracy. All base transformer layers remained frozen to isolate the fusion head's learning capacity.

## Core Architecture Blueprint

* Image Encoder: Google's Vanilla Vision Transformer (ViT) (Frozen).
* Text Encoder: Bidirectional Encoder Representations from Transformers (BERT) (Frozen).
* Fusion Head: Attention Fusion (Cross-Attention). 
* Classifier Head: Standard Fully Connected / Linear Layer (nn.Linear).

## Experimental Runs & Insights

**1. Deep Fusion Head (Depth 4)**
* Setup: Depth: 4, Hidden Dim: 512, Dropout: 0.1
* Result: Validation accuracy reached 56.58%.
* What it suggests: A deeper fusion network (4 layers) on frozen embeddings creates unnecessary complexity, slightly bottlenecking the model's ability to generalize on the smaller subset of data.

**2. Shallow Fusion Head (Depth 1)**
* Setup: Depth: 1, Hidden Dim: 512, Dropout: 0.1
* Result: Validation accuracy improved to 58.00%.
* What it suggests: Reducing the depth from 4 to 1 layer yields better results. A shallower network prevents overfitting and processes the frozen representations more efficiently.

**3. Reduced Hidden Dimensionality**
* Setup: Depth: 1, Hidden Dim: 256, Dropout: 0.1
* Result: Validation accuracy dropped slightly to 57.75%.
* What it suggests: Shrinking the hidden dimension restricts the representational capacity of the attention head. 512 dimensions allow for a richer alignment between the visual and textual data.

**4. Increased Dropout (The Winning Configuration)**
* Setup: Depth: 1, Hidden Dim: 512, Dropout: 0.3
* Result: Validation accuracy peaked at 58.23%.
* What it suggests: Increasing the dropout rate to 0.3 acts as a strong regularizer. It prevents the shallow fusion head from heavily relying on specific nodes, creating a more robust representation.

## Finalized Fusion Architecture
Based on these runs, the optimal architecture for the Fusion Head moving forward is **Attention Fusion with Depth 1, Hidden Dimensions of 512, and a Dropout Rate of 0.3**.

## Classifier Head Optimization (Experiments_5.ipynb - Experiment_7.ipynb)

**Objective:** With the Fusion Head architecture finalized (Attention Fusion, Depth 1, Hidden Dimensions 512, Dropout 0.3), the focus shifted to finding the most effective Classifier Head to interpret the fused representations. The 20% data subset with fully frozen encoders was used to isolate the classifier's performance.

## Core Architecture Blueprint

The following base pipeline was used across these three experiments:
* Image Encoder: Google's Vanilla Vision Transformer (ViT) (Frozen).
* Text Encoder: Bidirectional Encoder Representations from Transformers (BERT) (Frozen).
* Fusion Head: Attention Fusion (Depth: 1, Hidden Dimensions: 512, Dropout: 0.3).
* Classifier Head: Varied across experiments.

## Experimental Runs & Insights

**1. Standard Linear Layer (Experiment_5.ipynb)**
* Setup: Fully Connected / Linear Layer (nn.Linear).
* Result: Reached a validation accuracy of 56.53%.
* What it suggests: A basic linear layer serves as an adequate baseline but struggles to decode the complex, high-dimensional cross-attention alignments efficiently.

**2. Deep MLP (Experiment_6.ipynb)**
* Setup: Deep Multi-Layer Perceptron.
* Result: Reached a validation accuracy of 56.76%.
* What it suggests: Adding standard depth to the classifier provides only a marginal improvement over a single linear layer. Traditional activation functions in the MLP are not fully capturing the non-linear relationships in the fused data.

**3. SwiGLU Classifier (Experiment_7.ipynb)**
* Setup: SwiGLU (Swish-Gated Linear Unit) Classifier. 
* Result: Reached a validation accuracy of 59.26% with a strong validation loss of 0.8868.
* What it suggests: The SwiGLU classifier significantly outperforms standard MLPs and linear layers. By using gated non-linearities, it effectively filters and scales the most important features from the attention head, leading to a nearly 3% jump in accuracy on the frozen baseline.

## Finalized Architecture
Based on these runs, the SwiGLU Classifier was selected as the final classification head for the pipeline, completing the custom multimodal architecture.

## Core Architecture Blueprint

* Image Encoder: Google's Vanilla Vision Transformer (ViT).
* Text Encoder: Bidirectional Encoder Representations from Transformers (BERT).
* Fusion Head: Attention Fusion (Depth: 1, Hidden Dimensions: 512, Dropout: 0.3).
* Classifier Head: SwiGLU Classifier.

## Experimental Runs & Insights

**1. Training on 50% Data (Experiment_8.ipynb)**
* Setup: 50% of the dataset, with 2 layers of the transformers frozen.
* Result: The model showed stable, steady learning over 3 epochs, reaching a validation accuracy of 67.21% and a validation loss of 0.7516.
* What it suggests: The custom attention and SwiGLU architecture scales well to medium-sized data. The gradients remained stable, and the model successfully learned cross-modal alignments without overfitting.

**2. Training on 100% Data with Hard Negatives (Experiment_9.ipynb)**
* Setup: 100% of the dataset, infused with hard negatives. More transformer layers were unfrozen (4 layers unfrozen) to allow for maximum flexibility.
* Result: The training became highly unstable. In early runs, accuracy plateaued and triggered early stopping around 59.64% to 66.99%. In the final run, the model completely collapsed in Epoch 2, returning a `NaN` (Not a Number) training/validation loss and dropping to a baseline 33.37% accuracy. 

## Final Conclusion: The Simple Base Model Won
