# Glossary

## Data Handling & Preprocessing

- **Dataset**: Collection of samples used for training, validation, and testing.
- **Train/Validation/Test Split**: Common data division; e.g., 80/10/10 or 70/15/15. Controls evaluation reliability.
- **Training Set**: Subset used to update model weights.
- **Validation Set**: Used to monitor generalization and tune hyperparameters during training.
- **Test Set**: Used only at the end to evaluate final model performance.
- **Label**: Target output for supervised learning tasks.

---

## GenAI & Transformer Concepts

- **Embedding**: A dense vector representing a word/token.
- **Embedding Dimension**: Number of features in the vector (e.g., 100, 768, 1536).
- **Positional Encoding**: Injects position information into token embeddings for transformers.
- **Attention Mechanism**: Weighs importance of different tokens in the input.
- **Self-Attention**: Token attends to all other tokens (including itself) to determine representation.
- **Masked Self-Attention**: Self-attention where tokens cannot see future tokens — A mask hides them to enforce auto-regressive behavior (in decoder).
- **Transformer**: Architecture that uses self-attention instead of recurrence.
- **Decoder-Only Model**: Predicts next tokens only (e.g., GPT).
- **Encoder-Decoder Model**: Processes input via encoder and generates output via decoder (e.g., T5, BART).
- **Model Fine-Tuning**: A pretrained model is further trained on a specific labeled dataset to adapt it to a specialized task or domain.
- **Auto-regressive**: A model generates one token at a time, using only past tokens to predict the next token.

---

## Text Generation Concepts

- **Temperature**: Controls randomness in text generation. (=0: Always picks top token (greedy), >1: More randomness in all tokens)
- **Top-k Sampling**: Selects only the k most probable tokens, discards the rest. (=1: Only picks top token (greedy), =10: Limits to 10 top tokens)
- **Top-p Sampling**: Includes the smallest set of tokens such that the sum of their probabilities ≥ p (e.g., 0.9).
- **Zero-shot / Few-shot Prompting**: Ask LLM to perform a task by giving zero / a few examples directly in the prompt.
- **Zero-shot / Few-shot Learning**: Training a model to generalize to new tasks using zero / very few labeled examples (update model parameters).
- **Positional Encoding**: A vector that encodes each token's position in the sequence. (Why? transformers are not sequential as RNN, so need to inject order information; e.g., "cat sat on mat" vs "mat on sat cat").

---

## Evaluation & Inference

- **Inference**: Using the trained model to make predictions on new data without updating the model.
- **Accuracy**: Fraction of correct predictions.
- **Precision**: True positives / (True positives + False positives).
- **Recall**: True positives / (True positives + False negatives).

---

## Model Training & Optimization

- **Model Parameters**: Learned by the model during training, directly affect the output.
- **Hyperparameters**: Set by the user before training and influence the training process and model's structure, but not directly the model's outputs. (e.g., learning rate, batch size, number of epochs).
- **Forward Pass**: Input flows through the network to compute predictions.
- **Loss Function**: Measures the difference between prediction and target (e.g., MSE).
- **Gradient**: Slope of the loss function.
- **Backpropagation**: Algorithm that computes the gradient (slope) of the loss with respect to each weight in a neural network.
- **Gradient Descent**: Optimization algorithm to update weights based on gradients.
- **Learning Rate**: Size of the update step in gradient descent. (Fast learning may make the model overshoot, oscillate, etc.)
- **Epoch**: One full iteration over the training dataset.

---

## Training Issues

- **Overfitting**: Model fits training data well but fails to generalize to new data. (Model could be too complex).
- **Underfitting**: Model fails to capture underlying patterns. (Model could be too simple).
- **Vanishing Gradients**: Gradients become too small → very slow learning / stalls.
- **Exploding Gradients**: Gradients become too large → unstable updates.
- **Non-convergence**: Model training fails to reach low loss. (Fast training rate).
- **Dead Neurons**: Neurons that output zero permanently.

---

## Learning Paradigms

- **Supervised Learning**: Learning from labeled data (input-output pairs).
- **Unsupervised Learning**: Learning from data without labels, finding hidden patterns. (e.g., Clustering News Articles—grouping similar articles by topic, without knowing the categories).
- **Reinforcement Learning**: Learning via rewards and punishments based on actions in an environment. (e.g., Autonomous Vehicles—A car learns to stay in the lane, avoid crashes, and follow traffic signals).

---

This glossary covers key terms that are crucial for understanding the GenAI concepts and the technologies behind them. 
