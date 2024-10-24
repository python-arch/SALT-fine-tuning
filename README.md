# SALT - Singular value Adaptation with Low-rank Transformation

## Overview of SVD and LoRA

In standard Singular Value Decomposition (SVD), the weight matrix is decomposed into:

- Left singular vectors,
- Singular values (a diagonal matrix),
- Right singular vectors.

In the Low-Rank Adaptation (LoRA) technique, the weight matrix is updated by introducing two low-rank matrices:

- Learnable low-rank matrices are added, where the rank of these matrices is much smaller than the dimensions of the original weight matrix.

## Motivation

In SVD, singular values represent the "strength" or "importance" of the corresponding basis vectors. This is often applied in data compression, such as image compression, where the image can be represented by a combination of singular vectors. The contribution of each vector is determined by its corresponding singular value, allowing for effective compression by retaining only the most significant singular values.

To achieve a trade-off between full-rank computation and the number of trainable parameters, we can:

1. Truncate the singular values from the SVD decomposition by keeping only the most significant ones.
2. Introduce learnable low-rank matrices to add more trainable parameters to the model.

Additionally, a hyperparameter can be introduced to control the balance between full-rank computation and the number of additional parameters introduced by the low-rank matrices.

## Truncated SVD with LoRA

The goal is to fine-tune the model using truncated SVD and apply a LoRA-like low-rank update. First, we perform truncated SVD on the weight matrix, keeping the top singular values.

Next, a low-rank adaptation is applied by adding two learnable low-rank matrices, allowing the model to adjust without having to reconstruct the entire weight matrix.

## Parameter Efficiency and Flexibility

By using truncated SVD and LoRA-like updates, the number of parameters involved in fine-tuning is limited:

- In SVD fine-tuning, only the singular values are adjusted.
- In the LoRA-like update, additional transformations are captured by introducing low-rank matrices.

This approach provides flexibility in both retaining the top singular values and adding more trainable parameters to better fit the data.

## Trade-off Between Rank and Trainable Parameters

By adjusting the rank of truncated SVD and the rank of the LoRA matrices, we can control the trade-off between model capacity and the number of trainable parameters. Specifically:

- A higher SVD rank retains more information from the original matrix.
- A higher LoRA rank introduces more parameters during fine-tuning, increasing the model's expressiveness.

The total number of trainable parameters is determined by both the rank of SVD and LoRA, offering flexibility in model adaptation without excessive computational costs.

## More Details?
Refer to this [SALT- In a more formal way](SALT.pdf)

