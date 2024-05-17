# 圖形識別實務與應用- Final

### 目標

用 ORL_PCA_MaxMin_Data.mat 做 BackProbagation + one hot encoding + cross entropy

### ORL_PCA_MaxMin_Data.mat

TrainORL: 200 * 65dim

### Variable Dimensions

| hidden_matrix | 65 | 110 |
| --- | --- | --- |
| hidden_bias | 1 | 110 |
| out_matrix | 110 | 40 |
| out_bias | 1 | 40 |

| x | 1 | 65 |
| --- | --- | --- |
| hidden_sigma/hidden_net | 1 | 110 |
| output_sigma/output_net=y | 1 | 40 |
| y_ | 1 | 40 |
| ce | 1 | 40 |

| d_output_net | 40 | 1 |
| --- | --- | --- |
| delta_output | 40 | 1 |
| d_hidden_net | 110 | 1 |
| delta_hidden | 110 | 1 |

Ref.

[【TensorFlow】tf.nn.softmax_cross_entropy_with_logits的用法-CSDN博客](https://blog.csdn.net/mao_xiao_feng/article/details/53382790)

[Derivative of the Softmax Function and the Categorical Cross-Entropy Loss](https://towardsdatascience.com/derivative-of-the-softmax-function-and-the-categorical-cross-entropy-loss-ffceefc081d1)