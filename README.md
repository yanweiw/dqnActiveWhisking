# DQN for active whisking

In a 2017 NIPS paper [Toward Goal-Driven Neural Network Models for the Rodent Whisker-Trigeminal System](https://arxiv.org/abs/1706.07555), several deep neural networks that input whisker array information such as bending and twisting are trained for shape detection.The best top-5 accuracy is 44.8%, while chance is 0.85%. One potential improvement on this performance is active sensing, as the current sensing data are obtained from passive sweeping the whisker array against objects. Incorporating active whiskering based on past sensing signals reflects how rats behave in real world and can potentially improve the efficiency and accuracy of shape detection.

### Model Abstraction

The goal is to learn an optimal controller for a whisker array in space around an object to collect observations in order to sequentially improve the estimation of the object shape in an efficient manner. As a starter, I have simplified the problem to a 2d classification of triangles and hexagons and also the whisker array to a radially outward positioned laser array. The idea is that rats can figure out the distance between contact points and their corresponding whisker base from whisker signals. So we are using Euclidean distances as classification input.

### DQN Algorithm

![DQN Algorithm](images/dqnAlgorithm.png)

### Results

On average the whisker array is able to move to most differentiating regions and collect observation that gives rise to a 90% confident classification within 10 steps.

![](images/hex_6step.png)
![](images/tri_6step.png)

## More details to come!

For more information, you can find me at my [portfolio page](https://yanweiw.github.io/).
Cover photo credit to [SeNSE Lab](http://journals.plos.org/ploscompbiol/issue?id=10.1371/issue.pcbi.v07.i04).




<!-- # My battle with SNEAKY bugs!

### Retrained DNN on single observation

![](images/modified_reward_training.png)

![](images/modified_reward1.png)

![](images/modified_reward2.png)

![](images/modified_reward3.png)

![](images/modified_reward4.png)

![](images/modified_reward5.png)

![](images/modified_reward6.png)

![](images/modified_reward7.png)

![](images/modified_reward8.png)

Caveat: Overfitting

![](images/reward_loss.png)

![](images/overfitting1.png)

### Shannon Entropy

![](images/converged_shannon1.png)

![](images/converged_shannon2.png)

![](images/converged_shannon3.png)

![](images/converged_shannon4.png)

![](images/converged_shannon5.png)

![](images/converged_shannon6.png)

![](images/converged_shannon7.png)

![](images/shannon_result1.png)

Outlier:

![](images/outlier1.png)

![](images/outlier2.png)

![](images/outlier3.png)

![](images/outlier4.png)

![](images/shannon_result2.png)







# The beautiful demos of learned agents hide all the blood, sweat, and tears that go into creating them.

![1](images/dqn_somewhat_converging.png)

![1](images/learn_to_stay.png)

![1](images/explore1.png)

![1](images/explore2.png)

![1](images/explore3.png)

![1](images/test1.png)

![1](images/test2.png)

![1](images/test3.png)

### 1. The data is too sparse

1. learned that all behavior results in randomness, and this knowledge is burned in
2. reward is not rich, consider multiply by a scaler
3. reduce the dimension of exploration
4. consider teleporting the rat head
5. maybe 3d structure solves the issue

### 2. The reward is flawed

1. Currently I am using binary cross-entropy loss as negative reward
2. maybe try only one reward at the terminal state

### 3. Inexact representation of state

1. I am using the hidden state vector of LSTM layer to approximate state
2. Maybe use some more direct state representation such as consecutive observations

### 4. Neural network not deep enough / hyperparameters

1. Current network trains for 20 min. Should I start looking at GPU?
2. exploration decay parameter


![1](images/tri_5.png)
![2](images/tri_10.png)
![3](images/hex_5.png)
![4](images/hex_10.png)

![5](images/train1.png)

![6](images/pred1.png)

![7](images/train2.png)

![8](images/pred2.png)

![9](images/pred3.png)

![10](images/pred4.png)

![11](images/pred5.png)

![12](images/pred6.png)

![13](images/lstm_states.png)

![14](images/hex_102_1.png)

![15](images/hex_102_2.png)
<!-- ![5](images/tri_5.png) -->
