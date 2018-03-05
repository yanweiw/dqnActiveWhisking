# The beautiful demos of learned agents hide all the blood, sweat, and tears that go into creating them.

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
