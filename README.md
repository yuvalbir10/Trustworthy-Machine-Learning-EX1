# Q1

1. I completed the ```compute_accuracy``` code as asked, and the printed accuracy was:
```The test accuracy of the model is: 0.8750```

2. \[ X_{i+1} = \text{Project}(X_i + a \nabla_x \text{sign}(loss_f(X, Cx))) \]

TODO: i changed the generated adv examples by the mean of the gradients in each batch. should i do it seperately for each x in the batch?


3. As described in the lecture, for each input we will estimate the gradients using R s.t
we will take \[
  R \sim \mathcal{N}(\mu,\,\sigma^{2})\,.
\] 
![alt text](image.png)

TODO: the excute function currently gets batch of images but does not do anything with the batching. maybe i need to run it for each image in the batch seperately

so that we will have the estimated derivative as the average incline.



# TODO:
- remove all prints from the code
- fix all 'TODO's in the code
- in 1.3 i need to understand if i need to estimate the gradient for each input alone or all together, because in 1.2 i used the avg loss of all the inputs and took the gradient from this avg.
