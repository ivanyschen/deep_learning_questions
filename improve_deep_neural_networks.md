### Week1 Practical aspects of deep learning
**Q1. If you have 10,000,000 examples, how would you split the train/dev/test set?**  
A1. 98% train, 1% dev, 1% test  
  
**Q2. The dev and test set should:**  
A2. 
- \[x] Come from the same distribution
- \[ ] Come from different distributions
- \[ ] Be identical to each other (same (x,y) pairs) 
- \[ ] Have the same number of examples 
  
**Q3. If your Neural Network model seems to have high bias, what of the following would be promising things to try? (Check all that apply.)**  
A3.  
- \[ ] Get more training data
- \[x] Increase the number of units in each hidden layer
- \[ ] Add regularization
- \[x] Make the Neural Network deeper
- \[ ] Get more test data  
  
**Q4. You are working on an automated check-out kiosk for a supermarket, and are building a classifier for apples, bananas and oranges. Suppose your classifier obtains a training set error of 0.5%, and a dev set error of 7%. Which of the following are promising things to try to improve your classifier? (Check all that apply.)**  
A4. 
- \[x] Increase the regularization parameter lambda
- \[ ] Decrease the regularization parameter lambda
- \[x] Get more training data
- \[ ] Use a bigger neural network
  
**Q5. What is weight decay**
A5.  
- \[ ] A technique to avoid vanishing gradient by imposing a ceiling on the values of the weights
- \[ ] Gradual corruption of the weights in the neural network if it is trained on noisy data. 
- \[ ] The process of gradually decreasing the learning rate during training. 
- \[x] A regularization technique (such as L2 regularization) that results in gradient descent shrinking the weights on every iteration.  
  
**Q6. What happens when you increase the regularization hyperparameter lambda?**  
A6.  
- \[x] Weights are pushed toward becoming smaller (closer to 0) 
- \[ ] Weights are pushed toward becoming bigger (further from 0)
- \[ ] Doubling lambda should roughly result in doubling the weights
- \[ ] Gradient descent taking bigger steps with each iteration (proportional to lambda)
  
**Q7. With the inverted dropout technique, at test time:**
A7. You do not apply dropout (do not randomly eliminate units) and do not keep the 1/keep_prob factor in the calculations used in training  
  
**Q8. Increasing the parameter keep_prob from (say) 0.5 to 0.6 will likely cause the following: (Check the two that apply)**
A8.  
- \[ ] Increasing the regularization effect
- \[x] Reducing the regularization effect
- \[ ] Causing the neural network to end up with a higher training set error
- \[x] Causing the neural network to end up with a lower training set error
  
**Q9. Which of these techniques are useful for reducing variance (reducing overfitting)? (Check all that apply.)**
A9.  
- \[ ] Vanishing gradient
- \[ ] Gradient Checking
- \[x] Data augmentation
- \[x] Dropout
- \[ ] Exploding gradient
- \[ ] Xavier initialization
- \[x] L2 regularization  
  
**Q10. Why do we normalize the inputs?**
A10. It makes the cost function faster to optimize  
  
### Week2 Optimization algorithms
**Q2. Which of these statements about mini-batch gradient descent do you agree with?**  
A2.  
- \[x] One iteration of mini-batch gradient descent (computing on a single mini-batch) is faster than one iteration of batch gradient descent.  
- \[ ] You should implement mini-batch gradient descent without an explicit for-loop over different mini-batches, so that the algorithm processes all mini-batches at the same time (vectorization).  
- \[ ] Training one epoch (one pass through the training set) using mini-batch gradient descent is faster than training one epoch using batch gradient descent.  
  
**Q3. Why is the best mini-batch size usually not 1 and not m, but instead something in-between?**  
A3.  
- \[x] If the mini-batch size is 1, you lose the benefits of vectorization across examples in the mini-batch.
- \[x] If the mini-batch size is m, you end up with batch gradient descent, which has to process the whole training set before making progress. 
- \[ ] If the mini-batch size is m, you end up with stochastic gradient descent, which is usually slower than mini-batch gradient descent. 
- \[ ] If the mini-batch size is 1, you end up having to process the entire training set before making any progress.  
  
**Q4. Suppose your learning algorithm’s cost JJJ, plotted as a function of the number of iterations, looks like this. Which of the following do you agree with**  
![]()  
A4.  
- \[ ] Whether you’re using batch gradient descent or mini-batch gradient descent, this looks acceptable. 
- \[ ] If you’re using mini-batch gradient descent, something is wrong. But if you’re using batch gradient descent, this looks acceptable. 
- \[ ] Whether you’re using batch gradient descent or mini-batch gradient descent, something is wrong. 
- \[x] If you’re using mini-batch gradient descent, this looks acceptable. But if you’re using batch gradient descent, something is wrong.  
  
**Q5. \theta_1 = 10, \theta_2 = 10, v_0 = 0, beta = 0.5. Calculate v_2 and v_2^{corrected}**  
A5.  
v_2 = 7.5  
v_2^{corrected} = 10  
  
**Q6. Which of these is NOT a good learning rate decay scheme? Here, t is the epoch number.**  
A6. 
- \[ ] \alpha = \frac{1}{1 + 2 * t} * \alpha_0
- \[ ] \alpha = \frac{1}{t^{1/2}} * \alpha_0
- \[x] \alpha = e^{t} * \alpha_0
- \[ ] \alpha = 0.95^{t} * \alpha_0  
  
**Q7. You use an exponentially weighted average on the London temperature dataset. You use the following to track the temperature: v_t=\beta v_{t−1}+(1−\beta)* \theta_t = The red line below was computed using beta=0.9. What would happen to your red curve as you vary β\betaβ? (Check the two that apply)**  
A7.  
!()[]
- \[ ] Decreasing \beta will shift the red line slightly to the right.
- \[x] Increasing \beta will shift the red line slightly to the right.
- \[x] Decreasing \beta will create more oscillation within the red line.
- \[ ] Increasing \beta will create more oscillations within the red line.
### Week3
