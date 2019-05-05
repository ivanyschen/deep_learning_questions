### Week 1

**Q1. What does the analogy “AI is the new electricity” refer to?**  
  
A1. Similar to electricity starting about 100 years ago, AI is transforming multiple industries. 
  
**Q2. Which of these are reasons for Deep Learning recently taking off? (Check the three options that apply.)**
  
A2. 
  - Deep learning has resulted in significant improvements in important applications such as online advertising, speech recognition, and image recognition. 
  - We have access to a lot more data.
  - We have access to a lot more computational power. 
  
**Q3. Recall this diagram of iterating over different ML ideas. Which of the statements below are true? (Check all that apply.)**  

![](/img/wk1_img1.png)
 
A3. 
  - Being able to try out ideas quickly allows deep learning engineers to iterate more quickly. 
  - Faster computation can help speed up how long a team takes to iterate to a good idea. 
  - Recent progress in deep learning algorithms has allowed us to train good models faster (even without changing the CPU/GPU hardware). 
  
**Q4. When an experienced deep learning engineer works on a new problem, 
they can usually use insight from previous problems to train a good model on the first try, 
without needing to iterate multiple times through different models. True/False?** 

A4. False

**Q5. Which plot represents relu?**  
  
A5. 
  
**Q6. Images for cat recognition is an example of “structured” data, 
because it is represented as a structured array in a computer. True/False?**  
  
A6. False  
  
**Q7. A demographic dataset with statistics on different cities' population, GDP per capita, economic growth is an example of “unstructured” data 
because it contains data coming from different sources. True/False?**
  
A7. False  
  
**Q8. Why is an RNN (Recurrent Neural Network) used for machine translation, 
say translating English to French? (Check all that apply.)**

A8. 
  - It can be trained as a supervised learning problem. 
  - It is applicable when the input/output is a sequence (e.g., a sequence of words).  
  
**Q9. In this diagram which we hand-drew in lecture, 
what do the horizontal axis (x-axis) and vertical axis (y-axis) represent?**  

![](/img/wk1_img2.png)
A9. x-axis is the amount of data. y-axis (vertical axis) is the performance of the algorithm.  

**Q10. Assuming the trends described in the previous question's figure are accurate (and hoping you got the axis labels right), 
which of the following are true? (Check all that apply.)**
  
A10.
  - Increasing the size of a neural network generally does not hurt an algorithm’s performance, and it may help significantly.
  - Increasing the training set size generally does not hurt an algorithm’s performance, and it may help significantly. 
  
  
  
### Week 2: Neural Network Basics

**Q1. What does a neuron compute?**  
  
A1. A neuron computes a linear function (z = Wx + b) followed by an activation function
  
**Q2. Which of these is the "Logistic Loss"?**
  
A2. ![equation](https://latex.codecogs.com/gif.latex?\large&space;L(y_\text{pred}^{(i)},&space;y_\text{true}^{(i)})&space;=&space;y_\text{true}^{(i)}&space;\log{y_\text{pred}^{(i)}}&space;&plus;&space;(1-y_\text{true}^{(i)})&space;\log(1&space;-&space;y_\text{pred}^{(i)}))
  
**Q3. Suppose img is a (32,32,3) array, representing a 32x32 image with 3 color channels red, green and blue. How do you reshape this into a column vector?**. 
  
A3. x = img.reshape((32*32*3,1))  
  
**Q4. Consider the two following random arrays "a" and "b". What will be the shape of "c"?**
```
a = np.random.randn(2, 3) # a.shape = (2, 3)
b = np.random.randn(2, 1) # b.shape = (2, 1)
c = a + b
```
  
A4. c.shape = (2, 3)  
  
**Q5. Consider the two following random arrays "a" and "b". What will be the shape of "c"?**
```
a = np.random.randn(4, 3) # a.shape = (4, 3)
b = np.random.randn(3, 2) # b.shape = (3, 2)
c = a*b
```
  
A5. The computation cannot happen because the sizes don't match. It's going to be "Error"!
  
**Q6. Suppose you have n input features per example. Recall that X = [x(1)x(2)...x(m)] What is the dimension of X? **
  
A6. (n, m)  
  
**Q7. Consider the two following random arrays "a" and "b"**  
```
a = np.random.randn(12288, 150) # a.shape = (12288, 150)
b = np.random.randn(150, 45) # b.shape = (150, 45)
c = np.dot(a,b)
```
A7. (12288, 45)  
  
**Q8. How to vectorize the process?**  
```
# a.shape = (3,4)
# b.shape = (4,1)

for i in range(3):
  for j in range(4):
    c[i][j] = a[i][j] + b[j]
```
  
A8.
```
c = a + b.T
```

**Q9. Consider the following code. What will be c?**  
```
a = np.random.randn(3, 3)
b = np.random.randn(3, 1)
c = a * b
```
  
A9. This will invoke broadcasting, so b is copied three times to become (3,3), and \∗ is an element-wise product so c.shape will be (3, 3)  
  
Q10. Consider the following computation graph. What is the outpu J?  
  
![](/img/wk2_img1.png)
  
A10. J = (a - 1) * (b + c)
### Week 3: Shallow Neural Networks
**Q1. Which of the following are true? (Check all that apply)**  
  
A1.
  
**Q2. The tanh activation usually works better than sigmoid activation function for hidden units because the mean of its output is closer to zero, and so it centers the data better for the next layer. True/False?**

A2. True  
  
**Q3. **  
  
A3. 
  
**Q4. You are building a binary classifier for recognizing cucumbers (y=1) vs. watermelons (y=0). Which one of these activation functions would you recommend using for the output layer? **

A4. Sigmoid  
  
**Q5. Consider the following code. What will be B.shape?**
```
A = np.random.randn(4,3)
B = np.sum(A, axis = 1, keepdims = True)
```
  
A5. (A, 1)  
  
**Q6. Suppose you have built a neural network. You decide to initialize the weights and biases to be zero. Which of the following statements are True? (Check all that apply)**  
  
A6. Each neuron in the first hidden layer will perform the same computation. So even after multiple iterations of gradient descent each neuron in the layer will be computing the same thing as other neurons. 

**Q7. Logistic regression’s weights w should be initialized randomly rather than to all zeros, because if you initialize to all zeros, then logistic regression will fail to learn a useful decision boundary because it will fail to “break symmetry”, True/False?**

A7. False  
  
**Q8. You have built a network using the tanh activation for all the hidden units. You initialize the weights to relative large values, using np.random.randn(..,..)\*1000. What will happen? **

A8. This will cause the inputs of the tanh to also be very large, thus causing gradients to be close to zero. The optimization algorithm will thus become slow. 

### Week 4
