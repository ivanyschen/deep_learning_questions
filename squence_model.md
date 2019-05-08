### Week 1 Recureent Neural Networks

**Q1. Suppose your training examples are sentences (sequences of words). 
Which of the following refers to the jthj^{th}jth word in the ithi^{th}ith training example?**

A1. <img src="https://latex.codecogs.com/gif.latex?\inline&space;x^{(i)<j>}" title="x^{(i)<j>}" />

**Q2. Consider this RNN. THe specific type of architecture is appropriate when:**

![](/img/sequence_model/wk1_img1.png)

A2. <img src="https://latex.codecogs.com/gif.latex?\inline&space;T_x&space;=&space;T_y" title="T_x = T_y" />

**Q3. To which of these tasks would you apply a many-to-one RNN architecture? (Check all that apply).**

A3. 
- \[ ] Speech recognition (input an audio clip and output a transcript) 
- \[x] Sentiment classification (input a piece of text and output a 0/1 to denote positive or negative sentiment)
- \[ ] Image classification (input an image and output a label)
- \[x] Gender recognition from speech (input an audio clip and output a label indicating the speaker’s gender) 

**Q4. You are training this RNN language model. At the ttht^{th}tth time step, what is the RNN doing? Choose the best answer. **

![](/img/sequence_model/wk1_img2.png)

A4. Estimating <img src="https://latex.codecogs.com/gif.latex?\inline&space;P(y^{<t>}|y^{<1>},&space;y^{<2>},...,&space;y^{<t-1>})" title="P(y^{<t>}|y^{<1>}, y^{<2>},..., y^{<t-1>})" />

**Q5. You have finished training a language model RNN and are using it to sample random sentences, as follows. What are you doing at each time step t?**

A5. (i) Use the probabilities output by the RNN to randomly sample a chosen word for that time-step as <img src="https://latex.codecogs.com/gif.latex?\inline&space;\hat{y}^{<t>}" title="\hat{y}^{<t>}" />. (ii) Then pass this selected word to the next time-step.

**Q6. You are training an RNN, and find that your weights and activations are all taking on the value of NaN (“Not a Number”). Which of these is the most likely cause of this problem?**

A6. Exploding gradient problem.


**Q7. Suppose you are training a LSTM. You have a 10000 word vocabulary, and are using an LSTM with 100-dimensional activations <img src="https://latex.codecogs.com/gif.latex?\inline&space;a^{<t>}" title="a^{<t>}" />. What is the dimension of <img src="https://latex.codecogs.com/gif.latex?\inline&space;\Gamma&space;_u" title="\Gamma _u" /> at each time step?**

A7. 100

**Q8. Here’re the update equations for the GRU. Alice proposes to simplify the GRU by always removing the Γu. I.e., setting Γu = 1. Betty proposes to simplify the GRU by removing the Γr. I. e., setting Γr = 1 always. Which of these models is more likely to work without vanishing gradient problems even when trained on very long input sequences?**
![](/img/sequence_model/wk1_img3.png)

A8.
- [ ] Alice’s model (removing Γu), because if Γr≈0 for a timestep, the gradient can propagate back through that timestep without much decay. 
- [ ] Alice’s model (removing Γu), because if Γr≈1 for a timestep, the gradient can propagate back through that timestep without much decay. 
- [x] Betty’s model (removing Γr), because if Γu≈0 for a timestep, the gradient can propagate back through that timestep without much decay. 
- [ ] Betty’s model (removing Γr), because if Γu≈1 for a timestep, the gradient can propagate back through that timestep without much decay. 

**Q9. Here are the equations for the GRU and the LSTM: From these, we can see that the Update Gate and Forget Gate in the LSTM play a role similar to _______ and ______ in the GRU. What should go in the the blanks?**

![](/img/sequence_model/wk1_img4.png)

A9. Γu and 1−Γu

**Q10. You have a pet dog whose mood is heavily dependent on the current and past few days’ weather. You’ve collected data for the past 365 days on the weather, which you represent as a sequence as x<1>,…,x<365>. You’ve also collected data on your dog’s mood, which you represent as y<1>,…,y<365>. You’d like to build a model to map from <img src="https://latex.codecogs.com/gif.latex?\inline&space;x\rightarrow&space;y" title="x\rightarrow y" />. Should you use a Unidirectional RNN or Bidirectional RNN for this problem?**

A10. Unidirectional RNN, because the value of y<t>y^{<t>}y<t> depends only on x<1>,…,x<t>, but not on x<t+1>,…,x<365>


### Week 2 Natural Language Processing & Word Embeddings

**Q1. Suppose you learn a word embedding for a vocabulary of 10000 words. Then the embedding vectors should be 10000 dimensional, so as to capture the full range of variation and meaning in those words.**

A1. False

**Q2. What is t-SNE?**

A2. A non-linear dimensionality reduction technique

**Q3. Suppose you download a pre-trained word embedding which has been trained on a huge corpus of text. You then use this word embedding to train an RNN for a language task of recognizing if someone is happy from a short snippet of text, using a small training set.
Then even if the word “ecstatic” does not appear in your small training set, your RNN might reasonably be expected to recognize “I’m ecstatic” as deserving a label y=1.**

A3. True

**Q4. Which of these equations do you think should hold for a good word embedding? (Check all that apply) **

A4. 
- \[x] e_{boy} - e{girl} ~ e_{brother} - e_{sister}
- \[ ] e_{boy} - e{girl} ~ e_{sister} - e_{brother}
- \[x] e_{boy} - e_{brother} ~ e_{girl} - e_{sister}
- \[ ] e_{boy} - e_{brother} ~ e_{sister} - e_{girl}

**Q5. Let EEE be an embedding matrix, and let e1234e_{1234} be a one-hot vector corresponding to word 1234. Then to get the embedding of word 1234, why don’t we call E * e_{1234} in Python?**

A5. It is computationally wasteful.

**Q6. When learning word embeddings, we create an artificial task of estimating P(target∣context). It is okay if we do poorly on this artificial prediction task; the more important by-product of this task is that we learn a useful set of word embeddings. **

A6. True

**Q7. In the word2vec algorithm, you estimate P(t∣c), where ttt is the target word and ccc is a context word. How are ttt and ccc chosen from the training set? Pick the best answer.**

A7. c and t are chosen to be nearby words.

**Q8. Suppose you have a 10000 word vocabulary, and are learning 500-dimensional word embeddings. The word2vec model uses the following softmax function:**

![](/img/sequence_model/wk2_img1.png)

A8. 
- \[x] \theta_{t} and e_{c} are both 500 dimensional vectors.
- \[ ] \theta_{t} and e_{c} are both 10000 dimensional vectors.
- \[x] \theta_{t} and e_{c} are both trained with an optimization algorithm such as Adam or gradient descent.
- \[ ] After training, we should expect theta_{t} to be very close to e_c when t and c are the same word. 


**Q9. Suppose you have a 10000 word vocabulary, and are learning 500-dimensional word embeddings.The GloVe model minimizes this objective:**

![](/img/sequence_model/wk2_img2.png)

A9.
- \[ ] \theta_{i} and e_{j} should be initialized to 0 at the beginning of training
- \[x] \theta_{i} and e_{j} should be initialized randomly at the beginning of training.
- \[x] X_{ij} is the number of times word i appears in the context of word j.
- \[x] The weighting function f(.) must satisfy f(0)=0.

**Q10. You have trained word embeddings using a text dataset of m1 words. You are considering using these word embeddings for a language task, for which you have a separate labeled dataset of m2 words. Keeping in mind that using word embeddings is a form of transfer learning, under which of these circumstance would you expect the word embeddings to be helpful?**

A10. m1 >> m2
