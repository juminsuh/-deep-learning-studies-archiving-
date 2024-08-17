# 16주차

태그: 완료

## Basic of NLP

Natural language processing (NLP), which aims at properly understanding and generating human languages, emerges as a crucial appliation of AI with the advancements of DNN. NLP can be applied to language modeling (predict next words based on former context), machine translation, question answering, document classification, and dialog system.

**NLP**'s major conferences are ACL, ENMNLP, NAACL. There are various fields within NLP.

- Low-level parsing
    - Tokenization: For example, in sentence "I study math", each word can be a token. Process of breaking sentence into tokens is called tokenization.
    - Stemming: Word "study" has diverse variation like "studies", "studied", and so on. Process of extracting the root that holds the meaning of a word.
- Word and phrase level
    - NER(Named Entity Recognition): Recognize proper nouns. For example, "NewYork Times" should be interpreted in a single proper noun.
    - POS(Part-Of-Speech) tagging: Identify the parts of speech of sentence components.
- Sentence level
    - Sentiment Analysis: Analyze whether given sentence is positive or negative.
    - Machine Translation: Translate English sentence into Korean sentence and vice versa, considering proper translation of words, order of words, and so on.
- Multi-sentence and paragraph level
    - Entailment Prediction: Process of predict a logical structure within a sentence and reveal contradictions between sentences. For example, "John married yesterday" and "One of them married yesterday" are compatible. However, "John married yesterday" and "No one married yesterday" cannot be compatible.
    - Question Answering: Understand a question and provide an answer precisely.
    - Dialog Systems: Conversation, Chat Bot
    - Summarization: Automatically summarize documents.

**Text mining**'s major conferences are KDD, The WebConf, WSDM, CIKM, and ICWSM. These are various fields of text mning.

- Extract useful information and insights from text and document data. For example, analyzing the trends of music from massive news data.
- Document clustering like topic modeling. For example, clustering news data and grouping into different subjects.
- Highly related to computational social science. It statically yields social science insights by analyzing texts. For example, analyzing the evolution of people's political tendency based on social media data.

**Information retrieval**'s major conferences are SIGIR, WSDM, CIKM, and RecSys.

- It is highly related to computational social science. This area is not that actively studied now. However, recommendation system has evolved very active and automatic field as one of the information retrieval. Before user retrieves something, RecSys recommends keyword that might be preferred by the users.

### Trends of NLP

NLP is one of the fields that has developed with significant attention alongside Computer Vision.

In order to train NLP model, each word in text data should be represented as a vector through a technique such as Word2Vec or GloVe. This process is called **Word Embedding**.

Since sequence information of words consisting a sentence is important, **RNN** (Recurrent Neural Network) is used. **LSTM** (Long Short-Term Memory) and **GRU** (Gated Recurrent Unit) can complement RNN. These RNN-family model, which take the sequence of these vectors of words as input, are the main architectures of NLP tasks.

Overall performance of NLP tasks have been improved with **attention modeules and transformer model**. In 2017, Google published a paper named "Attention is all YOU need". Transformers with self-attention structures replaced RNNs. Transformer models are utilized in machine translation, video, development of medicine, time-series prediction, and so on.

Since transformer was introduced, models become larger by stacking its basic module, self-attention, and these models are trained with large-sized text datasets. **Self-supervised training** is a training that does not require additional labels for a particular task, but rather a model predicts those labels itself. For example, BERT and GPT-3 perform a self-supervised training. With a model trained by self-supervised training, without any notable modifications, it can outperform to different NLP tasks. This is called a **transfer learning**.

Currently, since NLP training requires large models and large datasets, large amounts of GPU are needed. Thus, major companies like Google and OpenAI, who have large amounts of sources, are leading NLP industries.

## Bag-Of-Words & Naive Bayes Classifier

### 1. Bag-Of-Words Representation

Bag-Of-Words is a method to represent text with number which does not consider the order of words but only frequency of words.

In this example, we register 8 words into vocabulary list. Repeated word like "this" and "really" are allowed to be counted only once. Since each word is a categorical variable, we can represent each word as vector with eight dimensions, using one-hot encoding. We can represent one sentence by summing words' one-hot vector. In this case, "John really really loves this movie" can be represented as [1 2 1 1 1 0 0 0]. However, for any pair of words, the Euclidean distance is all root of 2 and cosine similarity is 0. Thus, we cannot obtain any information about order, importance, and relations between words.

![https://blog.kakaocdn.net/dn/bjpmiT/btsH5VCPNoc/BwO89wB5rQpLKx4okD2Au1/img.png](https://blog.kakaocdn.net/dn/bjpmiT/btsH5VCPNoc/BwO89wB5rQpLKx4okD2Au1/img.png)

![https://blog.kakaocdn.net/dn/ofj1r/btsH5WBINVL/yp6JhzEx28E0upKv2TfCuk/img.png](https://blog.kakaocdn.net/dn/ofj1r/btsH5WBINVL/yp6JhzEx28E0upKv2TfCuk/img.png)

### 2. Naive Bayes Classifier

Naive Bayes classifier uses Bayes' theorem. 𝑃(𝑐|𝑑)=𝑃(𝑑|𝑐)𝑃(𝑐)/𝑃(𝑑)

𝐶𝑀𝐴𝑃 is Maximum A Posteriori, which means the most likely class c to given document d. Since denominator P(d) is a fixed value, we consider it as a constant and ignore this when calculating 𝐶𝑀𝐴𝑃.

![https://blog.kakaocdn.net/dn/dnIoa5/btsH7Ubau3Z/9caTv0XzEXjk5JWknGnas1/img.png](https://blog.kakaocdn.net/dn/dnIoa5/btsH7Ubau3Z/9caTv0XzEXjk5JWknGnas1/img.png)

For a document d, which consists of a sequence of words w and a class c. The probability of a document can be represented by multiplying the probability of each word appearing. 𝑃(𝑑|𝑐)𝑃(𝑐)=𝑃(𝑤1,𝑤2...|𝑐)𝑃(𝑐)=𝑃(𝑐)∏𝑤𝑖𝑃(𝑤𝑖|𝑐). For example, there are two CV documents and two NLP documents and we want to classify document 5.

![https://blog.kakaocdn.net/dn/VaoMP/btsH5uZ0EX0/1CnSHbhxFGusZcMcU2ZeF0/img.png](https://blog.kakaocdn.net/dn/VaoMP/btsH5uZ0EX0/1CnSHbhxFGusZcMcU2ZeF0/img.png)

𝑃(𝑐𝑐𝑣)=24=12,𝑃(𝑐𝑁𝐿𝑃)=24=12

![https://blog.kakaocdn.net/dn/JScja/btsH5I4L7Z4/pGravpSrILMAHHkF0VXJM1/img.png](https://blog.kakaocdn.net/dn/JScja/btsH5I4L7Z4/pGravpSrILMAHHkF0VXJM1/img.png)

![https://blog.kakaocdn.net/dn/eee5SD/btsH6TdguH4/55wNRy0Q2efZOXOKGeSRoK/img.png](https://blog.kakaocdn.net/dn/eee5SD/btsH6TdguH4/55wNRy0Q2efZOXOKGeSRoK/img.png)

Thus, document 5 will be classified as class "NLP".

Naive Bayes classifier also works to more than two classes. However, this method has a critical disadvantage. If one word does not present in training documents, even though other words have high probability, the total probability will be zero. To solve this, we can use regularization term.

## Word Embedding - Word2Vec & GloVe

Word Embedding is a method to represent each word as optimal vector (one point at the 3D space) onto the coordinate space. Then, how can we know which is the optimal vector? Let's take a simple example.

There are words, "kitty", "cat", and "hamburger". If we represent these words as vectors onto the coordinate space, "kitty" and "cat" will be located closely while "hamburger" will be located far from these two words. Thus, cosine similarity between two vectors will be high if two words have similar meaning.

### 1. Word2Vec

### 1. Idea of Word2Vec

Then, how can we train Word2Vec algorithm? Word2Vec is an algorithm for training vector representation of a word from context words. It assumes that words in similar context will have similar meanings. For example, in two sentences "The cat purrs" and "This cat hunts mice", "the" and "this" are words that both modify "cat" and "purrs" and "hurts" are both actions of "cat".

Model predicts a probability distribution of surrounding words for a given word. For example, if a word "cat" is given, then the probability of "meow" and "pet" will be high.

![https://blog.kakaocdn.net/dn/Gdgqr/btsH8FTSxIZ/xvDEXyOU0ktEi1maTVnaek/img.png](https://blog.kakaocdn.net/dn/Gdgqr/btsH8FTSxIZ/xvDEXyOU0ktEi1maTVnaek/img.png)

### 2. Computation of Word2Vec

Also, through neighbor words, meaning of center word can be represented. First, we can construct pair of words, using **sliding window** method. Sliding window method predicts a meaning of a center word by looking former and latter words of it. For example, if the size of window is 3,  in "I study math", if "I" is a center word, then we can first construct (I, study) pair. If window moves to left, then "study" becomes a center word. We can obtain (study, I), (study, math). We can use these pairs as trainging set.

![https://blog.kakaocdn.net/dn/yFV3w/btsH8fBn478/KeUHSkT79BDogwogIK7Zk1/img.png](https://blog.kakaocdn.net/dn/yFV3w/btsH8fBn478/KeUHSkT79BDogwogIK7Zk1/img.png)

The number of input and output are both 3 in this case, which is the size of vocabulary. The number of node in hidden layer is a hyperparameter and is 2 in this case. Let's take an example by using (study, math) pair as input. "study" is the actual input data and "math" is a ground-truth that this neural network should predict.

**Overall process**

1. "study" is [0 1 0] when it is represented as one-hot vector.
2. The shape of W1 is (2, 3) and that of W2 is (3, 2) for matrix calculation.
3. These parameters are trained to predict result as "math" correctly.

Looking more precisely, since x is a one-hot vector, only column colored blue will be used in W1. Thus, the first linear transformation (W1x) is a process of extracting a column of W1 that corresponds to given one-hot vector. Especially, this layer is called embedding layer.

In this case, when W2(W1x) passes through softmax, it predicts "math" with 100%. The parameters in W1 and W2 are trained to maximize the similarity of given ground-truth while minimize that of other words.

**Further example**

![https://blog.kakaocdn.net/dn/bfRwlB/btsH9pbIyqn/uksCG1ajCD64GxydbkLlwk/img.png](https://blog.kakaocdn.net/dn/bfRwlB/btsH9pbIyqn/uksCG1ajCD64GxydbkLlwk/img.png)

In this neural network, if training example is given as (eat, apple). Then, as "eat" is given as an input, neural network are trained to maximize the probability of "apple", using gradient descent. Since we embedded input as 5 dimensional vector, the input vector and output vector are represented as 8x5 matrix.

**Result**

- The emedded (input) vectors of "juice", "milk", "water" are similiar. Also, output vector of "drink" is similar to those.
- The embedded (input) vectors of "apple", "orange" are similar. Also, output vector of "eat" is similar to those.

![https://blog.kakaocdn.net/dn/bODq0f/btsH8ASL6EH/VaMYkTOQWiXX1L7v2O5fAK/img.png](https://blog.kakaocdn.net/dn/bODq0f/btsH8ASL6EH/VaMYkTOQWiXX1L7v2O5fAK/img.png)

![https://blog.kakaocdn.net/dn/pV133/btsH97VTE9x/tG64EH3NbEA5k7qnU7knS0/img.png](https://blog.kakaocdn.net/dn/pV133/btsH97VTE9x/tG64EH3NbEA5k7qnU7knS0/img.png)

The above graph transformed 5 dimensional vector into 2 dimensional vector using PCA. "milk", "juice", "water", "drink" are located closely. "milk", "juice", "water", "drink" are located closely.

### 3. Characteristics of Word2Vec

The word vector represents the relationship between the words. The same relationship is represented as the same vector.

For example, vec[queen]-vec[king] and vec[woman]-vec[man] have same vector. This shows that a model trained the meaning of words and relationship between words well.

![https://blog.kakaocdn.net/dn/nQi47/btsH8xIB2Nd/g5eOreKwjf72kDufT8cl7K/img.png](https://blog.kakaocdn.net/dn/nQi47/btsH8xIB2Nd/g5eOreKwjf72kDufT8cl7K/img.png)

### 4. Application of Word2Vec

- Word intrusion detection
    - Choose a word that has the most different meaning to other words, using average of Euclidean distance between words
    - ex) math shopping reading science
- Word2Vec is meaningful itself, it is widely used in diverse tasks.
    - In machine translation, it learns similarity between words and improve its translation performance.
    - In sentiment analysis, it helps emotion analysis and classification of positive/negative.
    - In image captioning, it helps tasks which extract features of images and represent them as sentence.

### 2. GloVe: Global Vectors for Word Representation

The biggest difference between Word2Vec and GloVe is that in advance, GloVe computes the probability 𝑃𝑖𝑗 of co-occurence of word i and word j within the same window. GloVe is trained to minimize this loss function.

𝐽(𝜃)=12∑𝑖,𝑗=1𝑊𝑓(𝑃𝑖𝑗)(𝑢𝑖𝑇𝑣𝑗−𝑙𝑜𝑔𝑃𝑖𝑗)**2

f(𝑃𝑖𝑗) is a weight function to adjust the weights of word pairs that have very low or very high frequencies.

In Word2Vec, there might be duplicated values in training set. Thus, identical word pairs might be trained repetitively. Since GloVe avoids this, the training speed is fast and it works well even with a small corpus.

## RNN

In practice, sequence data are given as input and output of RNN. For each time step t, new input 𝑥𝑡 and former computed hidden state ℎ𝑡−1 are given to A module and ℎ𝑡 will be an output. For example, in "I study math", "I", "study", "math" will be input 𝑥𝑡 for each time step t.

Module A is called recursively for each time step t. Thus, A module uses output of former A module as input.

Also, at some time step t, we might want to compute 𝑦𝑡. For example, if we want to know the part of speech, output will be computed at every time step.

![https://blog.kakaocdn.net/dn/dpZTKH/btsH7Vwrw0W/RZS0w5TmljmsnD6RLIdLhK/img.png](https://blog.kakaocdn.net/dn/dpZTKH/btsH7Vwrw0W/RZS0w5TmljmsnD6RLIdLhK/img.png)

![https://blog.kakaocdn.net/dn/DcCID/btsH8xu33qR/sjNKcuTm7U7ksvfPdK4l20/img.png](https://blog.kakaocdn.net/dn/DcCID/btsH8xu33qR/sjNKcuTm7U7ksvfPdK4l20/img.png)

left is unrolled RNN. right is rolled RNN.

### Computation of RNN

Let's see the structure of RNN more specifically.

ℎ𝑡=𝑓𝑊(ℎ𝑡−1,𝑥𝑡)

- ℎ𝑡−1: old hidden-state vector
- 𝑥𝑡: input vector at time step t
- ℎ𝑡: new hidden_state vector
- 𝑓𝑊: RNN function with parameter W
- 𝑦𝑡 output vector at time step t

One of the most important properties of RNN is that parameter W of f is shared through all time steps. These parameters are updated as learning progresses.

Let's assume that the dimension of 𝑥𝑡 is 3 and that of ℎ𝑡−1 is 2. Then we can stack them like in below figure.  Also, we can compute ℎ𝑡 and 𝑦𝑡 using below equations.

ℎ𝑡=𝑡𝑎𝑛ℎ(𝑊ℎℎℎ𝑡−1+𝑊𝑥ℎ𝑥𝑡)

𝑦𝑡=𝑊ℎ𝑦ℎ𝑡

By considering notation, we can identify that 𝑊ℎℎ is a weight matrix that transforms ℎ𝑡−1 into ℎ𝑡, 𝑊𝑥ℎ transforms 𝑥𝑡 into ℎ𝑡, 𝑊ℎ𝑦 transforms ℎ𝑡 into 𝑦𝑡.

![https://blog.kakaocdn.net/dn/cpwLkc/btsH94Zbx6b/9tgvKs4vKnAzRT4L71fYvK/img.png](https://blog.kakaocdn.net/dn/cpwLkc/btsH94Zbx6b/9tgvKs4vKnAzRT4L71fYvK/img.png)

### Types of RNN

![https://blog.kakaocdn.net/dn/ceAoKn/btsH7XgLd6y/FmKQkz3tp8udAkvhZHtZ8k/img.png](https://blog.kakaocdn.net/dn/ceAoKn/btsH7XgLd6y/FmKQkz3tp8udAkvhZHtZ8k/img.png)

- **one to one**: Standard neural net that uses data which is not sequential.
- **one to many**: Single input, multiple outputs
    - For 2nd and 3rd input, 0 vectors that have same dimension with 1st input are given.
    - ex) Image Captioning: Single image as input and multiple words for captioning as output
- **many to one**: Sequential input and a sinlge output at the final time step
    - ex) Sentiment Classification: "I love movie" as input and "positive" as output
- **many to many**
    - ex) machine translation: "I go home" as input and "나는 집에 간다" as output
    - Just like in the figure, the model reads the entire given sentence up to the 3rd hidden state, and then translates it into Korean from that point onward.
- **many to many**
    - Different to former many to many, it processes input without any delay.
    - ex) Video Classification on Frame Level: each frame as input and classification of that frame as output

## Character-level Language Model

Language model predicts the next word based on given sequences of words.

### Process of Character-level Language Model

Let's assume that there is only a word "hello". First, we have to build a vocabulary that includes only unique words which means that there are no duplicated words in one vocabulary. Thus, a vocabulary built from "hello" is [h, e, l, o]. We can represent each character of vocabulary with one-hot vector.

![https://blog.kakaocdn.net/dn/zzB4F/btsH71cobrf/Mv4uZsKEElBy1HMarDBl10/img.png](https://blog.kakaocdn.net/dn/zzB4F/btsH71cobrf/Mv4uZsKEElBy1HMarDBl10/img.png)

We can implement a task that predict next character based on a given character, using RNN. For example, if 'h' is given, then a model should predict next character, 'e'. Referred to previous post, we can compute ℎ𝑡 through below equation.

ℎ𝑡=𝑡𝑎𝑛ℎ(𝑊ℎℎℎ𝑡−1+𝑊𝑥ℎ𝑥𝑡+𝑏)

Let's assume that dimension of hidden state vector is 3. Since the first hidden layer does not have a previous hidden state, just put a default vector [0 0 0]. We can get predicted chars for each step by using below equation.

𝐿𝑜𝑔𝑖𝑡=𝑊ℎ𝑦ℎ𝑡+𝑏

If we compute logit, we can obtain 4 dimensional vector as an output. Then, put this vector as an input of softmax classifier. In this case, 𝑦1 will be predicted as 'o'. However, the ground-truth is 'e'. Thus, by using one-hot vector of target chars, which are ground-truths, a model is trained to maximize the probability of 'e'. The predicted char will be used as a input for next step.

### Application of Chatacter-level Language Model

1. We can predict long sequence data by using RNN. For example, predicting future stock price. If stock price of first day is given, then it will predict that of second day. If we implement this process recursively, we can predict further future's stock price with using same RNN model.
2. If we designate spaces, commas, and newline as special characters, even long texts can still be considered as one-dimensional sequence data. As shown in below figure, as learning is progressed, a model yields complete form of sentence.
3. Various linguistic features can be learned to generate text in cases such as character dialogues, LaTex-written papaers, and programming languages.

![https://blog.kakaocdn.net/dn/dOxUgT/btsH89fZs0O/osK3qkymTYN1zAKTrDJ1p1/img.png](https://blog.kakaocdn.net/dn/dOxUgT/btsH89fZs0O/osK3qkymTYN1zAKTrDJ1p1/img.png)

learning of RNN

## Backpropagation through time and Long-term dependency

**BPTT, Backpropagation through time** is a learning method to update weights based on computed gradients. If we repeat BPTT, a model will be trained the important parts of sequence data well.

There are parameter metrices in RNN like 𝑊ℎℎ, \(W_{xh)\), and 𝑊ℎ𝑦. If sequence data becomes larger and larger, it is very hard to compute all of ℎ𝑡, these parameter metrics, and loss to perform backpropagation with limited GPU. Therefore, we have to use **truncation**. Truncation is to slice a sequence to use in training.

![https://blog.kakaocdn.net/dn/CQQxd/btsH9ocTyUV/tYYJWFgx3RvE8Vs9AO35O0/img.png](https://blog.kakaocdn.net/dn/CQQxd/btsH9ocTyUV/tYYJWFgx3RvE8Vs9AO35O0/img.png)

In RNN, hidden states store information that RNN model uses to update parameters. For example, in generating programming language, a given hidden state has information that two spaces were made so far. Referring this hidden state, a model will generate more spaces as needed. Thus, we have to know where a required information is.

We can idenfity this by seeing how a particular dimension of hidden state works. Set a particular dimension of hidden states being active then see how sequence data changes. In below figure, blue means negative and red means positive. In this case, since condition parts are all positive and others are all negative, we can know that this dimension contains information about conditional parts of if statement cell.

![https://blog.kakaocdn.net/dn/VYZ0u/btsH7ULbYdw/TeucBXah2PGjWwUnrhxKa0/img.png](https://blog.kakaocdn.net/dn/VYZ0u/btsH7ULbYdw/TeucBXah2PGjWwUnrhxKa0/img.png)

However, with previous Vanilla RNN, it cannot be trained well like above. That is, multiplying the same matrix at each time step during backpropagation causes gradient vanishing or exploding. If common ratio is smaller than 0, gradient will be vanished and otherwise, gradient will be exploded.

**Example**

![https://blog.kakaocdn.net/dn/oQfWA/btsH7TlbnlW/OsauKeKM3B6dhGohpYSElk/img.png](https://blog.kakaocdn.net/dn/oQfWA/btsH7TlbnlW/OsauKeKM3B6dhGohpYSElk/img.png)

![https://blog.kakaocdn.net/dn/y1YVs/btsH7UEnzii/b0tJ0kkzKBi5T3whbmvju0/img.png](https://blog.kakaocdn.net/dn/y1YVs/btsH7UEnzii/b0tJ0kkzKBi5T3whbmvju0/img.png)

𝑊ℎℎ is mutiplied twice (= 3x3 =9) and as the length of sequence data increases, then gradients will be exploded. If 𝑊ℎℎ = 0.2, then gradient will be 0.2x0.2. It will be vanished as the length of sequence data increases.

Thus, long-term dependency will occur due to the operation of backpropagation and this model will not be trained well in both cases.

## LSTM & GRU

There is a long-term dependency problem in RNN which means that if the length of data increases, then gradient for updating parameters might be either vanished or exploded. We can resolve this problem with LSTM and GRU.

### 1. LSTM

Simply, the meaning of LSTM is to hold short-term memory longer.The core idea of LSTM is to pass cell state information straightly without any transformation. ℎ𝑡=𝑓𝑊(𝑥𝑡,ℎ𝑡−1) is the equation of computing hidden state vector. However, in LSTM, a model considers not only hidden state but also cell state.

𝑐𝑡,ℎ𝑡=𝐿𝑆𝑇𝑀(𝑥𝑡,𝑐𝑡−1,ℎ𝑡−1)

Then, between cell state and hidden state, which one has more significant information? In general, cell states are considered to have more important information because hidden state is filtered (fabricated) once. As shown in below figure, there are four gates in a middle node.

![https://blog.kakaocdn.net/dn/pPLD3/btsH88BoDVB/HaJOIXzOr4kW7stqJQHzEk/img.png](https://blog.kakaocdn.net/dn/pPLD3/btsH88BoDVB/HaJOIXzOr4kW7stqJQHzEk/img.png)

![https://blog.kakaocdn.net/dn/cG7Pdr/btsH9oYg8gF/Zg7iU07kPG5wS8zlJgsB8K/img.png](https://blog.kakaocdn.net/dn/cG7Pdr/btsH9oYg8gF/Zg7iU07kPG5wS8zlJgsB8K/img.png)

Since there are four gates to pass through, the row dimension of w is 4h. Four gates are called Ifog.

- I: Input gate which represents the degree to which new information (Gate gate) will be used to cell state, with values ranging from 0 to 1.
    - expression:
        
        𝑠𝑖𝑔𝑚𝑜𝑖𝑑(𝑊(𝑥𝑡,ℎ𝑡−1))
        
    - After applies sigmoid, this value flows either cell state or hidden state.
- f: Forget gate which represents the degree to which previous cell state to be erased, with values ranging from 0 to 1.
    - expression:
        
        𝑠𝑖𝑔𝑚𝑜𝑖𝑑(𝑊(𝑥𝑡,ℎ𝑡−1))
        
- o: Output gate which represents the degree to which cell state's information to be used in hidden state which influences to output y, with value ranging from 0 to 1.
    - expression:
        
        𝑠𝑖𝑔𝑚𝑜𝑖𝑑(𝑊(𝑥𝑡,ℎ𝑡−1))
        
- g: Gate gate which represents the degree to which new information is reflected in the cell state, with values ranging from 0 to 1.
    - expression:
        
        𝑡𝑎𝑛ℎ(𝑊(𝑥𝑡,ℎ𝑡−1))
        

**Specific Process**

1. Forget gate

![https://blog.kakaocdn.net/dn/cRpkWZ/btsH9gsBKQT/DjdPzxibeudxK41Lbl7tF1/img.png](https://blog.kakaocdn.net/dn/cRpkWZ/btsH9gsBKQT/DjdPzxibeudxK41Lbl7tF1/img.png)

Let's assume that previous cell state (𝐶𝑡−1) is [3, 5, -2]. 𝑓𝑡=𝜎(𝑊𝑓[ℎ𝑡−1,𝑥𝑡]+𝑏𝑓) is forget gate vector. Forget gate vector is also 3-dimensional and let's assume that it is [0.7, 0.4, 0.8]. In here, 0.7 means that it will preserve only 70% of that information, so erasing 30% of information. Then, multiply element-wisely, the result will be [2.1, 2.0, -1.6].

2. Gate gate & Input gate

![https://blog.kakaocdn.net/dn/OOWvz/btsH8G6sZWD/S6H05iWP5viEQzFCVXrF5K/img.png](https://blog.kakaocdn.net/dn/OOWvz/btsH8G6sZWD/S6H05iWP5viEQzFCVXrF5K/img.png)

Input gate vector is 𝑖𝑡=𝜎(𝑊𝑖[ℎ𝑡−1,𝑥𝑡]+𝑏𝑖) and Gate gate vector is 𝐶𝑡^=𝑡𝑎𝑛ℎ(𝑊𝐶[ℎ𝑡−1,𝑥𝑡]+𝑏𝐶). We can generate new cell state 𝐶𝑡=𝑓𝑡𝐶𝑡−1+𝑖𝑡𝐶𝑡^. This equation means that first, by multiplying 𝑓𝑡 (forget gate value) to previous cell state (𝐶𝑡), it adjusts some parts of information to be erased. Second, by multiplying Input gate vector (𝑖𝑡) to Gate gate vector (𝐶𝑡^) and adding it to former term, new cell state can have necessary information. This is how cell state is updated.

3. Output gate

![https://blog.kakaocdn.net/dn/es9GQ5/btsH92tCle8/i9klj84vKtOTkaGXtdEf51/img.png](https://blog.kakaocdn.net/dn/es9GQ5/btsH92tCle8/i9klj84vKtOTkaGXtdEf51/img.png)

Generate hidden state by passing cell state to tanh and output gate. Output gate vector is 𝑜𝑡=𝜎(𝑊𝑜[ℎ𝑡−1,𝑥𝑡]+𝑏𝑜) and compute hidden state ℎ𝑡=𝑜𝑡𝑎𝑛ℎ(𝐶𝑡). Output gated determines the degree of cell state's information to be used in hidden state. This is how hidden state is updated.

Therefore, cell state includes all memory but hidden state includes brief and filtered information that directly needs to prediction. For example, let's assume that we are generating "hello" and we have currently generated up to "hell. The information that the quotation mark is open and needs to be closed later is not immediately necessary, so this information is stored in the cell state. However, the information directly needed to generate the next character 'o' is stored in the hidden state.

### 2. GRU

GRU is a model that lighten LSTM by unify cell state and hidden state. However, operation of GRU is very similar to that of LSTM. Unified vector ℎ𝑡−1 is similar to cell state because it contains all information.

![https://blog.kakaocdn.net/dn/n4ETA/btsH8HYDxFS/mbOk6JIDm4mPA76yzkURc1/img.png](https://blog.kakaocdn.net/dn/n4ETA/btsH8HYDxFS/mbOk6JIDm4mPA76yzkURc1/img.png)

In GRU, it uses 𝑧𝑡 instead of 𝑖𝑡 and 1−𝑧𝑡 instead of 𝑓𝑡. Let's assum that 𝑧𝑡 is [0.6 0.3 0.8], then 1−𝑧𝑡 is [0.4 0.7 0.2]. Thus, as 𝑧𝑡 increases, values of forget gate vector get smaller. It is weight average between ℎ𝑡−1 and ℎ𝑡^.

GRU is a model that reduces computational and memory requirements compared to LSTM, while also achieving performance that is similar or better than LSTM.

### 3. Backpropagation of LSTM & GRU

There is no gradient vanishing or explosion problem since hidden state is computed by multiplying forget state vector and then **adding** required information to it. Addition can convey gradients even furtuer data without any transformations, thus can resolve long-term dependency.

RNNs allow a lot of flexibility in architecture design. However, Vanilla RNNs don't work very well due to long-term dependeny. Backward flow of gradients in RNN can explode or vanish. Thus, it is common to use LSTM or GRU.