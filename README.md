# Episodic Extractive Text Summarization with Reinforcement Learning

We conceptualize extractive summarization as a sequential task and propose a novel RL
framework - *current summary* and the *next possible sentence* as the **state** and
decides whether to *include* or *exclude* it from the summary as **action**, at each timestep.

The loss function is a complementary reward scheme guiding the likelihood maximization 
objective, using the ROUGE evaluation metric.
We use a policy-based RL as the summarization agent, and use the policy gradient
REINFORCE algorithm to train our model.

Agent model architecture consists of 3 parts:
1. Article Encoder - encodes sentences in an article
2. SummmaryState Encoder - encodes sentences in the current summary state
3. Buffer Decoder - *attention* based sequential decoder to predict whether to keep/skip sentence at each timestep

We modify REINFORCE - instead of searching the summary space, 
we look at the high probability samples.
We approximate this set by selecting top candidate summaries which receive highest ROUGE scores:
* Assemble candidate summaries - select 10 sentences from the document with high ROUGE scores
* Generate all combinations summaries of maximum length 4 from the 10, and evaluate against gold summary
* Ranked according to the mean of ROUGE-1, ROUGE-2, and ROUGE-L. 

During training, sample *y* from this set instead of *p*(*y*|*D*,*θ*).