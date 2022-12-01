# Rating student essays through continuous soft prompts and adversarial learning

## This repo contains the solution for the [Feedback prize - English language learning](https://www.kaggle.com/competitions/feedback-prize-english-language-learning) competition. This approach achieved a silver medal(25th out of 2654 participants).

### Competition

The goal of this competition is to assess the language proficiency of 8th-12th grade English Language Learners (ELLs). Utilizing a dataset of essays written by ELLs will help to develop proficiency models that better supports all students.

### Evaluation

Submissions were scored using [MCRMSE](https://www.kaggle.com/competitions/feedback-prize-english-language-learning/overview/evaluation), mean columnwise root mean squared error.

### Approach

#### 1. The addition of continuous prompts before the actual sequence
This idea is motivated from the paper [The Power of Scale for Parameter-Efficient Prompt Tuning](https://aclanthology.org/2021.emnlp-main.243.pdf). We added 40 continuous tokens before the actual sequence by keeping the model parameters frozen. The model adopted for this was the Deberta-v3-large model.

#### 2. Layer Wise Learning-rate Decay (LLRD)
This idea is motivated from the [ULMFit](https://arxiv.org/pdf/1801.06146.pdf) paper. Having different learning rates for different layers helps to generalize better on the downstream task. The main idea is that earlier layers learn more general features while the later layers learn more task-specific features.

#### 3. Adversarial Learning
Techniques like AWP (Adversarial Weight perturbation), FGM were used to prevent overfitting and improve geenralization capability of the model.

#### 4. Faster training 
Mixed precision training, layer freezing, gradient accumulation, and gradient checkpointing were implemented to enable faster training and prevent CUDA Out of Memory Errors.

