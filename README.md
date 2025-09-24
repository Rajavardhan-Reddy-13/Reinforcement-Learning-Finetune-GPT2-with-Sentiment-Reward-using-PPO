# 🤖 GPT-2 Fine-Tuning with RLHF and Sentiment-Based Rewards (PPO + IMDB)

This project fine-tunes a **GPT-2 language model** using **Reinforcement Learning from Human Feedback (RLHF)** to generate **more positive and human-aligned text responses**. The model is trained on movie review prompts and rewarded based on the **positivity of its responses**, as judged by a sentiment analysis model. The training uses the **Proximal Policy Optimization (PPO)** algorithm and is inspired by real-world RLHF techniques used to align large language models with human preferences.

---

## 📌 Project Goal

To train GPT-2 to generate **positive movie review responses** by using **reward scores** from a sentiment classifier and optimizing behavior using **reinforcement learning (PPO)**.

---

## 📚 Dataset

### 🔹 IMDB Movie Reviews
- Source: Hugging Face Datasets
- Description: Contains 50,000+ movie reviews labeled as positive or negative.
- Use: Prompts are extracted from long reviews to train the model on realistic user inputs.

---

## 🧠 Models Used

- **GPT-2 (`lvwerra/gpt2-imdb`)**  
  > Base language model used for response generation and PPO optimization.

- **DistilBERT Sentiment Classifier (`lvwerra/distilbert-imdb`)**  
  > Reward model that scores the positivity of generated responses.

---

## 🛠️ Tools & Libraries

- **Python**
- **Google Colab** (training environment)
- **Hugging Face Libraries:**
  - `transformers` – model and tokenizer handling
  - `datasets` – loading IMDB data
  - `trl` – PPO-based RLHF training
- **Weights & Biases (wandb)** – experiment logging
- **PyTorch** – deep learning backend

---

## 🚀 Implementation Steps

### ✅ 1. Install Required Libraries
```bash
pip install transformers==4.37.2
pip install datasets==2.16.1
pip install trl==0.7.10
pip install tqdm==4.66.1
pip install torch==2.2.0
pip install peft==0.10.0
pip install "numpy<2"
```

### ✅ 2. Load Dataset and Tokenizer
- Load IMDB dataset and extract only long reviews.
- Use short excerpts (2–8 tokens) from reviews as **prompts**.

### ✅ 3. Initialize Models
- GPT-2 with a **value head** is used for PPO training.
- A **reference GPT-2** is used to measure divergence.
- Tokenizer pad token is set to EOS for consistency.

### ✅ 4. Sentiment-Based Reward Function
- A **DistilBERT sentiment classifier** is used to assign rewards:
  - Higher reward for more **positive** responses.
  - Only the **"POSITIVE" score** is used in PPO training.

### ✅ 5. Training Loop with PPO
- For each prompt:
  - GPT-2 generates a response.
  - Prompt + response is sent to the reward model.
  - PPOTrainer uses the reward to improve GPT-2's behavior.
- PPO minimizes deviation from the reference model while maximizing reward.

### ✅ 6. Evaluation
- A batch of prompts is run through:
  - Original GPT-2 (before training)
  - Fine-tuned GPT-2 (after training)
- Both outputs are scored using the sentiment classifier.
- Comparison is printed using **mean** and **median** scores.

---

## ✅ Example Output

**Prompt:**
```
This movie had amazing acting and a beautiful story.
```

**Before Fine-Tuning:**
```
But the ending was boring and predictable.
```

**After Fine-Tuning (RLHF + PPO):**
```
It’s one of the most heartwarming films I’ve seen!
```

✅ Shows how the model learns to favor **positive, helpful responses** over negative ones.

---

## 📊 Results Summary

| Metric                   | Before Training | After Training |
|--------------------------|------------------|-----------------|
| Avg. Sentiment Score     | ~0.34            | ~0.81           |
| Output Tone              | Mixed/Neutral    | Positive         |
| Alignment with Human Preference | Low         | High             |

---

## 📈 Future Enhancements

- Train for more PPO steps with longer prompts
- Add constraints for fluency or coherence (using custom rewards)
- Evaluate using human preference labels (if available)
- Integrate outputs into chat-style interface using Gradio

---

## 🧾 References

- 🔗 TRL PPO Example: [https://github.com/huggingface/trl](https://github.com/huggingface/trl)
- 🔗 PPO Paper (OpenAI): [https://arxiv.org/abs/1707.06347](https://arxiv.org/abs/1707.06347)
- 🔗 YouTube Guide: [https://www.youtube.com/watch?v=qGyFrqc34yc](https://www.youtube.com/watch?v=qGyFrqc34yc)
- 🔗 IMDB Dataset: [https://huggingface.co/datasets/imdb](https://huggingface.co/datasets/imdb)
- 🔗 Starter Code Repo: [https://github.com/hkproj/rlhf-ppo](https://github.com/hkproj/rlhf-ppo)
- 🔗 Hugging Face GPT-2 Sentiment Example: [https://github.com/huggingface/trl/blob/main/examples/notebooks/gpt2-sentiment.ipynb](https://github.com/huggingface/trl/blob/main/examples/notebooks/gpt2-sentiment.ipynb)

---

## © Copyright

```
This project is for academic and research use only.  
All datasets and models used are subject to their respective licenses.
```
