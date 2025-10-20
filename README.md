🔐 Sentra-Guard: A Multilingual Human-AI Framework for Real-Time Defense Against Adversarial LLM Jailbreaks

📍 Submitted to IEEE Transactions on Dependable and Secure Computing (TDSC)

👤 First Author: Md Mehedi Hasan

📄 Abstract

Sentra-Guard is a modular, real-time defense framework designed to detect and mitigate jailbreak and prompt injection attacks targeting Large Language Models (LLMs). The system combines retrieval-augmented reasoning with fine-tuned transformers to deliver state-of-the-art performance against direct and obfuscated adversarial attacks — even across 100+ languages.

Key features include:
	•	A multilingual normalization pipeline for cross-lingual prompt understanding.
	•	Hybrid semantic retrieval using FAISS-indexed SBERT embeddings.
	•	A fine-tuned transformer classifier for high-precision detection.
	•	A zero-shot inference branch to capture unseen attack variants.
	•	A Human-in-the-Loop (HITL) feedback loop enabling live learning.
	•	Real-time performance (avg. processing time < 47ms).
	•	AUC = 1.00, F1 = 1.00, and ASR = 0.004%, outperforming LlamaGuard-2 and OpenAI Moderation.

⸻
## Datasets: [Link-1](https://huggingface.co/datasets/Spony/harmbench-dataset) and [Link-2](https://huggingface.co/datasets/JailbreakV-28K/JailBreakV-28k)
##🧠 Methodology Overview
<img width="1303" height="900" alt="Methodology" src="https://github.com/user-attachments/assets/50ec4667-33ba-47a9-840c-fbf1d76f7be9" />

The Sentra-Guard architecture is composed of six interconnected modules, designed for modularity, scalability, and low latency:

1. 🌐 Language Normalization & Translation (MLT)
	•	Translates non-English prompts into English using neural machine translation (NMT).
	•	Ensures consistent processing across all languages and dialects.

2. 🔍 Semantic Retrieval (SR) Engine
	•	Embeds input using Sentence-BERT (SBERT).
	•	Retrieves the top-k most similar vectors from a FAISS-indexed knowledge base containing labeled prompts.
	•	Evaluates semantic similarity to known benign or malicious patterns.

3. 🤖 Fine-Tuned Transformer Classifier (FTTC)
	•	Uses models like DistilBERT or DeBERTa-v3.
	•	Trained on a large, labeled adversarial prompt dataset.
	•	Outputs a probabilistic risk score (benign or harmful).

4. 🧪 Zero-Shot Classifier (ZSC)
	•	Uses facebook/bart-large-mnli for Natural Language Inference.
	•	Determines if a prompt semantically entails harmful behavior.
	•	Detects novel or obfuscated jailbreaks that may bypass traditional classifiers.

5. ⚖️ Decision Fusion Aggregator (DFA)
	•	Combines outputs from SR, FTTC, and ZSC.
	•	Uses weighted rules and thresholds to assign final risk labels.
	•	Escalates uncertain cases for human review.

6. 🧍 Human-in-the-Loop (HITL) Feedback Module
	•	Human experts validate borderline cases.
	•	Validated samples are used to:
	•	Update the FAISS vector index.
	•	Incrementally fine-tune the classifier.
	•	Enables live system adaptation under adversarial pressure.

⸻

## 📈 Performance Comparison

| **Metric**                   | **Sentra-Guard** | **LlamaGuard-2** | **OpenAI Moderation** |
|-----------------------------|------------------|------------------|------------------------|
| **Detection Rate (AUC)**    | 1.00             | 0.987            | 0.963                  |
| **F1 Score**                | 1.00             | 0.968            | 0.932                  |
| **Attack Success Rate (ASR)** | 0.004%          | 1.3%             | 3.7%                   |
| **Avg. Inference Time**     | < 47ms           | >150ms           | Varies                 |

flowchart TD
    A[Input Prompt (Any Language)] --> B[MLT: Translate to English]
    B --> C1[Semantic Retrieval (SR)]
    B --> C2[Fine-Tuned Transformer Classifier (FTTC)]
    B --> C3[Zero-Shot Classifier (ZSC)]
    C1 --> D[Decision Fusion Aggregator (DFA)]
    C2 --> D
    C3 --> D
    D -->|High Confidence| E[Final Risk Label]
    D -->|Low Confidence| F[HITL Feedback]
    F -->|Validated| G[Update FAISS / Fine-tune Classifier]


 💡 Why Sentra-Guard Matters
	•	Multilingual Coverage: Works across 100+ languages with language-agnostic translation.
	•	Transparency: Unlike black-box moderation APIs, all components are interpretable and adaptable.
	•	Modularity: Easily deployable in both research and commercial environments.
	•	Continual Learning: Human oversight enables rapid response to new attack techniques.
	•	Research-Ready: Ideal for academic extension, reproducibility, and experimentation in adversarial LLM security.

⸻

📚 Keywords

Large Language Models (LLMs) · Jailbreak Detection · Prompt Injection · Transformer Classifiers · Semantic Retrieval · FAISS · SBERT · Human-in-the-Loop (HITL) · Zero-Shot Inference · Multilingual NLP

⸻

📫 Contact

## 📫 Contact
**Md Mehedi Hasan**  
Lecturer | Cybersecurity & AI Researcher  (LLMs)
📧 Email: [mehedi.hasan.ict@mbstu.ac.bd](mehedi.hasan.ict@mbstu.ac.bd)  
🔗 [LinkedIn](https://www.linkedin.com/in/md-mehedi-hasan-878379193/) • [Portfolio Website](https://md-mehedi-hasan-resume.vercel.app/)


# LLM-Based-Sentry-Cyber-Security-
LLM-Sentry is a hybrid detection pipeline combining real-time retrieval (FAISS), prompt similarity (SBERT), and fine-tuned classification (DistilBERT) trained on JailbreakBench. It supports adaptive learning through HITL updates and performs real-time filtering in a modular, reproducible Google Colab environment.
 View Website architechtue link: https://claude.ai/public/artifacts/25398610-0385-4d90-8aff-c4cec42e4124
 
