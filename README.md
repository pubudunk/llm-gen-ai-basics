# GenAI Basics

Welcome to the **GenAI Basics** repository! This repository is to provide an introductory understanding of Generative AI (GenAI) with practical examples.

## Table of Contents

1. [Overview](#overview)
2. [Interacting with LLMs](#interacting-with-llms)
3. [Evaluating Models for Your Application](#evaluating-models-for-your-application)
4. [External Learning Resources](#external-learning-resources)
5. [Examples](#examples)
6. [Glossary](GLOSSARY.md)

---

## Overview

Generative AI (GenAI) is a subset of Artificial Intelligence that focuses on generating new content or solving complex problems by learning from data. 

- **Artificial Intelligence (AI)**: Computer systems that perform tasks typically requiring human intelligence, such as image recognition, language translation, and decision-making.
- **Machine Learning (ML)**: A subset of AI that involves algorithms that allow computers to learn from data and improve over time without being explicitly programmed for each task.
- **Neural Networks (NN)**: A family of ML models inspired by the human brain, made up of layers of "neurons" that process data in a way similar to human cognitive processes.
- **Deep Learning (DL)**: A subset of NN that uses networks with many layers (known as "deep" networks). Key architectures include:
  - **Convolutional Neural Networks (CNN)**: Primarily used for image processing tasks.
  - **Recurrent Neural Networks (RNN)**: Designed to handle sequential data, such as text and speech.
  - **Long Short-Term Memory Networks (LSTM)**: Addressed the limitations of RNNs.
- **Generative AI (GenAI)**: Built on deep learning models, GenAI focuses on generating new content such as text, images, and videos. Examples include:
  - **GPT (Generative Pre-trained Transformer)** by OpenAI
  - **Gemini** by Google DeepMind
  - **Llama** by Meta
  - **DeepSeek** by DeepMind

### Evolution of AI and GenAI:
The journey of AI from its early beginnings to the present day involves several milestones:
- **Feedforward Networks**: Early neural networks used for analyzing objects and images.
- **CNNs (Convolutional Neural Networks)**: Emerged to solve problems in image recognition.
- **RNNs (Recurrent Neural Networks)**: Designed to handle sequential data such as text and speech.
- **LSTMs (Long Short-Term Memory Networks)**: Addressed the limitations of RNNs, such as the vanishing gradient problem.
- **Transformers**: A breakthrough in neural architecture, solving issues with RNNs and LSTMs (Sequential bottleneck, limited long-range modeling etc). Transformers revolutionized the NLP field with their self-attention mechanism, allowing models to capture dependencies in data more effectively.
  
### Today's Landscape of LLMs:
- **Encoder-only LLMs** (e.g., BERT) focus on understanding and encoding input data. These are often used for tasks like classification and question answering.
- **Decoder-only LLMs** (e.g., GPT) are generative models that can create new content, making them suitable for text generation tasks, including chatbots and code completion.
- **Encoder-Decoder LLMs** (e.g., T5, BART) combine both understanding and generation, suitable for tasks like summarization and translation.


---

## Interacting with LLMs

There are two primary options when integrating a large language models (LLMs) into your application.

### Cloud Hosted

**Pros:**
- **Ease of Access**: You can start using models right away without worrying about hardware or infrastructure.
- **Scalability**: Cloud services can scale up to meet your demands as needed.
- **Maintenance-Free**: The cloud provider handles updates and maintenance.

**Cons:**
- **Cost**: Continuous usage can become expensive over time.
- **Dependency**: Your system will be dependent on the availability of the cloud service.
- **Data Privacy**: Sensitive data might be processed outside your control, depending on the cloud provider.

**Common Providers**:
- **OpenAI (GPT)**
- **Google Cloud AI (Gemini)**
- **Hugging Face API**

### On-Prem Hosted

**Pros:**
- **Cost Control**: You only pay for hardware and storage, without per-usage charges.
- **Data Privacy**: Complete control over sensitive or proprietary data.
- **Customization**: You can fine-tune models to fit your specific needs.

**Cons:**
- **Setup Complexity**: Requires substantial setup, hardware, and maintenance.
- **Scalability**: Scaling might be more difficult compared to cloud-based solutions.
- **Maintenance**: You are responsible for updates, model optimization, and troubleshooting.

**Recommended Hardware**:
- **NVIDIA GPUs** for model inference.
- **High-performance CPUs and sufficient RAM** for handling large models.

---

## Evaluating Models for Your Application

When selecting an LLM for your application, consider the following factors:

1. **Performance and Accuracy**: Choose models that offer the best performance for your specific tasks (e.g., text generation, classification).
   - **Metrics**: Look for evaluation metrics like perplexity and accuracy on benchmark datasets.
   
2. **Latency and Response Time**: Consider models that can provide real-time responses, especially if you're integrating them into interactive applications (e.g., chatbots).
   - **Optimization**: Some models can be fine-tuned or pruned to reduce latency without significant loss in performance.

3. **Resource Requirements**: Some models require substantial computational resources (e.g., memory, GPUs) while others are more lightweight and suitable for edge devices or mobile applications.
   - **Consider Deployment**: Will your application run on the cloud, edge, or local devices?

4. **Cost**: Depending on the model and deployment method (cloud vs. on-prem), the cost can vary significantly. Calculate the total cost of ownership over time.
   - **Cloud Costs**: Pay-as-you-go models (like OpenAI's GPT) can be expensive with high usage.
   - **On-Prem Hardware**: Initial setup costs for servers/GPU hardware can be high but might be more cost-effective in the long term.

5. **Community Support and Documentation**: Look for models that are well-supported by the community with good documentation, tutorials, and APIs.
   - **Popular Frameworks**: Models like GPT, Gemini, and Llama have strong community support and libraries (e.g., Hugging Face, OpenAI API).

---

## External Learning Resources

Here are some simple but high-quality external resources to deepen your understanding of GenAI. Credit of the resources
should go to original creators.

   - **[Neural Networks Explained in 5 minutes](https://www.youtube.com/watch?v=jmmW0F0biz0)**
   - **[What are Convolutional Neural Networks (CNNs)?](https://www.youtube.com/watch?v=QzY57FaENXg)** 
   - **[What is Back Propagation](https://www.youtube.com/watch?v=S5AGN9XfPK4)** 
   - **[What is a Vector Database? Powering Semantic Search & AI Applications](https://www.youtube.com/watch?v=gl1r1XV0SLw)** 
   - **[How Large Language Models Work](https://www.youtube.com/watch?v=5sLYAQS9sWQ)** 
   - **[What is Retrieval-Augmented Generation (RAG)?](https://www.youtube.com/watch?v=T-D1OfcDW1M)** 
   - **[RAG vs Fine-Tuning vs Prompt Engineering: Optimizing AI Models](https://www.youtube.com/watch?v=zYGDpG-pTho)** 
   - **[Generative vs Agentic AI: Shaping the Future of AI Collaboration](https://www.youtube.com/watch?v=EDb37y_MhRw&t=311s)** 
   - **[5 Types of AI Agents: Autonomous Functions & Real-World Applications](https://www.youtube.com/watch?v=fXizBc03D7E)** 
   - **[10 Use Cases for AI Agents: IoT, RAG, & Disaster Response Explained](https://www.youtube.com/watch?v=Ts42JTye-AI)** 
   - **[Risks of Agentic AI: What You Need to Know About Autonomous AI](https://www.youtube.com/watch?v=v07Y4fmSi6Y)** 
   - **[How Transformer LLMs Work](https://www.deeplearning.ai/short-courses/how-transformer-llms-work/)** : Short course for a deep dive
   - **[Attention Is All You Need (2017)](https://arxiv.org/pdf/1706.03762)** : Google’s original paper on transformer architecture, removing recurrence, with self-attention.

---

## Examples

This section provides simple examples to get hands-on with GenAI concepts.