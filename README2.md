# Curious George Adventure Creator: AI Comic Book Generator

Welcome to the future of storytelling with the Curious George Adventure Creator! Dive into the playful and adventurous world of Curious George like never before. Our innovative AI model, designed for children and fans of the lovable monkey, brings George's stories to life with just a starting image and a chosen theme.

---

## Video Demo

<a href="https://youtu.be/_t87U1tLiyQ"><img src="assets/logo.png" width="300" height="300" alt="Thumbnail"></a>

*Here is a video demo of our App.*

---

## What is it?

The Curious George Adventure Creator allows you to craft unique episodes featuring the sweet and curious African monkey, Curious George, and his ever-patient friend, "The Man in the Yellow Hat." George's adventures, often filled with playful curiosity and unforeseen trouble, are brought to life through our cutting-edge AI technology.

### Key Features

1. **Easy Episode Creation**: Simply upload a starting image and select a theme, and our AI model will generate a personalized Curious George episode. Watch George explore, learn, and get into his usual delightful mishaps, all tailored to your input.
   
2. **Learning and Fun**: Each episode emphasizes themes of learning, forgiveness, and curiosity. It's not just entertainment; it's an educational experience wrapped in fun and adventure.

3. **Voice Options**: 
   - **Loan Your Voice**: Bring a personal touch to your episode by lending your own voice to the characters.
   - **Audio Library**: Choose from a variety of pre-recorded voices, including cloned voices that perfectly match the characters.

## Technical Aspects

### This project is developer using SEED-Story

A Multimodal Large Language Model (MLLM) capable of generating multimodal long stories consisting of rich and coherent narrative texts, along with images that are consistent in characters and style, based on [SEED-X](https://github.com/AILab-CVC/SEED-X). 

### How It Works

1. **Stage 1: Visual Tokenization & De-tokenization**
   - Pre-train an SD-XL-based de-tokenizer to reconstruct images by taking the features of a pre-trained Vision Transformer (ViT) as inputs.

2. **Stage 2: Multimodal Sequence Training**
   - Sample an interleaved image-text sequence of a random length.
   - Train the MLLM by performing next-word prediction and image feature regression between the output hidden states of the learnable queries and ViT features of the target image.

3. **Stage 3: De-tokenizer Adaptation**
   - The regressed image features from the MLLM are fed into the de-tokenizer for tuning SD-XL, enhancing the consistency of the characters and styles in the generated images.

Given the same initial image but different opening texts, SEED-Story can generate different multimodal stories. For instance, starting with text referencing “the man in the yellow hat” will lead to images that include the character, while omitting this reference will result in a different narrative direction.

## Usage

### Dependencies
- Python >= 3.8
- PyTorch >= 2.0.1

### Installation
Clone the repo and install dependent packages:

```bash
git clone https://github.com/YourRepo/SEED-Story.git
cd SEED-Story
pip install -r requirements.txt
```


### Model Weights
Download the pre-trained models and save them under the `./pretrained` folder:

- Tokenizer
- De-Tokenizer
- SEED-X-pretrained
- SEED-Story-George
- Detokenizer-George

Also, download models like [stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0), [Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf), and [Qwen-VL-Chat](https://huggingface.co/Qwen/Qwen-VL-Chat).

### App

Experience the magic of Curious George's world like never before. Start creating your own episodes today and let your imagination soar!

Ready to embark on an adventure with Curious George?

```bash
python3 app.py
```

