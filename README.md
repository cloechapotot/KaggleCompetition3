# KaggleCompetition3

# Emotion Classification CLI

A simple, pip-installable command-line tool for predicting the primary emotion expressed in a tweet-style text.  
Supports six basic emotions: **sadness**, **joy**, **love**, **anger**, **fear**, and **surprise**.

---

## Features

- **`inference --input "<your text>"`**  
  Classify any input string and print the predicted emotion label.

- **`inference --kaggle`**  
  Print your Kaggle username (for easy verification/submission).

---

## Installation

1. **Generate a GitHub Personal Access Token**  
   - Go to **GitHub → Settings → Developer settings → Personal access tokens**.  
   - Create a new token with **`repo: read`** scope and copy it.

2. **Install via pip**  
   Replace `<YOUR_TOKEN>` and `yourusername/yourrepo` with your details:

   ```bash
   pip install git+https://<YOUR_TOKEN>@github.com/cloechapotot/KaggleCompetition3.git
