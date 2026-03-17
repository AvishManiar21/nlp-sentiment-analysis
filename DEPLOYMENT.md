# Deployment Guide: Deep Learning Models on Streamlit Cloud

This guide explains how to deploy the NLP Sentiment Analysis dashboard to Streamlit Cloud with deep learning models stored on HuggingFace Hub.

## Problem Statement

Deep learning model files (`.pt`, `.keras`) are large (10-30 MB each) and are excluded from git via `.gitignore`. When deploying to Streamlit Cloud from a GitHub repository, these model files are not included, so the dashboard cannot find them.

## Solution: HuggingFace Hub Integration

We solve this by storing models on HuggingFace Hub and downloading them automatically when the Streamlit app starts on the cloud.

**Benefits:**
- ✓ Clean git repository (no large binary files)
- ✓ Automatic model download on Streamlit Cloud
- ✓ Easy model versioning and updates
- ✓ Works seamlessly for both local and cloud deployment

---

## Step 1: Train Your Models Locally

If you haven't already, train your deep learning models locally:

```bash
# Activate virtual environment
source venv_py312/Scripts/activate  # On Windows Git Bash
# OR
venv_py312\Scripts\activate  # On Windows CMD

# Train all models (takes ~10-15 minutes on CPU)
python main.py --train-dl --dl-framework both --dl-model-type both --use-embeddings --embedding-name glove-wiki-gigaword-100 --dl-epochs 15 --dl-batch-size 32
```

**Verify models exist:**
```bash
ls -lh models/dl/
```

You should see files like:
- `cnn_tensorflow.keras`
- `cnn_tensorflow_pretrained.keras`
- `cnn_pytorch.pt`
- `cnn_pytorch_pretrained.pt`
- `lstm_pytorch_pretrained.pt`

---

## Step 2: Create HuggingFace Account

1. Go to https://huggingface.co/
2. Sign up for a free account
3. Verify your email

---

## Step 3: Install HuggingFace CLI and Login

```bash
# Install huggingface-hub if not already installed
pip install huggingface-hub

# Login to HuggingFace
huggingface-cli login
```

When prompted, enter your HuggingFace access token:
1. Go to https://huggingface.co/settings/tokens
2. Click "New token"
3. Give it a name (e.g., "nlp-sentiment-models")
4. Select "Write" permission
5. Copy the token and paste it in the terminal

---

## Step 4: Upload Models to HuggingFace Hub

Use the provided upload script:

```bash
# Upload to a public repository (recommended for open source projects)
python scripts/upload_models_to_hub.py YOUR_USERNAME/nlp-sentiment-models

# OR upload to a private repository (for private projects)
python scripts/upload_models_to_hub.py YOUR_USERNAME/nlp-sentiment-models --private
```

**Replace `YOUR_USERNAME`** with your HuggingFace username.

**Example:**
```bash
python scripts/upload_models_to_hub.py johndoe/nlp-sentiment-models
```

The script will:
- ✓ Check if models exist locally
- ✓ Create the repository on HuggingFace
- ✓ Upload all model files
- ✓ Create a README with model information

**Verify upload:**
1. Go to https://huggingface.co/YOUR_USERNAME/nlp-sentiment-models
2. Verify all model files are listed
3. Note down the repository ID (you'll need it for Streamlit Cloud)

---

## Step 5: Deploy to Streamlit Cloud

### 5.1 Push Code to GitHub

Make sure your code is pushed to GitHub:

```bash
git add .
git commit -m "Add HuggingFace Hub integration for model deployment"
git push origin main
```

**Note:** Model files are gitignored and won't be pushed (this is intentional).

### 5.2 Deploy on Streamlit Cloud

1. Go to https://streamlit.io/cloud
2. Sign in with GitHub
3. Click "New app"
4. Select your repository: `YOUR_GITHUB_USERNAME/nlp-sentiment-analysis`
5. Main file path: `app.py`
6. Click "Advanced settings"

### 5.3 Add Environment Variable

In the "Secrets" or "Advanced settings" section, add an environment variable:

**Variable name:** `HF_MODEL_REPO`
**Value:** `YOUR_HF_USERNAME/nlp-sentiment-models`

**Example:**
```
HF_MODEL_REPO=johndoe/nlp-sentiment-models
```

7. Click "Deploy"

### 5.4 First Run

On the first run, the Streamlit app will:
1. Check if models exist locally (they don't on cloud)
2. Detect the `HF_MODEL_REPO` environment variable
3. Download models from HuggingFace Hub
4. Cache them for future runs
5. Display the dashboard with all model features

**Subsequent runs:**
- Models are cached, no download needed
- App starts immediately

---

## Step 6: Verify Deployment

Once deployed:

1. Wait for the app to load (first run takes 1-2 minutes for model download)
2. Navigate to the "Model Performance" tab
3. Verify all deep learning models are listed:
   - CNN (TensorFlow)
   - CNN + GloVe (TensorFlow)
   - CNN (PyTorch)
   - CNN + GloVe (PyTorch)
   - BiLSTM + GloVe (PyTorch)

---

## Troubleshooting

### Issue: Models not showing on dashboard

**Check 1: Environment variable set correctly**
```
Streamlit Cloud → App Settings → Secrets
Verify: HF_MODEL_REPO=your-username/nlp-sentiment-models
```

**Check 2: Models uploaded to HuggingFace**
- Visit https://huggingface.co/your-username/nlp-sentiment-models
- Verify all `.pt` and `.keras` files are there

**Check 3: Repository is public OR you have access**
- If private repo, make sure the HuggingFace token has access
- Recommendation: Use public repo for easier deployment

**Check 4: Check Streamlit Cloud logs**
```
Streamlit Cloud → Your App → Manage app → Logs
Look for errors related to model downloading
```

### Issue: Download fails with authentication error

**Solution:** Repository might be private. Either:
1. Make the repository public on HuggingFace, OR
2. Add HuggingFace token to Streamlit Cloud secrets:
   ```
   HF_TOKEN=your_huggingface_token
   ```

### Issue: Models work locally but not on cloud

**Check:** Filenames might be incorrect. Models must be named:
- `cnn_tensorflow.keras`
- `cnn_tensorflow_pretrained.keras`
- `cnn_pytorch.pt`
- `cnn_pytorch_pretrained.pt`
- `lstm_pytorch_pretrained.pt`

**Fix:** Rename files locally and re-upload:
```bash
# Rename if needed (example)
cd models/dl
mv cnn_scratch.keras cnn_tensorflow.keras
cd ../..

# Re-upload
python scripts/upload_models_to_hub.py YOUR_USERNAME/nlp-sentiment-models
```

### Issue: Out of memory on Streamlit Cloud

Streamlit Cloud free tier has limited resources. If models are too large:

1. **Option A:** Upload only essential models (e.g., just the pretrained ones)
2. **Option B:** Upgrade to Streamlit Cloud paid tier
3. **Option C:** Use smaller embeddings (glove-50d instead of glove-100d)

---

## Local Development

For local development, you have two options:

### Option 1: Use Local Models (Recommended)

Just train models locally:
```bash
python main.py --train-dl ...
```

Models are detected automatically from `models/dl/`.
No HuggingFace download happens.

### Option 2: Test HuggingFace Download Locally

Set the environment variable:
```bash
export HF_MODEL_REPO=your-username/nlp-sentiment-models  # On Mac/Linux
# OR
set HF_MODEL_REPO=your-username/nlp-sentiment-models  # On Windows CMD
```

Then run:
```bash
streamlit run app.py
```

The app will download models from HuggingFace even locally.

---

## Updating Models

When you retrain models and want to update the cloud version:

1. **Train new models locally:**
   ```bash
   python main.py --train-dl ...
   ```

2. **Re-upload to HuggingFace:**
   ```bash
   python scripts/upload_models_to_hub.py YOUR_USERNAME/nlp-sentiment-models
   ```

3. **Clear Streamlit Cloud cache:**
   - Go to Streamlit Cloud → Your App → Manage app
   - Click "Reboot app"
   - Models will be re-downloaded on next run

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Cloud                          │
│  ┌────────────────────────────────────────────────────┐    │
│  │  app.py                                            │    │
│  │  ↓                                                 │    │
│  │  1. Check if models exist locally                 │    │
│  │  2. If not, check HF_MODEL_REPO env var           │    │
│  │  3. Download from HuggingFace Hub                 │    │
│  │  4. Cache models for future runs                  │    │
│  │  5. Load and use models                           │    │
│  └────────────────────────────────────────────────────┘    │
│                            │                                │
└────────────────────────────┼────────────────────────────────┘
                             │
                             │ Download
                             ↓
                 ┌────────────────────────┐
                 │   HuggingFace Hub      │
                 │  (Model Storage)       │
                 │                        │
                 │  - cnn_tensorflow.keras│
                 │  - cnn_pytorch.pt      │
                 │  - lstm_*.pt           │
                 │  - ...                 │
                 └────────────────────────┘
                             ↑
                             │ Upload
                             │
                 ┌────────────────────────┐
                 │  Local Development     │
                 │  (Your Computer)       │
                 │                        │
                 │  python main.py        │
                 │    --train-dl          │
                 │                        │
                 │  python scripts/       │
                 │    upload_models...    │
                 └────────────────────────┘
```

---

## FAQ

**Q: Do I need to pay for HuggingFace?**
A: No, HuggingFace Hub is free for public repositories with reasonable usage.

**Q: Can I use private repositories?**
A: Yes, but you'll need to add authentication tokens to Streamlit Cloud.

**Q: How much storage do the models use?**
A: Total ~100-150 MB for all models. Well within HuggingFace free tier limits.

**Q: What if I don't want to use HuggingFace?**
A: Alternatives:
- AWS S3 + boto3
- Google Cloud Storage + google-cloud-storage
- Direct file hosting (not recommended)

**Q: Can I skip deep learning models entirely?**
A: Yes! The dashboard works fine with just classical ML models (Logistic Regression, Naive Bayes). Just don't upload DL models or set the HF_MODEL_REPO variable.

---

## Summary Checklist

- [ ] Train models locally
- [ ] Create HuggingFace account
- [ ] Install and login with HuggingFace CLI
- [ ] Upload models using the upload script
- [ ] Verify models on HuggingFace Hub
- [ ] Push code to GitHub
- [ ] Deploy to Streamlit Cloud
- [ ] Set `HF_MODEL_REPO` environment variable
- [ ] Verify models appear in dashboard
- [ ] Celebrate! 🎉

---

## Support

For issues:
1. Check Streamlit Cloud logs
2. Verify HuggingFace repository is accessible
3. Test model download locally with `HF_MODEL_REPO` set
4. Open an issue on GitHub with logs

---

**Last updated:** March 2026
**Author:** NLP Sentiment Analysis Project
