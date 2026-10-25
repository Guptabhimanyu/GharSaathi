# Deploy to Streamlit Community Cloud (free)

## 1. Build the index on your own computer first
```bash
pip install -r requirements.txt
python refresh.py          # downloads the official documents and builds data/index/
streamlit run app.py       # quick local test (needs at least GROQ_API_KEY or GEMINI_API_KEY in .env)
```
Check that `data/raw/` has your documents and `data/index/` exists.

## 2. Push to GitHub
Commit everything **including `data/`** (the built index lets the live app start fast, and the saved
documents are a fallback if a government site blocks the host's servers).
Do NOT commit `.env` (it is in `.gitignore`).

## 3. Create the app
share.streamlit.io -> New app -> pick your repo, branch, and main file `app.py`.
Under Advanced settings, choose Python 3.11 or 3.12 and paste your keys into **Secrets**:
```toml
GROQ_API_KEY = "..."
GEMINI_API_KEY = "..."
# optional: MISTRAL_API_KEY, CEREBRAS_API_KEY, OPENROUTER_API_KEY
```

## 4. After deploying
- First load is slow (it downloads the embedding model). Later loads are faster.
- Free apps go to sleep when idle; the next visitor wakes them.
- If the app runs out of memory, tell me: the next step is a lighter setup.
- Mic input needs HTTPS (Streamlit Cloud provides it).
