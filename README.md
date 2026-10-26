# GharSaathi

GharSaathi ("home companion") is an unofficial RAG assistant that helps people understand government housing schemes in India
(PMAY-U 2.0, with a focus on Affordable Rental Housing and loan routes). It interviews the user,
gives a personalized guide with citations, and answers follow-up questions by text or voice (Hindi/English).
Not official or financial advice.

## How it works
1. `sources.json` lists official documents (.gov.in / .nic.in only). `fetch.py` downloads and refreshes them.
2. `ingest.py` chunks them and builds a vector index (`fastembed`, runs on CPU).
3. `rag.py` retrieves passages and asks an LLM to answer using only those passages, with citations.
4. `llm.py` routes to free LLM providers with automatic fallback; if all fail, the app shows the raw passages.
5. `voice.py` handles speech-to-text (Groq Whisper), Hindi detection/translation, and optional text-to-speech.
6. `loan_calc.py` does the EMI maths. `app.py` is the Streamlit interface.

## Run locally
```bash
pip install -r requirements.txt
cp .env.example .env        # add at least GROQ_API_KEY and/or GEMINI_API_KEY
python refresh.py           # download documents + build index
streamlit run app.py
```
See `DEPLOY.md` for Streamlit Community Cloud.

## Roadmap
- [ ] `eval/`: ~50 hand-checked Q&A pairs (English + Hindi, incl. unanswerable) with scores per model
- [ ] Hybrid retrieval (BM25 + vectors) and reranking, compared in a results table
- [ ] Gujarat state documents, Gujarati support
- [ ] Check for low-quality model answers and retry on the next provider
- [ ] Nightly scheduled refresh

## Data credits
`static/cities.json` is derived from the open dataset github.com/nshntarora/Indian-Cities-JSON (names tidied; Sikkim, Ladakh and Lakshadweep have no entries, so users can type their town). Scheme documents come only from official .gov.in / .nic.in sources listed in `sources.json`.
