# BrowseComp-Plus ranking eval

Search the BrowseComp-Plus corpus with Bigdata.com and rank evidence with
GPT-5.6 Luna on AWS Bedrock. The harness downloads the official datasets,
uploads the corpus to a private Bigdata index, runs all 830 queries, and
reports document-level nDCG@10.

No benchmark plaintext, corpus data, credentials, or run artifacts are
included. Generated files under `browsecomp_plus/data/` are gitignored.

## What it runs

- Corpus: [`Tevatron/browsecomp-plus-corpus`](https://huggingface.co/datasets/Tevatron/browsecomp-plus-corpus) (100,195 documents).
- Encrypted benchmark: [`Tevatron/browsecomp-plus`](https://huggingface.co/datasets/Tevatron/browsecomp-plus) (830 queries).
- Upload: YAML title as the Bigdata filename, tag `browsecomp-plus`, private
  files, title/date removed from the body, author retained, translation
  enrichment only for non-English text.
- Retrieval: Bigdata Search API `fast` mode over `my_files` plus the tag,
  with `search`, `grep`, and `get_document`.
- Agent: `us.openai.gpt-5.6-luna`, reasoning effort `high`, at most 80
  search/grep calls, 12 document fetches, 16 rounds, and five concurrent
  queries.
- Ranking: the agent calls `submit_ranking` to stop search. A second Bedrock
  call then ranks the collected evidence (full text for up to 20 documents;
  titles and snippets beyond that).

Prompts and budgets are in `browsecomp_plus/eval/`.

## Setup

Python 3.10 is recommended. `requirements-lock.txt` pins the installed
dependency graph; `requirements.txt` lists the direct dependencies. Run the
commands below from the repository root.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-lock.txt
cp .env.example .env
```

Set `BIGDATA_API_KEY` in `.env`. Prefer the normal AWS credential chain or an
SSO/profile. Never commit `.env`.

## 1. Download the corpus and decrypt the benchmark

```bash
python -m browsecomp_plus.run download
python -m browsecomp_plus.eval.decrypt_eval
```

This downloads all 830 queries and writes a local scorer file (query, answer,
and gold/evidence docids). It fails unless it obtains exactly 830 unique
official query IDs. Decrypted outputs stay under the gitignored data folder.

## 2. Upload the corpus

```bash
python -m browsecomp_plus.run upload --concurrency 12
python -m browsecomp_plus.run upload --concurrency 12 --retry-failed
python -m browsecomp_plus.run export-mapping
python -m browsecomp_plus.run progress
```

Keep the upload private; do not pass `--share-with-org`. A successful PUT
means Bigdata accepted the bytes; indexing continues asynchronously. Some very
long documents may fail ingestion. Check coverage before starting the eval:

```bash
python -m browsecomp_plus.eval.corpus_check
python -m browsecomp_plus.eval.corpus_check --remote qrels
python -m browsecomp_plus.eval.corpus_check --remote all
```

The all-corpus check is resumable and may take several hours. Do not start
the eval while a check exits nonzero.

## 3. Run the eval

With no extra flags this runs all 830 queries at the concurrency in
`eval_config.json` (5).

```bash
python -m browsecomp_plus.eval.run_agent_eval --check
python -m browsecomp_plus.eval.run_agent_eval --limit 1 --dry-run
python -m browsecomp_plus.eval.run_agent_eval
```

`--check` calls AWS STS, not the model. To resume, repeat the same command
with the original `--run-id`; completed queries are skipped. Search and
generation are stochastic, so scores will vary across runs.

Outputs go to `browsecomp_plus/data/eval/runs/<run-id>/`. Summarize a run
with:

```bash
python -m browsecomp_plus.eval.report \
  browsecomp_plus/data/eval/runs/<run-id>
```

## Scoring

Evidence and gold nDCG@10 use binary gains, deduplicate document ids, and
score the first ten ranks. The retrieval stop call is not scored. Recall@10
is relevant documents in the top ten divided by all relevant documents.
Trace-oracle nDCG ranks all relevant documents seen during retrieval first;
retention measures how many of those the ranking kept.

Dataset attribution is in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
