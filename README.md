# Loan Eligibility Assistant (demo prototype)

Built for a business-leader demo: a streaming chat assistant that answers
loan pre-qualification questions by retrieving the relevant internal
eligibility policy (RAG) and giving a grounded, consistent answer — instead
of a staff member manually checking rules by hand.

## Business problem -> build -> success metrics

**Business problem:** pre-qualification takes staff time and yields
inconsistent answers.

**What this builds:**

- A streaming chat app (`loan_assistant_frontend.html`) talking to a FastAPI
  backend (`loan_assistant_main.py`) over Server-Sent Events.
- RAG over eligibility rules (`loan_assistant_rules.json`): a small,
  dependency-free TF-IDF retriever finds the right policy document, and a
  deterministic composer builds the answer directly from that policy's
  structured thresholds — so the numbers in the answer always match the
  numbers in the source policy.
- CI/CD with eval gates: `loan_assistant_promptfooconfig.yaml` runs a fixed
  regression suite against the live API; `loan_assistant_github_workflow.yml`
  wires that into a required GitHub PR check.
- Audit logs: every question, retrieved sources, verdict, and latency is
  appended to `audit_log.jsonl` and viewable in the app's "Audit & Metrics" tab.

**Success metrics -> where they're measured:**

| Metric | Where |
|---|---|
| Eligibility answer accuracy | `[accuracy]`-tagged promptfoo tests check the verdict (eligible / not_eligible / needs_more_info) against known-correct expectations for each loan type. Pass rate surfaces in the app's Metrics tab once you've run an eval. |
| Faithfulness | `[faithfulness]`-tagged promptfoo tests check that every number the assistant states matches the number in the actual policy doc (e.g. it can't say "700" when the mortgage policy says "680"). Because generation here is template-based from the same thresholds used for retrieval, faithfulness failures would indicate a real bug, not model hallucination — worth knowing when you swap in a real LLM later (see Limitations). |
| Blocked-merge rate on regression | Tracked by GitHub over time once `loan_assistant_github_workflow.yml` is a required status check: every PR where `promptfoo eval` fails gets blocked automatically. That rate lives in your CI history/branch-protection dashboard, not in this app. |
| Latency | Live: `/api/metrics` computes avg/p95 latency from the audit log. CI-time: promptfoo records per-test latency in `promptfoo_results.json`, and the `latency` assertion in the eval config fails any test over 2000ms. |

## Run it

```bash
pip install -r loan_assistant_requirements.txt
uvicorn loan_assistant_main:app --reload --port 8000
```

Open http://localhost:8000 — you'll land on the Chat tab with sample
questions you can click for the demo. The "Audit & Metrics" tab shows live
latency/verdict stats and (once you've run an eval) the accuracy/faithfulness
pass rates from CI.

## Demo script (suggested flow for the leader meeting)

1. Click the first sample question (personal loan, all criteria met) — show
   the streaming answer, the green "eligible" badge, and the cited source doc.
2. Click the second sample (low credit score) — show the red "not_eligible"
   badge and the specific failing criterion in the answer.
3. Ask a partial question (e.g. just "Can I get an auto loan?") — show the
   amber "needs_more_info" badge and how it tells you exactly what's missing,
   instead of guessing.
4. Switch to "Audit & Metrics" — show the full audit trail (every question,
   verdict, and latency logged) and frame it as the compliance/consistency
   story: every answer is traceable to a specific policy document and a
   specific reasoning trail, unlike an unstructured staff judgment call.
5. If you've run `npx promptfoo eval` beforehand, show the accuracy/
   faithfulness pass rate in that same tab and explain the CI gate: this
   suite reruns on every pull request, and a regression blocks the merge.

## Running the eval gate locally

```bash
# terminal 1
uvicorn loan_assistant_main:app --port 8000

# terminal 2 (needs Node.js for npx)
npx promptfoo eval -c loan_assistant_promptfooconfig.yaml --output promptfoo_results.json
```

Refresh the Metrics tab afterward to see the pass rates. To wire this into
GitHub: `mkdir -p .github/workflows && cp loan_assistant_github_workflow.yml .github/workflows/eval-gate.yml`,
then mark the `promptfoo-eval` job as a required check in branch protection
so failing evals actually block merges.

## Limitations (say these out loud in the demo, don't let leaders assume more than what's built)

- **No real LLM call.** Per your call to keep this offline/controlled for
  the demo, the "generation" step is a deterministic template built from the
  same structured thresholds used for retrieval — not a language model. This
  is why faithfulness is essentially guaranteed here; a real LLM integration
  would need the promptfoo faithfulness checks to catch actual hallucination
  risk, which this build doesn't exercise. Swapping in a real model means
  replacing `build_answer()` in `loan_assistant_main.py` with a call to your
  LLM provider, passing the retrieved chunks as context.
- **Retrieval is TF-IDF, not embeddings.** Dependency-free and fully
  deterministic (good for a demo), but a production RAG system would likely
  use an embedding model for better semantic matching on paraphrased
  questions.
- **Streaming is simulated.** The full answer is composed first, then
  streamed word-by-word to the UI for the live-typing effect — there's no
  token-by-token generation happening underneath.
- **promptfoo HTTP provider syntax may need a small tweak.** It has shifted
  slightly across promptfoo versions; if `npx promptfoo eval` errors on
  parsing the config, check the `body`/`transformResponse` keys against
  https://promptfoo.dev/docs/providers/http for your installed version. The
  test cases and assertions themselves don't depend on that syntax.
- **Rules are synthetic.** Five sample policy documents (personal loan, auto
  loan, mortgage, credit card, general KYC) with realistic-looking but made
  up thresholds — swap in `loan_assistant_rules.json` for your actual policy
  content when ready.
- **Single-process, no auth.** Fine for a demo; add authentication and a
  real database before any real usage.

## Files

- `loan_assistant_main.py` — FastAPI backend (retrieval, answer logic, streaming, audit log, metrics)
- `loan_assistant_rules.json` — the eligibility policy corpus (RAG source documents)
- `loan_assistant_frontend.html` — chat UI + audit/metrics dashboard (single file, no build step)
- `loan_assistant_requirements.txt` — Python dependencies (FastAPI, uvicorn, pydantic)
- `loan_assistant_promptfooconfig.yaml` — accuracy + faithfulness regression suite
- `loan_assistant_github_workflow.yml` — CI eval-gate workflow (copy to `.github/workflows/eval-gate.yml`)
- Created at runtime: `audit_log.jsonl`, `promptfoo_results.json`
