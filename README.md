# wans-ple: a small, measured harness for PLE

Entry for the Kaggle competition *Google - Agent Harness on Physical-Engineering's Last Exam*.
Solo entry. The goal is a small harness whose every change is measured on its own.

## What it is

A single tool-calling loop (`bash`, `read_file`, `write_file`) adapted from the
`minimal` sample of [google-research/ple](https://github.com/google-research/ple),
plus four additions. Each addition is one commit and was tested separately:

1. **MCP support**, adapted from the `custom-openai` sample. Tools a task serves over MCP
   (documents, spreadsheets, email, CAD) are offered to the model.
2. **Requirements-first system prompt.** List the requirements and their sources before
   computing, mark assumed numbers as assumptions, prefer the latest issued revision of a
   document, and report unmet requirements honestly in the final message.
3. **Deliverable guard.** Before the final answer is accepted, the harness checks that the
   files named by absolute path in the task instruction exist (at most 2 nudges).
4. **Old tool-result trimming.** Tool results older than the newest 6 and longer than 3,000
   characters are cut to 1,500 characters; the full text is saved under `/tmp/tool_results/`
   in the task container.

All four are general behaviors. There is no per-task logic, and the harness never reads
graders or reference files.

## Results so far

Everything below is local, on one task (`wind-diaphragm-shear`, graded by `checks`), one run
per row, with `gemini/gemini-3.5-flash-lite` from the free tier of Google AI Studio unless
stated otherwise. One sample per row is far too little to compare scores, so the table
mainly shows what each change did to tokens and failures.

| Run | Harness | Score | Requests | Input tokens / request | HTTP 429 | What happened |
|---|---|---|---|---|---|---|
| 1 | `minimal`, `gemini-3.5-flash` | 0.000 | n/a | n/a | 5 | Free-tier daily quota exhausted; harness gave up |
| 2 | `minimal` | 0.043 | 37 | 45.6k | 3 | Valid zip; hard constraints violated |
| 3 | + MCP + requirements prompt | 0.000 | 47 | n/a | 7 | Package built, but no file at the graded path |
| 4 | + deliverable guard | 0.037 | 68 | 60.2k | 21 | File in place; guard never fired |
| 5 | + trim old results | 0.029 | 95 | 18.7k | 0 | Input per request down about 69% |
| 6 | + "extract each document once" | 0.011 | 89 | 21.3k | 4 | No reduction; **reverted** |

Observations that held up in the logs:

- The model reads the right documents, but its designs violate hard constraints, and its
  final message still claims success.
- Relative paths are a trap: every `bash` call is a fresh shell, so `cd` does not persist.
- Text written through shell quoting can break silently (backticks were swallowed).

Not established: whether any change improves the score. The scores sit between 0.000 and
0.043 and the spread of a single sample covers all of them.

## Limits and status

- `model_api` is a local model, not a `google-cloud/<id>` model, so `kit check` cannot print
  `ready to submit` yet.
- The development machine (4 CPU, 8 GB RAM) cannot run all 29 dev tasks, so artifacts will be
  partial.
- Total input tokens are reported both with and without the automatic cache, because trimming
  lowers total tokens while also lowering the cache share.

## Reproduce

```bash
git clone https://github.com/google-research/ple && cd ple
uv tool install harbor==0.20.0 --with websockets
# put the model key and HF_TOKEN in an env file outside the entry (never in this repo)
uv run python -m kit check ../ple-agent-harness --env-file local.env
uv run python eval/run.py ../ple-agent-harness --env-file local.env --tasks wind-diaphragm-shear --no-export
```

## License

The repository is MIT licensed (`LICENSE`). `serve.py` and `taskbridge.py` derive from files
in google-research/ple that are Apache-2.0 licensed (Copyright 2026 Google LLC); their
license headers are kept.
