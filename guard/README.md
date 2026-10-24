# RS guard

[日本語](README.ja.md)

**Right before an AI agent uses a tool, RS guard compares "the task you asked for" and "the action it is about to take" with your usual work. If the action is outside your usual work, it asks you to confirm before it runs.**

Example: you asked "Who is coming to tomorrow's meeting?", and the agent is about to email an unknown address it found in a calendar event. This is a typical prompt injection (an attack where instructions hidden in a fetched document or email hijack the agent). RS guard stops the call before it runs and tells you why:

```
RS: the to of send_email comes from the result of search_calendar_events. Similar past requests never had this shape.
```

- **Your usual work is built automatically** from the conversation logs already on your machine (`~/.claude/projects/` for Claude Code), on the day you install it. Nobody has to write an allow list.
- **Learns in one step**: an action you approve at a confirmation becomes part of your usual work right away.
- **No LLM calls, nothing leaves your machine**: each decision is a lookup against your logs and takes 50–60 ms.

Part of Reflex-Sentinel (RS). RS itself is research on catching concealment by reading an agent's internal state before it acts ([../README.md](../README.md)).

## Results

**Stopping attacks**: measured on the public run logs of [AgentDojo](https://github.com/ethz-spylab/agentdojo) (the standard prompt-injection benchmark for agents: 97 tasks over email, calendar, banking, Slack and travel). The attack success rate (ASR) is the share of attacked runs in which the attacker's intended action was executed. Numbers are for repeated work (the same kind of task was done before).

| Model | No defense: ASR / task success | RS guard: ASR / task success |
|---|---|---|
| GPT-4o | 47.7 % / 69.1 % | **1.4 % / 69.1 %** |
| Claude 3.5 Sonnet | 33.9 % / 79.4 % | 3.8 % / 79.4 % |
| Claude 3.7 Sonnet | 5.0 % / 88.7 % | 0.3 % / 87.6 % |
| Gemini 1.5 Pro | 28.6 % / 46.4 % | 5.2 % / 44.3 % |
| Gemini 2.0 Flash | 14.1 % / 43.3 % | 0.4 % / 43.3 % |
| GPT-4 (0125) | 56.3 % / 66.0 % | 4.5 % / 64.9 % |
| GPT-4o mini | 27.2 % / 68.0 % | 2.1 % / 67.0 % |
| Llama 3.3 70B | 23.1 % / 57.7 % | 1.6 % / 57.7 % |
| Command R+ | 4.5 % / 24.7 % | 3.8 % / 22.7 % |

For unseen kinds of tasks (the 97 tasks split into 5 folds, usual work built only from the other tasks), GPT-4o gets 1.3 % / 45.4 %: attacks are stopped just as well, but more normal tasks get interrupted.

**Interrupting normal work**: replaying one developer's Claude Code logs (72 conversations over 28 days) in time order, RS guard asked for confirmation on 17 of 102 outward-acting calls (0.6 per day). Almost all of them were genuinely new kinds of writes (for example, the first few writes in the first conversation that edited a spreadsheet).

## Install (Claude Code)

```bash
python3 -m venv ~/.rs-guard/venv
~/.rs-guard/venv/bin/pip install "git+https://github.com/matu79go/reflex-sentinel@v1.0.0#subdirectory=guard"
~/.rs-guard/venv/bin/rs-guard learn            # build your usual work from local logs
~/.rs-guard/venv/bin/rs-guard install claude   # add 3 hooks to ~/.claude/settings.json
```

- `install` keeps existing hooks and saves the original settings to `~/.claude/settings.json.rs-guard.bak`. To remove RS guard, restore that file or delete the three rs-guard entries under `hooks`.
- How often and why it asked: `rs-guard status` / `rs-guard explain`.
- Reasons are shown in English or Japanese following `LANG` (fix it with `RS_GUARD_LANG=en` / `ja`).
- No dependencies (Python 3.9+).

## How it decides

1. Only outward-acting tools are checked: MCP tools whose name starts with send / create / update / delete / invite / share / post …, and opening web pages (WebFetch). Add more with `extra_tools` in `~/.rs-guard/config.json`.
2. For each argument value (email, URL, ID, short string), find where it came from: your request, the result of a specific tool, or nowhere.
3. If every value came from your request, allow (it is your instruction).
4. If the same shape (tool, argument name, source) already ran in this conversation, allow.
5. Find the most similar past request (word overlap). If the same tool was used there with values from the same sources, allow. Otherwise ask.
6. For web pages only the URL matters. URLs from your request, from search results, or written by the model itself are allowed. Only when the host appears in another tool's result (a fetched page, an email, a file) is it compared with your usual work.

"The current request" is your last 3 messages (a follow-up such as "ok, go ahead" alone does not say what the task is).

## Files and privacy

| File | Contents |
|---|---|
| `~/.rs-guard/normal.jsonl` | Usual work: request text, tool names, argument names, source categories. **Values themselves (recipients, bodies) are not stored** |
| `~/.rs-guard/decisions.jsonl` | Decision log (what was asked and why) |
| `~/.rs-guard/sessions/` | Per-conversation notes (your messages and tool results). Claude Code writes its log asynchronously, so these keep the latest result from being missed. Deleted after 7 days |

All files are mode 600. Nothing is sent anywhere.

## Scope and limits

- **Shell (Bash) and file writes are not checked.** Use RS guard **alongside** a rule-based guard (command allow/deny lists) for things like sending data out with `curl`. Claude Code runs multiple hooks side by side.
- Abuse inside your usual work (sending a different body to your usual recipient in your usual way) is not caught.
- For a new kind of request, every outward-acting call asks (this decreases with use).
- Only the last 4 MB of the conversation log is read (for speed). Values that appeared only before that window count as "from nowhere", which lets web fetches through.
- v1.0 supports Claude Code only.

## Related work

Learning allowed action flows and value sources from normal runs is essentially the idea of [Agent-Sentry](https://arxiv.org/abs/2603.22868) (ASR 3.7 %) and [Praetor](https://arxiv.org/abs/2604.26274) (no LLM, folds in human approvals). Tagging values with their source comes from [CaMeL](https://arxiv.org/abs/2503.18813). What RS guard adds:

- Usual work is built per user, on install day, from **the conversation logs already on the user's machine** rather than from research logs.
- Fitting to real conversations (follow-up messages, repetition within a conversation, logs written late).
- Drops straight into tool hooks (`pip install` plus two commands).

## Reproduce

`repro/` contains the AgentDojo evaluation (AgentDojo `runs/` are MIT and not bundled).

```bash
python repro/dojo_gt.py --src <agentdojo>/src/agentdojo/default_suites --out gt.json
python repro/dojo_product.py --runs <agentdojo>/runs --gt gt.json --seen-task --model gpt-4o-2024-05-13
```

Unit tests: `python -m pytest -q tests` (pytest is needed for tests only).

## License

Apache-2.0 ([../LICENSE](../LICENSE)).
