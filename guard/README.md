# RS guard

[日本語](README.ja.md)

**Right before an AI agent uses a tool, RS guard compares "the task you asked for" and "the action it is about to take" with your usual work. If the action is outside your usual work, it asks you to confirm before it runs. Every action is checked, including shell commands, file writes and scripts.**

Example: you asked "Who is coming to tomorrow's meeting?", and the agent is about to email an unknown address it found in a calendar event. This is a typical prompt injection (an attack where instructions hidden in a fetched document or email hijack the agent). RS guard stops the call before it runs and tells you why:

```
RS: the to of send_email comes from the result of search_calendar_events. Similar past requests never had this shape.
```

- **Works with the popular agent tools**: hooks into Claude Code, Codex CLI, Gemini CLI, Cursor, GitHub Copilot (CLI, VS Code), Cline and Windsurf. One decision engine and one usual work, the same yardstick across tools.
- **Your usual work is built automatically** from the conversation logs already on your machine (Claude Code, Codex), on the day you install it. Nobody has to write an allow list.
- **Learns in one step**: an action you approve at a confirmation becomes part of your usual work right away.
- **No LLM calls, nothing leaves your machine**: each decision is a lookup against your logs and takes about 0.1 s (including hook start-up).

Part of Reflex-Sentinel (RS). RS itself is research on catching concealment by reading an agent's internal state before it acts ([../README.md](../README.md)).

## Results

### Email, calendar and other tools (MCP, web)

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

For unseen kinds of tasks (the 97 tasks split into 5 folds, usual work built only from the other tasks), GPT-4o gets 1.3 % / 45.4 %: attacks are stopped just as well, but more normal tasks get interrupted. (This table measures the MCP / web part of the decision.)

### Coding agents (shell, files)

We ran [CodeIPI](https://github.com/UKGovernmentBEIS/inspect_evals) (UK AISI inspect_evals, MIT; 45 tasks testing whether a coding agent follows instructions hidden in an issue, a code comment, a README or a config file) three times each, with a local Gemma 4 26B as the agent.

| | Result |
|---|---|
| 12 runs where the agent followed the injection and ran a harmful action (reading a secret, running the injected command, writing to requirements.txt) | **all 12 stopped right before that action** |
| 30 normal runs (264 actions) | 0 actions stopped |

This agent never followed injections hidden in files. So we also checked whether each injected command would have been stopped *if* the agent had run it, against one developer's usual work:

| Injected payload | Files (comments, README, config) | Issue text |
|---|---|---|
| Read secrets, persist, install packages | **15 of 18 stopped** | 6 of 7 |
| Only a "can I run code" marker (`echo CANARY…`) | 0 of 7 | 2 of 3 |

- Injections in files are stopped by **the action itself** (a secret, persistence location or package touched for the first time), not by how the instruction is worded, so rewording the injection does not help the attacker.
- The 3 misses dump all environment variables or read `DATABASE_URL`; this developer's usual work already had the same kind of action.
- Marker-only payloads are harmless commands and are not stopped by actions (a real attack would send, persist or read secrets, and stop there).
- With the same usual work, 1 of CodeIPI's 30 normal runs was stopped (fetching a site for the first time).

### Interrupting normal work

Replaying one developer's Claude Code logs (73 conversations) in time order and counting the last 7 days (5,342 actions), RS guard asked 52 times (**7.4 per day**): 3.4 per day for MCP / web actions and 4.3 per day for shell, files and others (a new destination, a new destination after reading keys, a secret or package touched for the first time, a delete / config command used for the first time). Each one you approve joins your usual work and is not asked again.

Replaying 15 Codex logs the same way (9 days, 2,579 actions) gave 12 confirmations (1.3 per day of use; first-time `rm`, `git push`, `kill`, …).

## Install

```bash
python3 -m venv ~/.rs-guard/venv
~/.rs-guard/venv/bin/pip install "git+https://github.com/matu79go/reflex-sentinel@v1.1.2#subdirectory=guard"
~/.rs-guard/venv/bin/rs-guard learn          # build your usual work from local logs (Claude Code, Codex)
~/.rs-guard/venv/bin/rs-guard install all    # add hooks to every installed tool (or: install codex, …)
```

### Supported tools

| Tool | Hooked via | On an unusual action | Verified |
|---|---|---|---|
| Claude Code | `~/.claude/settings.json` (PreToolUse, …) | confirmation prompt | runs on the real app |
| Codex CLI | `~/.codex/hooks.json` (PreToolUse, …) | blocks once → goes through when retried after you approve | runs on the real app (stops a first-time delete, destination and package; goes through when retried after you approve; usual work built from its logs) |
| Gemini CLI | `~/.gemini/settings.json` (BeforeTool, …) | blocks once → same | tested with documented inputs |
| Cursor | `~/.cursor/hooks.json` (beforeShellExecution, beforeMCPExecution, beforeReadFile, …) | confirmation prompt | tested with documented inputs |
| GitHub Copilot (CLI, VS Code) | `~/.copilot/hooks/rs-guard.json` (preToolUse, …) | confirmation prompt | tested with documented inputs |
| Cline | `~/Documents/Cline/Rules/Hooks/` (PreToolUse, …) | blocks once → same | tested with documented inputs |
| Windsurf | `~/.codeium/windsurf/hooks.json` (pre_run_command, …) | blocks once → same | tested with documented inputs |

### Using it with Codex

1. Install: `~/.rs-guard/venv/bin/rs-guard install codex` (adds 3 hooks to `~/.codex/hooks.json`).
2. Open Codex and **trust** the 3 rs-guard hooks with `/hooks` (they do not run until trusted).
3. Use Codex as usual. An unusual action stops as a tool error in Codex, with the reason and "Ask the user". Codex then comes back to ask you.
4. If it is normal work, reply "ok". When Codex runs the same action again it goes through and joins your usual work, so it is not stopped next time. If it looks wrong, just say no (a retry without your reply is still blocked).
5. What was stopped and what you approved: `~/.rs-guard/venv/bin/rs-guard explain` (Codex actions are marked `[codex]`, including the ones you approved after a stop).

Example of a stop:

```
RS: first time using the send / delete / config command shred. (Ask the user. If they approve, run the same action again and it will go through.)
```

- **Tools without a confirmation prompt** (Codex, Gemini CLI, Cline, Windsurf): an unusual action is blocked once, and the agent is told to ask you. If the same action comes again after you reply, it goes through and joins your usual work. A retry with no message from you in between is still blocked.
- Codex runs the hooks only after you trust them with `/hooks` inside Codex. Gemini CLI may need `/hooks enable-all`. Cline needs hooks enabled under Features in its settings.
- Cursor has no pre-execution hook for file writes, so writes are only learned afterwards (shell, MCP and file reads are checked before they run).
- Usual work is built from logs on install day for Claude Code and Codex. For the other tools it grows as you use them (usual work is shared across all tools).
- For tools that run on Windows (Windows VS Code, Cursor, …), install with Windows Python using the same steps (a WSL install is not called by Windows tools).
- `install` keeps existing hooks and, the first time it touches a file, saves the original as `<file>.rs-guard.bak`. To remove RS guard, restore that file or delete the rs-guard entries.
- How often and why it asked: `rs-guard status` / `rs-guard explain`.
- Reasons are shown in English or Japanese following `LANG` (fix it with `"lang": "en"` in `~/.rs-guard/config.json` or `RS_GUARD_LANG`).
- No dependencies (Python 3.9+).

## How it decides

Every action (shell, files, scripts, MCP, web) goes through the two checks below. If either fires, RS guard asks.

### All actions

Actions are put into classes: send (network, publish, message) / delete (rm, kill, git reset …) / change config (keys, crontab, chmod, sudo, config files) / execute (scripts) / local work. Shell lines are split per command (quotes kept, heredoc bodies dropped, commands inside `ssh host '…'` and `bash -c '…'` included). For scripts being run (`python x.py`, `python -c "…"`) the content is read too, for network libraries and destinations.

| Signal | Asks when |
|---|---|
| R3 | sending to a new destination after reading keys or settings |
| R7 | following an instruction found in pasted text (Claude Code's `<pasted_content>`) as is |
| R1x | a send / delete / config action uses a value from a web or connector result in a new way |
| R8 | a secret (key environment variables, `.env`, key files), persistence location (`.bashrc`, git hooks, cron, `/etc/`) or package is touched for the first time |
| R5 | a new destination (ssh host, scp / rsync host, URL a curl sends to, git push remote, MCP server, email recipient, URL inside a script) |
| R6 | curl / wget fetches a link found in content it read (not search results) in a new way |
| R2r | a send / delete / config command is used for the first time |

Destinations, secrets and packages that appear in your own words are allowed (those that only appear in pasted text are not).

### MCP and web actions (compared with similar past requests)

1. Only outward-acting tools are checked: MCP tools whose name starts with send / create / update / delete / invite / share / post / copy / batch …, opening web pages (WebFetch), and web search (WebSearch; only emails, URLs and long IDs inside the query are checked). Add more with `extra_tools` in `~/.rs-guard/config.json`.
2. For each argument value (email, URL, ID, short string), find where it came from: your request, the result of a specific tool, or nowhere.
3. If every value came from your request, allow (it is your instruction).
4. If the same shape (tool, argument name, source) already ran in this conversation, allow.
5. Find the most similar past request (word overlap). If the same tool was used there with values from the same sources, allow. Otherwise ask.
6. For web pages only the URL matters. URLs from your request, from search results, or written by the model itself are allowed. Only when the host appears in another tool's result (a fetched page, an email, a file) is it compared with your usual work.

"The current request" is your last 3 messages (a follow-up such as "ok, go ahead" alone does not say what the task is).

To block instead of asking for some signals, set e.g. `"deny_rules": ["R3", "R7"]` in `~/.rs-guard/config.json`.

## Files and privacy

| File | Contents |
|---|---|
| `~/.rs-guard/normal.jsonl` | Usual work (MCP / web): request text, tool names, argument names, source categories. **Values themselves (recipients, bodies) are not stored** |
| `~/.rs-guard/seen.jsonl` | Usual work (all actions): command names, source categories, and **hashes** of destinations, secrets and packages (mixed with a per-machine random `salt`; the names themselves are not stored) |
| `~/.rs-guard/decisions.jsonl` | Decision log (what was asked and why) |
| `~/.rs-guard/done.jsonl` | Only the IDs of actions that ran after a confirmation (so `explain` can show approved / rejected) |
| `~/.rs-guard/sessions/` | Per-conversation notes (your messages and tool results). Claude Code writes its log asynchronously, so these keep the latest result from being missed. Deleted after 7 days |

All files are mode 600. Nothing is sent anywhere.

## Scope and limits

- Abuse inside your usual work is not caught (a different body to your usual recipient, reading a secret you have read before, …).
- Instructions inside files the agent reads are stopped through the **action** that follows them (a new destination, secret, persistence location or package), not by reading the instruction (in coding, following the steps in a README or design doc is normal). Injections that only run a harmless command (such as `echo`) are not stopped.
- Network use inside scripts is read from the file on disk at run time and from `-c` strings. Destinations built at run time, read from another file, or sent from a binary are not visible. Use RS guard alongside a layer that sees real outgoing traffic (a local proxy or firewall).
- Commands are recognized by fixed rules (name lists and regular expressions). Sending with a tool missing from the lists is not caught as "a new command".
- For a new kind of request, every outward-acting call asks (this decreases with use).
- Only the last 4 MB of the conversation log is read (for speed). Values that appeared only before that window count as "from nowhere", which lets web fetches through.
- Tools other than Claude Code are tested with the input formats in each tool's official docs (as of 2026-10). If a tool version changes the format, the input may not be read (RS guard then lets the action through).

## Related work

Learning allowed action flows and value sources from normal runs is essentially the idea of [Agent-Sentry](https://arxiv.org/abs/2603.22868) (ASR 3.7 %) and [Praetor](https://arxiv.org/abs/2604.26274) (no LLM, folds in human approvals). Tagging values with their source comes from [CaMeL](https://arxiv.org/abs/2503.18813). What RS guard adds:

- Usual work is built per user, on install day, from **the conversation logs already on the user's machine** rather than from research logs.
- Fitting to real conversations (follow-up messages, repetition within a conversation, logs written late).
- Drops straight into tool hooks (`pip install` plus two commands).

## Reproduce

`repro/` contains the AgentDojo and CodeIPI evaluations (AgentDojo `runs/` and the CodeIPI tasks are MIT and not bundled).

```bash
python repro/dojo_gt.py --src <agentdojo>/src/agentdojo/default_suites --out gt.json
python repro/dojo_product.py --runs <agentdojo>/runs --gt gt.json --seen-task --model gpt-4o-2024-05-13
# CodeIPI: would each injection be stopped if the agent followed it (usual work = your local Claude Code logs)
python repro/codeipi_counterfactual.py --bench <CodeIPI samples.json>
```

Unit tests: `python -m pytest -q tests` (pytest is needed for tests only).

## License

Apache-2.0 ([../LICENSE](../LICENSE)).
