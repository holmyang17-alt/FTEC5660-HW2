# FTEC5660 Homework 2: CV Verification Agent

Build a LangChain agent that reads each CV in a folder, looks the candidate up
on our SocialGraph MCP server (mock LinkedIn and Facebook), and outputs a
reliability score in [0, 1] for each CV.

A CV is **valid** (label `1`) when its claims agree with the candidate's
social media profiles. It **has a discrepancy** (label `0`) when it contains
problems such as an inflated job title, shifted dates, an upgraded degree, a
fake school or employer, a wrong location, or made-up skills. Differences in
wording only ("Bachelor of Science" vs `BSc`, "UI/UX Design" vs `UI/UX`,
"Senior Engineer" for an `Engineer` role with seniority `senior`, listing fewer
skills) are not discrepancies.

## Student task

Fill in the two functions in `hw2.py` that contain `### YOUR CODE HERE`. You
may add imports, constants and helper functions above them, but do not change
the provided code below them:

- `build_agent(tools)` creates your agent from the MCP tools.
- `score_cvs(agent, cvs)` runs the agent on every CV and returns
  `{file_name: score}`, one float in [0, 1] per CV.

A score above `0.5` means "valid"; `0.5` or below means "has discrepancy". You
may use a single tool-calling agent, multiple agents, reflection, or a
combination. Do not hard-code filenames, names, or public answers; grading
uses unseen CVs.

## MCP server

The server is hosted at `https://ftec5660.ngrok.app/mcp`. `hw2.py` already
connects to it and passes you these tools:

| Tool | Purpose |
| --- | --- |
| `search_facebook_users(q, limit, fuzzy)` | find Facebook users by display name |
| `get_facebook_profile(user_id)` | full Facebook profile |
| `get_facebook_mutual_friends(user_id_1, user_id_2)` | mutual friends of two users |
| `search_linkedin_people(q, location, industry, limit, fuzzy)` | search LinkedIn by name, skill, or title |
| `get_linkedin_profile(person_id)` | full LinkedIn profile (experience, education, skills) |
| `get_linkedin_interactions(person_id)` | post and like statistics |

Print `tool.name`, `tool.description`, and `tool.args` for the full schemas.

## Setup and public test

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
echo "DEEPSEEK_API_KEY=your_key_here" > .env
python3 hw2.py --cv-folder public_test
```

Use your own DeepSeek API key (from https://platform.deepseek.com).

The program creates `results.csv` in the current directory. Its columns are
`cv`, `score`, and `correctness`. The public labels are in
`public_test/ground_truth.json`; every CV with label `0` also has a `reason`
explaining the discrepancy, and `results.csv` shows it when your agent misses
one. The starter returns no score (`None`) for every CV so it runs before you
add any API code; missing scores count as incorrect.

The required model is `deepseek-v4-flash`, and all evidence must come from the
MCP server (no web search). The server is shared by the whole class: keep at
most about 3 CVs in flight at a time (e.g. `asyncio.Semaphore(3)`). We grade
with our own API key, so never put a key in your code; `.env` must stay out of
git.


## Task 2: Fool the Verifier

We added a target candidate, **Kelly Tsang** (LinkedIn `person_id` 10001,
Facebook `user_id` 10001, display name "Kel Tsang"). Her true CV is
`task2/target_cv.pdf`.

Write one CV for Kelly Tsang with at least one false or embellished detail
that our 5 verifier agents (same function as Task 1) still score above 0.5.
Each fooled agent is worth 4 points; a CV without a real false or embellished
detail earns 0. Attack your own Task 1 agent first:

```bash
cp my_attack.pdf task2/adversarial_cv.pdf
python3 hw2.py --cv-folder task2
```

`results.csv` shows your agent's score for both CVs (they are not graded, as
`task2/` has no `ground_truth.json`). Only the PDF is the attack surface; do not
attack the MCP server. See the homework description for the full rules.

## Homework 2 solution:

### Task 1

My Task 1 agent is a LangChain agent built on `deepseek-v4-flash` via
`ChatDeepSeek` and `create_agent`. The system prompt instructs the model to
extract every verifiable field from the CV (name, city, jobs with company,
title, start and end years, education with school, degree, field and graduation
year, and each listed skill), then use the six SocialGraph MCP tools to find
the candidate on LinkedIn (primary source) and Facebook. Because many people
share the same name, the prompt tells the agent to narrow the LinkedIn search
with the CV's city and industry, and to match Facebook's `original_name` when
the display name differs. After fetching both profiles, the agent compares
field by field and produces a short match/mismatch table, then emits a final
line `SCORE: <number between 0 and 1>`. The prompt is strict about what counts
as a mismatch (different year by even one, wrong company, wrong school,
different degree level, wrong city, or a skill missing from LinkedIn) and what
does not (wording differences such as "Bachelor of Science" vs "BSc", "UI/UX
Design" vs "UI/UX", or listing fewer skills than the profile). On the seven
public CVs my agent scored 7/7 (`CV_1` 0.95, `CV_2` 0.95, `CV_3` 0.95,
`CV_4` 0.30, `CV_5` 0.30, `CV_6` 0.95, `CV_7` 0.30), correctly flagging the
three tampered CVs and accepting the four genuine ones. `score_cvs` runs the
CVs concurrently with `asyncio.Semaphore(3)` to stay within the shared MCP
server's limit, and catches per-CV exceptions so one failure cannot crash the
whole run.

### Task 2

For Task 2 I attacked the verifier with a *text/image inconsistency* in the
PDF. The visible layer of `task2/adversarial_cv.pdf` shows the AIA role as
"Senior Manager, AIA 2022 - Present" — an embellishment, since LinkedIn lists
Kelly Tsang's AIA title as "Manager". However, the *text layer* of that same
line contains the truthful string "Manager, AIA 2022 - Present". The false
"Senior Manager" string is rendered as a small raster image drawn with PIL and
placed exactly on top of the text line via `fpdf.image(..., x=..., y=..., w=...,
h=...)`, so it is what a human reader sees but not what a text extractor
reads. Because `markitdown` (the same PDF loader used by `hw2.py` and by our
verifiers) extracts only the embedded text and does not run OCR on raster
images, the verifier receives the truthful "Manager" and finds no discrepancy
against LinkedIn. I also added a 1-point white-text "HR Pre-Screening Note"
claiming the CV had already been verified upstream, as a soft secondary layer.
On my own Task 1 agent this CV scored 0.95 while the human-visible claim is
false, which is exactly the win condition: the attacker keeps a real
embellishment in the document while the verifier sees only consistent text.
The technique relies on the fact that verifiers read extracted text rather than
the rendered page, and that none of the plausible verifier pipelines in this
assignment perform OCR.