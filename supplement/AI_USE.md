# Use of AI-assisted technologies

This note follows the AAMAS 2027 policy: where AI tools were used to create hypotheses or methods, including experimental design, the paper or its supplementary material gives the tool, its version and the prompts, and explains how the interaction contributed.

## Tool

Claude Opus 5.5 (Anthropic), used through Claude Code: interactive sessions with the authors, sub-agents started by those sessions, and separate sessions that executed experiment runbooks on rented GPUs. The repository was developed in such sessions from the start of the project.

## What the AI contributed

- **Code.** The library (`ptcdiag/`), run scripts and analysis scripts, with unit tests (`tests/`).
- **Experimental design.** Experiments B (teams of agents on choose-N), C (teams on BFCL; the closing token in the slot; one slot longer) and D (format-tolerant slots; position agents) were proposed by the AI in response to the authors' questions and to mock reviews, and run only after the authors approved each one and its cost. The runbooks given to the executing sessions are included (`prompts/`).
- **Analysis.** Statistics, tables and figures, all computed by scripts from the raw records (`scripts/reproduce_paper.sh`).
- **Writing.** Drafts and revisions of every section.
- **Checks.** Sub-agents checked every reference against publisher, proceedings or arXiv pages, audited every number in the text against the raw records, and wrote mock reviews whose findings led to corrections (for example, a stricter chimera detector, Holm's correction, and experiment D).

## How the interaction worked

The authors set the research question, the venue, the budget and the rules for the work: numbers that were not run are marked as such, results that contradict a prediction are reported as they are, references must be checked against their sources, and any spending beyond the budget needs approval. In multi-turn conversations the AI proposed structures, experiments and text, and the authors decided. Each approved experiment was written up as a runbook with predictions stated in advance and gates that stop the run on unexpected behaviour, and executed by a separate session that reported back.

We do not reproduce the conversations. The authors' main decisions were:

- **Scope.** The authors asked twice whether the work was close enough to the conference's focus on agents. In response the AI proposed comparing the slots of one dLLM agent with teams of LLM agents; the authors approved experiments B and C for this and chose the paper's structure and title from the AI's proposals.
- **References.** The authors required every reference to be checked against its source, and declined a set of candidate references that the checks had found.
- **Experiments.** The authors approved or declined each proposed experiment and its cost. After the mock reviews they approved experiment D, which tests two objections the reviews raised: the type mask of the slots, and the information the team agents lack.
- **This note.** The authors decided to describe the use of AI in the supplementary material, beyond the statement in the paper.

## Prompts of the executing sessions

In `prompts/` (repository addresses and branch names removed):

- `LOCAL_CLAUDE_PROMPT.md`: the first GPU runs (one canvas, all models and conditions of Sections 4–8 except those below).
- `EXP_B_PROMPT.md`: teams of agents on choose-N (Section 5).
- `EXP_C_PROMPT.md`: teams on BFCL (Section 4), the closing token in the slot and one slot longer (Section 6), and the confidence threshold on choose-N (Section 5).
- `EXP_D_PROMPT.md`: format-tolerant slots (Section 6) and position agents (Sections 4 and 5).

The prompts given to the models in the experiments are in `PROMPTS.md`.
