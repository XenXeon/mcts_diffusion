# Dissertation build conventions

Submission format: **Word**, using the EEEM004 template. Chapters are drafted here as
Markdown structured to the template's heading levels, then pasted in and styled.

## Word workflow

| Markdown | Word paragraph style |
|---|---|
| `# Chapter Title` | Heading 1 |
| `## Section` | Heading 2 |
| `### Subsection` | Heading 3 |
| `#### Sub-subsection` | Heading 4 |
| body text | Normal |
| `> quoted block` | Quote |

**Heading numbers.** Section numbers ARE included in these drafts (e.g.
`## 3.2 The tree algorithm`) so the drafts are readable during review. The template's
Heading styles are self-numbering, so strip them on paste:

```
find (regex):  ^([0-9]+(\.[0-9]+)*)\s+
replace:       (nothing)
```
applied to the pasted headings only. If the template's auto-numbering is off, keep them.

**Cross-references.** Written as plain text (`Section 3.2`, `Figure 4-1`, `Table 3-1`).
Each must be converted to a Word cross-reference field (Insert → Cross-reference) so it
updates automatically — the template guidance requires this and it is easy to lose marks
on stale numbers. A checklist of every cross-reference is maintained in the final-pass
task.

**Figures and tables.** Caption style per the template: `Figure 3-1 - <title and
essential information>`, placed *below* the figure; `Table 3-1 - <title>` placed *above*
the table. Every figure and table must be referred to in the body text before it appears.
Insert captions with References → Insert Caption so the List of Figures builds itself.

**Equations.** Centre the equation in a two-column borderless table with the number
right-aligned, per the template's worked example. Numbered by chapter: (3.1), (3.2), …

## Source of truth for numbers

Every experimental number traces to exactly one place. Do not re-derive figures from the
older working notes — several contain superseded values.

| domain | authoritative source |
|---|---|
| maze2d DV-backbone rows | `notes/maze2d_startmatched_correction.md` |
| all other arms | `notes/results_chapter.md` + the per-rollout `results/*.json` |
| protocol / compute | the T1 table (moving into Chapter 3) |
| figures | `scripts/make_figures.py` — F1/F2/F3 derive from `results/*.json`, so they cannot drift from the text |

**Rule inherited from the start-matching correction:** never quote a difference between
two arms without first asserting their `starts` arrays are equal, and never difference an
arm against a baseline computed over a different seed set.

## Register

The working notes are written in a punchy research-log voice. The dissertation is not.
On the final pass, remove: "the poison", "money shot", "honesty row", "caveats of
record", "disarming", and similar. Keep the directness; drop the swagger. Voice is
first-person plural past tense ("we measured"), consistent throughout.

## Status

Chapter files appear here as they are drafted. `CONTENTS.md` is the planned contents
page — this is the document to agree with the supervisor before full write-up, per the
module guidance.
