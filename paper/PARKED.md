# Parked paper material

Text removed from `paper/talk_like_a_graph.tex` on 2026-09-16 because it described
machinery that has never been run, and the paper had to fit 5 body pages. Nothing in
the `.tex` refers to any of it any more — that was deliberate, so removal is invisible
to a reader. This file is the only record. Paste blocks back verbatim if the
experiments run.

| Block | What must run before it can return |
|---|---|
| §2.4 ladder | The 18 `(n, k̄)` rungs on `node_degree`/`none` for all four arms; no rung has been confirmed valid for any model. |
| §2.4 retrieval probe | `scripts/build_retrieval_probe.py` + a generation pass for all four arms. `runs/qwen3-1.7b.retrieval` and `runs/*.probe100` exist but cover neither `qwen3-4b` arm, so the caveat would be asymmetric. |
| §2.4 rewiring | A degree-preserving double-swap generation pass paired against un-rewired graphs. Implemented, never run as a primer test. |
| Setup "Node naming" | A `.got.` generation pass. `prompts.densfull40.got.jsonl` is built; there are no matching `runs/` files. |

Restoring any block means restoring its cross-references too: the abstract clause, the
Introduction's third contribution, and the `Section~\ref{sec:ladder}` pointers that
§5.4, the Discussion and the Limitations used to carry (all rewritten, see git history
for `paper/talk_like_a_graph.tex`). `fosdick2018configuring` in `custom.bib` is cited by
nothing now; the rewiring block is its only user.

---

## 1. Abstract clause (was lines 38–41)

Sat between "…as a content-only lower bound, a length-matched filler control as a
length-only upper bound," and "Evaluated at $n=40$ on…".

```latex
and a screening procedure --- a difficulty ladder,
a graph-free reading-limit probe, and a degree-preserving rewiring test ---
that determines, for each model, where a primer effect is even
interpretable before one is measured.
```

## 2. Introduction, third contribution (was lines 95–101)

The paragraph opened "This paper makes three contributions." and closed with the
sentence below; both were changed to two contributions.

```latex
Third, a screening design (Section~\ref{sec:ladder}) that
determines, for a given model, where a primer effect is even interpretable
in the first place: a graph-length/answer-magnitude ladder, a graph-free
reading-limit probe, and a degree-preserving rewiring procedure that
isolates a primer's content effect from prompt length by construction. We
evaluate all three within a density sweep of GraphQA tasks on two model
sizes, \texttt{qwen3-1.7b} and \texttt{qwen3-4b} (Section~\ref{sec:setup}).
```

## 3. §2.4 in full (was lines 223–278)

```latex
\subsection{Locating an interpretable test: ladder, retrieval probe, and rewiring}
\label{sec:ladder}

A content effect and a length effect are not the only quantities a primer
delta can measure. A model that cannot read the prompt at all is neither
helped nor harmed by a primer's content, and a model that reads perfectly
but is not actually taxed by the task offers a primer no room to show an
effect. Both failure modes yield a null result indistinguishable from
``the primer has no effect'' unless they are ruled out beforehand. We rule
them out with three linked tools rather than one, since they fail in
opposite directions.

\paragraph{The ladder.} We screen every model across 18 \emph{rungs},
cells of an $(n, \bar{k})$ grid spanning graph length $n \in \{20, \dots,
400\}$ nodes and mean degree (answer magnitude) $\bar{k} \in \{4, \dots,
24\}$, on the \texttt{node\_degree} task under \texttt{none} alone. These
two axes induce difficulty with opposite implications for a primer: a
\emph{length-limited} rung means the model cannot read the graph and
therefore cannot exploit a primer either, whereas a
\emph{magnitude-limited} rung means the model reads the graph without
difficulty yet still miscounts, precisely the failure a primer could
circumvent. A one-dimensional size sweep can expose only the former. A
rung is discarded prior to any generation if it is \emph{blind} (a
majority-class guesser scores above 0.25), \emph{silent} (the primer's
rendered value has a standard deviation below 0.10, rendering it
effectively constant and uninformative at any sample size), or
\emph{unreadable} (beyond the model's measured reading limit, defined
below). A rung that survives all three gates is \emph{valid}, the only
kind for which a primer test is meaningful.

\paragraph{Retrieval probe.} The reading limit against which each rung is
screened comes from a separate, graph-free needle-in-haystack probe
\citep{kamradt2023needle,liu2024lostinthemiddle}: a target statement is
embedded among $k \in \{160, \dots, 640\}$ distractor statements at
varying depth, with no graph present in the prompt, isolating raw
token-position retrieval from graph reasoning entirely.
% TODO(followup): report qwen3-1.7b/qwen3-4b (all four arms) retrieval-probe
% results here once measured. Broader-model screening results removed for
% now to keep Results focused on the study's own model pair.

\paragraph{Rewiring.} Even once a valid rung is identified, the primer
test itself must still avoid the length confound described above, since
adding primer content necessarily adds characters. We resolve this with a
degree-preserving double edge swap \citep{fosdick2018configuring}:
removing edges $(a,b)$ and $(c,d)$ and adding $(a,d)$ and $(c,b)$
preserves every node's degree exactly, leaving the degree sequence, every
gold answer, and the majority-class baseline unchanged; and because the
\texttt{incident} encoder emits one line per node, the rendered prompt
retains the \emph{same string length} before and after the swap. What
changes is triangle count and clustering: measured at $n{=}80$,
$m{=}334$, transitivity rises from 0.000 to 0.630. This yields a
within-instance paired design (the same graph, rewired, with the same
question and the same gold answer) that isolates a primer's content effect
from length by construction, rather than through the post-hoc control the
\texttt{filler} condition (Section~\ref{sec:primers}) provides when
rewiring is not used.
```

## 4. Setup reading-limit caveat, already commented out (was lines 343–348)

```latex
% TODO(followup): reading-limit caveat pulled for now -- qwen3-1.7b's
% ~1,509-token / qwen3-1.7b-think's ~2,449-token retrieval-probe limits, and
% the lost-in-the-middle finding at 6,305 tokens (0.89 edges vs. 0.195
% middle, p=0.00002), are real but asymmetric: no retrieval probe has been
% run for either qwen3-4b arm. Restore this caveat, with symmetric qwen3-4b
% coverage, once that probe is run.
```

## 5. Setup "Node naming" paragraph and its footnote (was lines 350–376)

```latex
\paragraph{Node naming.} In addition to the default integer node ids, we
substitute a fixed set of 40 Game-of-Thrones character names\footnote{By
node index: 0~Ned, 1~Catelyn, 2~Daenerys, 3~Jon, 4~Bran, 5~Sansa, 6~Arya,
7~Cersei, 8~Jaime, 9~Petyr, 10~Robert, 11~Jorah, 12~Viserys, 13~Joffrey,
14~Maester, 15~Theon, 16~Rodrik, 17~Lysa, 18~Stannis, 19~Osha, 20~Tyrion,
21~Brienne, 22~Davos, 23~Varys, 24~Tormund, 25~Podrick, 26~Missandei,
27~Margaery, 28~Loras, 29~Oberyn, 30~Ellaria, 31~Melisandre, 32~Gendry,
33~Meera, 34~Jojen, 35~Ygritte, 36~Grenn, 37~Samwell, 38~Gilly, 39~Shireen.
The first 20 names are \citet{fatemi2023talklikeagraph}'s own vendored
list (character 1 respelled ``Catelyn'' in full, from the vendored
abbreviation ``Cat'', which collides with an ordinary English word); the
remaining 20 are added by this project (\texttt{graphtalk/node\_naming.py})
to cover every node at $n{=}40$, since the vendored list extends only to
20 names -- sufficient for the original benchmark, which never requires
more.} (\texttt{graphtalk/node\_naming.py}) for the integer node ids
GraphQA otherwise uses, assigned by node index rather than by any property
of the graph. Substitution and desubstitution are implemented as a text
pass around the existing, unmodified prompt-building and scoring code: an
id is replaced only where it follows the literal word ``node''/``Node''
(never a count, a coefficient, or a rendered statistic), and a model's
response is converted back to integers before it reaches the same scorer
used for the integer-id runs. This tests whether the primer/shortcut
relationship reported here depends on node-label surface form, the same
axis \citet{fatemi2023talklikeagraph} vary with their themed encodings. As
with the main density sweep, this condition has not yet been run through
the model pipeline; results are reserved in
Section~\ref{sec:results-main}.
```

## 6. Limitations paragraphs (was lines 823–840)

The first is a fragment: the sentence below closed the opening Limitations paragraph,
which otherwise survives.

```latex
Compute and time constraints limited this sweep to a single
node-count/density design rather than a broader sweep over graph sizes;
the ladder of Section~\ref{sec:ladder} locates where a broader design
would be interpretable, but has not itself yet been run as a primer test.

The ladder, retrieval probe, and rewiring mechanism (Section~\ref{sec:ladder})
are implemented but have not yet been run to completion on
\texttt{qwen3-1.7b}/\texttt{qwen3-4b} themselves: no rung has yet been
confirmed valid for either model, so the primer-\emph{content} rewiring
test has not yet been exercised on any comparison. Both are follow-up work
rather than a reported result here.

This project's node-naming substitution (Game-of-Thrones character names
in place of integer ids) is implemented and tested, and is included above
as a reserved result (Section~\ref{sec:results-main}); whether it alters
the primer/shortcut relationship reported here, the same axis
\citet{fatemi2023talklikeagraph} vary with their themed encodings, remains
unknown at the time of writing.
```

## 7. Cross-reference sites that were rewritten, not deleted

Three passages referred to §2.4 but carry their own argument. They were rewritten to
stand alone; restoring §2.4 means restoring the pointer, not the whole passage.

- **§5.4, density subsection** — ended "This is precisely the condition the ladder of
  Section~\ref{sec:ladder} is designed to detect in advance: a cell that is saturated
  for one model and taxing for another cannot yield a model-independent primer effect,
  and an estimate pooled over the two arms averages a ceiling against a floor."
- **Discussion, length paragraph** — ended "This is the empirical justification for the
  degree-preserving rewiring design of Section~\ref{sec:ladder}, which removes the
  length term by construction rather than by subtraction."
- **Discussion, closing paragraph** — opened "Taken together, these results argue that
  the interesting quantity is not whether a structural preamble helps, but where it
  \emph{could} help, which is the screening question Section~\ref{sec:ladder}
  formalizes."
