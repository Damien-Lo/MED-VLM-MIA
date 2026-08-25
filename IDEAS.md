# MIA Project — Idea Shortlist

Running list of extensions/experiments to the LOGAN-style MIA pipeline, beyond what's
currently implemented. Each entry is written to stand alone — assume the reader (possibly
future us) has forgotten the conversation that produced it.

---

## 1. Per-sample noise-sweep landscape mapping (added 2026-08-25)

**Current baseline setup (for context):** the pipeline builds a target set of 300 distinct
samples and sweeps them across a range of random perturbations at set noise levels — i.e.
one (or few) perturbation draw(s) per sample per noise level, aggregated *across* samples to
get a member vs. non-member signal (see `mia/src/metrics/proposed_metrics.py`,
`renyi_kl_div_maxk` / `renyi_divergence_maxk`, and the ripple/sliced variants
`renyi_kl_div_ripple_maxk` / `renyi_divergence_ripple_maxk`).

**The idea:** instead of (or in addition to) sweeping many different samples once each,
pick a single root image and run many repeated trials (150–300) of random perturbations
at increasing noise strength *on that one image*. Since every trial originates from the
same root, this traces out a per-sample loss/KLD-vs-perturbation-strength curve — i.e. maps
the local neighborhood ("landscape") around that one point, rather than getting one noisy
reading per sample. Do this for a stratified set of known members and known non-members and
compare the shape of their curves — look for systematic differences, e.g. "wells" (a dip in
divergence before it rises again) that appear more/differently for members vs non-members.

**Why this is worth doing — connection to existing evidence:** the codebase already has a
commented-out variant (`renyi_kl_div_ripple_maxk` / `renyi_divergence_ripple_maxk`, dead code
near the bottom of `proposed_metrics.py`) with a note referencing "LOGAN paper REGION II":
at low-noise/low-std settings, members were observed to have *lower* KLD than non-members —
the opposite of the normal convention (higher divergence = more likely member) — which
required a score flip. Commit `388f3a1` ("Commit before reversing convention back to normal
members have higher KLD and implimenting the flip at analysis graphing time") shows this was
worked through by flipping at analysis/graphing time rather than resolving *why* the
non-monotonic region exists. That flip was discovered from an aggregate signal across the
300-sample sweep. This per-sample landscape-mapping idea is a way to directly observe
*whether that dip is a real per-sample landscape feature* (e.g. members sitting in a
narrower/sharper basin that a small perturbation escapes quickly, producing a well before
climbing again) rather than an artifact of aggregating across heterogeneous samples. If a
visible per-sample well shows up more/differently for members, it explains REGION II and
could become a standalone metric (basin width/depth/slope) rather than just a sign flip.

**Design decisions discussed, not yet locked in:**
- **Number of root images**: a single image only gives one curve — not enough to compare
  member vs non-member. Plan: pull a stratified subset from the existing 300 (proposed:
  ~10–20 known members + ~10–20 known non-members) and run the full per-image trial budget
  on each. Note the compute cost: 10–20 trials/level design below × ~20-40 images ≈
  3,000–12,000 extra forward passes for just the pilot — estimate GPU time before committing.
- **Trial budget allocation** (given ~150–300 trials per image): choose between (a) a dense
  noise grid with few repeats per level (better curve shape resolution, noisier per-point
  estimate) vs (b) fewer noise levels with many repeats per level (better error bars per
  point, coarser curve). Leaning proposal: ~10–15 noise levels × ~15–20 repeats each ≈
  150–300 total — gives a usable curve *and* a per-level variance estimate. The per-level
  variance itself might be a separate signal worth comparing between members/non-members,
  not just the mean curve.
- **Perturbation scope**: reuse the existing augmentation/perturbation mechanism as-is
  (image-only, matching current sweep), unless we later want to test text-prompt
  perturbations too — not discussed yet, treat as a separate idea if pursued.
- **Exploratory vs. metric first**: treat this as exploratory first — run the pilot on the
  stratified subset, plot curves per image, eyeball for visible wells/shape differences
  between members and non-members. Only formalize into an extracted-feature metric (e.g.
  near-origin slope/curvature, well depth, well location, post-well slope, area under curve)
  if a visible pattern actually shows up. Don't build the feature-extraction pipeline before
  the pilot confirms there's something there.

**Status:** not started. Next step when picked up: decide the stratified image subset size
and trial-budget split above, then run the pilot and look at raw curves before deciding on
any extracted metric.

---

## 2. Localizing Regime II/III behavior to a specific submodule (added 2026-08-25)

**Paper context (for future reference):** the LOGAN paper section "Gaussian Perturbation
Analysis" (`sec:4_the_low_std_region`) sweeps Gaussian perturbation magnitude $\tilde\sigma$
and tracks the member/non-member divergence gap $\Delta\bar D(\tilde\sigma) = \bar D_m -
\bar D_n$, identifying three regimes:
- **Regime I** ($\Delta\bar D \approx 0$, near-zero noise): members and non-members respond
  almost identically — generic local sensitivity of the model, not a membership signal.
- **Regime II** ($\Delta\bar D < 0$, small nonzero noise): the paper's central finding —
  non-members diverge *more* than members, i.e. members are more locally robust. The paper
  explicitly likens this to label-only/adversarial-perturbation MIAs against (non-generative)
  image classifiers (Choquette-Choo et al. 2021; Zhang et al. 2022), where members show
  greater local robustness to input perturbation. Framed as an "image-classifier-like"
  behavior emerging from the VLLM at low noise.
- **Regime III** ($\Delta\bar D > 0$, high noise): the conventional behavior reported in
  prior LLM/VLLM MIA work (Liu et al., EncoderMI) — members diverge more than non-members.
  Framed as "traditional-LLM-like" behavior.

LOGAN (the attack proposed in the paper) calibrates over the full noise range per
model-dataset pair to pick whichever regime (and score orientation) gives the cleanest
separation, since which regime dominates is dataset/model-pair dependent (one evaluated pair
favors Regime III only).

**The idea:** test whether the Regime II ("classifier-like") vs. Regime III ("LLM-like")
split is a genuine architectural artifact — i.e. actually attributable to specific
submodules — rather than just a descriptive analogy. Concretely: rerun the same perturbation
sweep, but in addition to measuring divergence on the final LLM output token distribution,
also capture the **projector output** (the image embeddings right after the vision
encoder+projector, before they enter the LLM) and compute an original-vs-perturbed distance
there across the same noise range. Compare the shape of the projector-space signal to the
LLM-output-space signal (the existing $\Delta\bar D(\tilde\sigma)$ curve):
- If the Regime II low-noise "members more robust" signal is already present in
  projector-space embedding distance, that signal originates upstream of the LLM (vision
  encoder/projector side) — genuinely classifier-like, since it exists in continuous
  embedding space before any autoregressive decoding happens.
- If the Regime III high-noise "members diverge more" signal only shows up once you look at
  the LLM's output distribution, and isn't present (or looks qualitatively different) in
  projector space, that pins Regime III as something the LLM's forward pass introduces or
  amplifies, not something inherited from the vision side.

**Design points discussed:**
- **Metric mismatch**: projector output is a continuous embedding, not a token distribution,
  so Rényi/KL divergence (Eq. `renyi_div`/`renyi_norm` in the paper) doesn't directly apply.
  Need a distance metric in embedding space instead — e.g. cosine distance or L2, aggregated
  per-image-token similarly in spirit to the existing max-k token approach, as the
  projector-side analog of the current metric.
- **Extend beyond a binary before/after LLM split**: could also tap intermediate LLM layers
  (not just final output), to see whether the classifier-like → LLM-like transition happens
  gradually across depth or is a sharp switch at some specific layer — this would localize
  *where inside* the LLM it happens, not just confirm it's the LLM as a whole.
- **A null result is informative too**: if the projector shows no member/non-member gap at
  all, that's evidence the Regime II signal isn't purely a vision-side artifact either — it
  might instead emerge from the image-token/text cross-attention interaction specifically at
  low perturbation, which would be a more nuanced claim than "it's just the vision encoder."
- **Feasibility check needed before committing**: confirm the model wrappers in `mia/llava`,
  `mia/minigpt`, `models/hulu-med-7b-encoder` actually expose/allow hooking the projector
  output and intermediate hidden states — check this before planning the full experiment.

**Status:** not started. Next step when picked up: feasibility check on hooking
projector/intermediate-layer outputs in the existing model wrappers, then decide binary
(projector vs. final output) vs. layer-wise version of the probe.

---
