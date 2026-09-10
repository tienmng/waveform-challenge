# AI Usage Memo

**How I used it.** I used an AI coding assistant (Claude) as a fast pair-programmer working
under my direction. I also used Google AI (Gemini) when searching for informations about quality-control threshold, feature
choice, clinical interpretation, and method selections.

**Tools.** Claude for the signal-processing and quality-control code, the feature extraction,
the figures, the methodology audit, and the analysis notebook; standard scientific Python
(numpy, scipy, scikit-learn, pandas, matplotlib) to actually run everything.

**How I checked its work.**
- *Do the methods and reasoning make sense?* I looked at figures to check the results. I compared the method reasoning between different AI chat sessions with Claude and provided information obtained from Gemini.
- *Reproducibility:* the notebook runs start to finish with no errors.

**Where I overrode the AI**
1. **I rejected an AI rewrite of the quality checks.** Its detectors were tuned on synthetic
   signals; on the real data they dropped the usable-clip rate from 92% to 0.4%. I kept only
   its two sound pieces (a guard against missing values and a fast run-length routine) and made
   the rest report-only.
2. **For contrast, I accepted different AI suggestions** — a leakage-aware feature-ranking
   helper — but only after confirming on the real data that it correctly excluded the arterial
   signal and ranked sensible features.
3. **I directed and redirected AI to follow requirements closely** For example: to include extra figures for representations of data, or to not focus on BP prediction (which happened many times).

**Attribution.** Code was AI-generated or AI-adapted, then reviewed by
me. The full conversation is in `ai_conversation/llm_transcript.md`.
