# Reflection — Part 4

**What I'd do differently with more time.**
I built and tested the quality checks and features on 100 patients. Next I'd run them on the
full 1,524-patient group and then build the predictor I deliberately stopped short of: a model
that uses several features at once, trained and scored **per patient rather than pooled**, and
**calibrated for each patient**. I'd do it this way because the results already show the
blood-pressure signal lives mostly *within* a patient over time, not *across* patients. I'd
also use longer clips (so slow heart-rate-variability rhythms become measurable), recover a
true breathing-based pressure-variation measure from the breathing channel, and validate with
patient-grouped cross-validation and, ideally, a second hospital's data rather than one split.

**Spend more time on rigorously understanding methods and codes used within these data analysis and feature engineering tasks**
