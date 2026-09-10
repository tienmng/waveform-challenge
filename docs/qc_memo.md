# Quality-Control Memo — MIMIC-BP Pilot

**Bottom line.** The data are good quality. On a 100-patient sample (3,000 thirty-second clips),
all three signals — **ECG, PPG, and ABP** — pass their quality checks in the large majority of
clips, and about **90% of clips have all three clean at once** (with valid labels).

**What I checked.** ECG, PPG, and ABP, sampled 125 times per second, each screened on the same
footing for flat or frozen stretches, values pinned at a sensor limit, brief dropouts,
physically impossible readings, and a steady heartbeat rhythm. The recorded blood-pressure
values were also checked for internal consistency. *(The breathing channel is recorded but not
assessed in this pass.)*

**Per-signal quality.**

1. **ECG — clean in ~91%.** The weakest of the three: it hits a low limit in about **7%** of
   clips and shows no steady rhythm in about **2%**, and its scale and offset vary a lot between
   patients, so each ECG clip should be rescaled before clips are compared.
2. **ABP — clean in ~99%.** It briefly clips/saturates in about **1%** of clips (the systolic
   peak flattened at a limit); otherwise it stays within a physically valid range.
3. **PPG — the cleanest, ~99.9%.** Its only real flaw is a brief drop to zero when the sensor
   loses contact (~**0.1%** of clips), which is easy to exclude.

Reassuringly, the recorded **labels are internally consistent** (systolic above diastolic,
sensible ranges, matching the arterial trace) in **100%** of clips, and the defects **cluster in
a few patients** rather than spreading evenly.

**Recommendation.** The ~90% of clips with ECG, PPG, and ABP all clean are ready for downstream
analysis (flagged per clip). Rescale each ECG clip, and screen at the patient level, since the
defects concentrate in a few patients.
