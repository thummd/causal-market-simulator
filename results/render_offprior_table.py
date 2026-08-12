"""Render the off-prior stress table (moved out of the paper for space).

Reads ``offprior_stress.json`` (this directory) and prints the table the
paper's off-prior stress paragraph summarizes: causal-model bias and paired
do-head effect per generator, with episode-bootstrap 95% CIs (n=500 each,
gamma=1, hard interventions, exact shared-noise counterfactuals).

Args (none): run as ``python render_offprior_table.py``.
Returns: prints Markdown; writes ``offprior_stress_table.md``.
Raises: FileNotFoundError if the JSON is missing.
"""
import json

LABELS = {"control": "control (on-prior)", "feedback_weak": "feedback k=0.1",
          "feedback_strong": "feedback k=0.2", "jumps": "core jumps",
          "saturation": "saturated impact"}

d = json.load(open("offprior_stress.json"))
lines = ["| generator | causal bias [95% CI] | do-head effect [95% CI] |",
         "|---|---|---|"]
for key, label in LABELS.items():
    c = d["configs"][key]
    b, bci = c["market-dotpfn"]["bias"], c["market-dotpfn"]["bias_ci"]
    e, eci = c["delta_vs_ablated"]["mean"], c["delta_vs_ablated"]["ci"]
    lines.append(f"| {label} | {b:+.2f} [{bci[0]:+.2f}, {bci[1]:+.2f}] "
                 f"| {e:+.2f} [{eci[0]:+.2f}, {eci[1]:+.2f}] |")
out = "\n".join(lines)
print(out)
open("offprior_stress_table.md", "w").write(out + "\n")
