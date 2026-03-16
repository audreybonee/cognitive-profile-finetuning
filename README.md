# Cognitive Profile Fine-Tuning

**Can you encode a cognitive style into model weights?**

This project investigates whether LoRA fine-tuning can teach open-source LLMs (LLaMA 3.1 8B) to make decisions consistent with a specific cognitive profile, not just through prompting, but through learned behavioral patterns that transfer across domains.

The first case study is **Prospect Theory** (Kahneman & Tversky, 1979), a well-documented cognitive profile with precise, measurable deviations from rational decision-making. This makes it ideal for fine-tuning validation: we know exactly what the **right wrong answer** looks like.

## The Prospect Theory Cognitive Profile

The target profile encodes six documented biases:

| Bias | Description | Key Prediction |
|---|---|---|
| **Loss aversion** | Losses feel ~2× as painful as equivalent gains | Reject positive-EV gambles when losses are salient |
| **Certainty effect** | Overweight guaranteed outcomes | Prefer sure $900 over 90% chance of $1,000 |
| **Reflection effect** | Risk-averse in gains, risk-*seeking* in losses | Gamble to avoid certain losses |
| **Status quo bias** | Prefer the default/current state | Stick with existing option even when switching is EV-positive |
| **Endowment effect** | Overvalue what you already possess | Demand more to sell than you'd pay to buy |
| **Framing sensitivity** | Respond differently to logically equivalent framings | "Save 200 of 600" vs "400 will die" flips preference |


## References

- Kahneman, D., & Tversky, A. (1979). Prospect Theory: An Analysis of Decision under Risk. *Econometrica*, 47(2), 263-291.
- Tversky, A., & Kahneman, D. (1981). The Framing of Decisions and the Psychology of Choice. *Science*, 211(4481), 453-458.
- Kahneman, D., Knetsch, J. L., & Thaler, R. H. (1990). Experimental Tests of the Endowment Effect. *Journal of Political Economy*, 98(6), 1325-1348.
- Arkes, H. R., & Blumer, C. (1985). The Psychology of Sunk Cost. *Organizational Behavior and Human Decision Processes*, 35(1), 124-140.
- Samuelson, W., & Zeckhauser, R. (1988). Status Quo Bias in Decision Making. *Journal of Risk and Uncertainty*, 1(1), 7-59.