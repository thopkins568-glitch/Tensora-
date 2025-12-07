⭐ Tensora

A Benchmark for Tension-Coupled Optimization

“Make the space itself do the work.”


---

📌 What Tensora Actually Is

Tensora is a clean, falsifiable, open-source experiment testing one precise hypothesis:

> Does adding a dynamic tension field to standard optimizers reduce FLOPs — while reaching equal or better solutions?



No mysticism.
No hand-wavey physics.
No exotic claims.

Just controlled trials, identical seeds, auditable FLOP counting — and one brutally honest question:

Does tension help? Or not?
Either answer is a discovery.


---

🎯 The Core Hypothesis

Most classical algorithms explore search spaces “blind,” guided only by a cost function. Tensora adds a second mechanism:

The Tension Field

A coupling layer that:

Pulls coherent configurations together

Repels chaotic divergences

Smooths the search landscape

Encourages faster convergence


If this reduces the effective dimensionality of the search, then:

> The tension-augmented solver should use fewer FLOPs than the baseline.



That’s the entire point of the project.
Not speed.
Not wall-time.
Not hype.
FLOP reduction under controlled conditions.


---

🧪 What Tensora Measures

Every benchmark records:

Exact FLOPs (custom instrumented counters)

Iterations to convergence

Solution quality

Tension energy over time

Random seed + full solver config


Full reproducibility.
Full transparency.
No tricks. No profiler inference. No guesswork.


---

🧩 Supported Problem Types (v0)

TSP variants (Euclidean + random)

Continuous minimization (Rastrigin, Ackley, bowl potentials)

Structured constraints (toy protein chains, spring-mesh relaxations)


Each problem includes:

Baseline solver: standard algorithm

Tension solver: identical algorithm + tension layer

Same seeds, same configs


If it’s not fair, it’s not Tensora.


---

📁 Project Structure

tensora/
├── tensora/
│   ├── core/          # FLOP counter, convergence utilities
│   ├── fields/        # Tension formulations
│   ├── solvers/       # Baseline + tension-augmented versions
│   ├── problems/      # TSP, continuous, structured
│   └── utils/
├── experiments/       # Scripts for published runs
├── results/           # Raw JSON logs (immutable)
├── docs/
├── tests/
├── run.py
└── README.md

Minimal. Predictable. Clean.


---

🚀 Quick Start

git clone https://github.com/yourname/tensora
cd tensora
pip install -r requirements.txt

python run.py --problem tsp --size 64 --tension 0.15 --seed 42

Outputs a fully reproducible JSON log — FLOPs included.
That’s Tensora’s heartbeat.


---

📊 Verification Table (Blank Until Deserved)

Problem	Size	Baseline FLOPs	Tensora FLOPs	Δ FLOPs	Seeds	Status

TSP	64	–	–	–	–	pending
Rastrigin	512-d	–	–	–	–	pending


Numbers appear only after they survive reproducibility tests.
Zero hype.


---

🧬 Tension Field (v0)

Hookean spring-like coupling:

Fₜ(i) = Σⱼ w(i,j) · k · (xⱼ − xᵢ)

Where:

w(i,j) = coupling weight

k = tension strength

xᵢ, xⱼ = state vectors


Fully documented in docs/tension_formulation.md.


---

🧘 Scientific Commitments

Tensora follows four uncompromising rules:

1. Exact FLOP counting — every operation accounted for.


2. Falsifiability — negative results get published immediately.


3. Reproducibility — seeds, configs, and logs stored forever.


4. No spin — Tensora reports what is true, not what is exciting.




---

❓ FAQ (No BS Edition)

Q: What if tension doesn’t help?
Then we publish that. That’s science.

Q: Is this analog/quantum/exotic computing?
No. Pure classical computation with explicit FLOP metrics.

Q: Why FLOPs instead of wall-time?
Because FLOPs test the algorithm, not your hardware.


---

📝 License

MIT — fork it, break it, improve it.


---

🔥 Final Word

Tensora is an experiment.
A clean one.
A dangerous one.
A necessary one.

Whether tension reduces FLOPs or not, the answer will be real — because the method is real.

Welcome to Tensora.
Let the numbers speak.
