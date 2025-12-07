⭐ Tensora

A Benchmark for Tension-Coupled Optimization
“Make the space itself do the work.”


---

📌 What Tensora Actually Is

Tensora is a clean, falsifiable, open-source experiment testing one precise hypothesis:

> Does adding a dynamic tension field to standard optimizers reduce FLOPs — while reaching equal or better solutions?



No mysticism.
No hand-wavey physics.
No claims about exotic computation.

Just controlled trials, identical seeds, auditable FLOP counting, and a single brutally honest question:

Does tension help? Or not?

Either answer is a discovery.


---

🎯 The Core Hypothesis

Many classical algorithms explore search spaces “blind,” guided only by cost functions. Tensora adds a second mechanism:

The Tension Field

A coupling layer that:

Pulls coherent configurations together

Repels chaotic divergences

Smooths the search landscape

Encourages faster convergence


If this mechanism reduces the effective dimensionality of the search, then:

…the tension-augmented solver should use fewer FLOPs than the baseline.

That’s the entire point of the project.

Not performance.
Not speed.
Not hype.
FLOP reduction under controlled conditions.


---

🧪 What Tensora Measures

Every benchmark records:

Exact FLOPs (via custom instrumented counters)

Iterations to convergence

Solution quality

Tension energy over time

Random seed + full solver config


Full reproducibility.
Full transparency.
No profiler tricks, no inference, no guesswork.


---

🧩 Supported Problem Types (v0)

TSP variants (Euclidean + random)

Continuous minimization (Rastrigin, Ackley, bowl potentials)

Structured constraints (toy protein chains, spring-mesh relaxations)


Each problem includes:

Baseline solver: standard algorithm

Tension solver: same algorithm + tension layer

Identical seeds


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

Everything is minimal.
Everything is where you expect it.
No noise.


---

🚀 Quick Start

git clone https://github.com/yourname/tensora
cd tensora
pip install -r requirements.txt

python run.py --problem tsp --size 64 --tension 0.15 --seed 42

This produces a JSON file containing all metrics — FLOPs included.

That’s the heartbeat of Tensora.


---

📊 Verification Table (Blank Until Deserved)

Problem	Size	Baseline FLOPs	Tensora FLOPs	Δ FLOPs	Seeds	Status

TSP	64	–	–	–	–	pending
Rastrigin	512-d	–	–	–	–	pending


Numbers appear only after they’ve survived reproducibility tests.
Zero hype.


---

🧬 Tension Field (v0)

Hookean spring-like coupling:

F_tension(i) = Σ_j w(i,j) · k · (x_j - x_i)

Where:

w(i,j) = coupling weight

k = tension strength

x_i, x_j = state vectors


Fully documented in docs/tension_formulation.md.


---

🧘 Scientific Commitments

Tensora follows 4 uncompromising rules:

1. Exact FLOP counting — every operation accounted for.


2. Falsifiability — negative results are published immediately.


3. Reproducibility — seeds, configs, and logs stored forever.


4. No spin — Tensora reports what is true, not what is exciting.




---

❓ FAQ (No BS Edition)

Q: What if tension doesn’t help?
Then we publish that. That’s science.

Q: Is this analog/quantum/exotic computing?
No. Pure classical computation with explicit FLOP metrics.

Q: Why FLOPs instead of wall time?
Because FLOPs are hardware-independent and test the algorithm, not your CPU.


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


---

