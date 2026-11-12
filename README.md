# Shipment Route Recommender — cost, time and risk under uncertainty

Aiqon AI Internship take-home. Given a shipment and 3–5 candidate multi-leg routes, the system
recommends a route, explains the choice in operator language, and says how confident it is.

```
shipment ─► Monte Carlo metrics ─► hard constraints ─► Pareto filter ─► profile-weighted score ─► explanation
            (time, delay, on-time)  (deadline, budget,   (drop routes     (fixed reference units,     + robustness
                                     geo-risk cap)        beaten on all)   no rank reversal)           + break-even
```

## Setup

Requires **Python 3.9 or newer** (tested on 3.12 with both the newest libraries and the oldest supported
ones: numpy 1.26, pandas 2.2, matplotlib 3.8 — results are identical).

```bash
git clone <this-repo-url>
cd route-optimizer

python -m venv .venv
# Windows (PowerShell):  .venv\Scripts\Activate.ps1
# Windows (cmd):         .venv\Scripts\activate.bat
# macOS / Linux:         source .venv/bin/activate

pip install -r requirements.txt
```

The repository already contains the generated dataset (`data/`) and evaluation outputs (`results/`),
so every command below works immediately after installing.

## Usage

```bash
python -m pytest -q                      # 16 tests, < 1 s
python recommend.py --list               # list the 40 shipments
python recommend.py SHP-037              # recommendation + explanation for one shipment
python recommend.py SHP-037 --profile time_critical   # same shipment, different priorities
python recommend.py --file my_shipment.json           # any shipment in the same JSON schema
python evaluate.py                       # full evaluation -> results/   (~5 s)
python generate_data.py                  # regenerate data/ (seed 7, reproduces the committed files exactly)
python report/build_report.py            # rebuild report/technical_report.pdf from results/
streamlit run app.py                     # interactive demo in the browser
```

All scripts locate their files relative to themselves, so they can be run from any working directory.

### Running in VS Code

1. **File → Open Folder…** and choose `route-optimizer`.
2. Install the **Python** extension if prompted, then **Ctrl/Cmd+Shift+P → Python: Select Interpreter**
   and pick the `.venv` interpreter.
3. Open a terminal (**Terminal → New Terminal**) and run the commands above.
4. Tests also appear in the **Testing** panel (pytest is pre-configured in `.vscode/settings.json`
   and `pyproject.toml`).

### Troubleshooting

- `ModuleNotFoundError: numpy` (or pandas, …): the virtual environment is not active, or VS Code is using
  a different interpreter — re-run the activate command / re-select the interpreter.
- PowerShell refuses to run `Activate.ps1`: run
  `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or use `cmd` with `activate.bat`.
- `streamlit` or `reportlab` are only needed for the demo and for rebuilding the PDF; everything else
  runs without them.

Example output (`python recommend.py SHP-037`):

```
  ID  Route                                         Cost $  E[days]    P90  Delay%   Geo  Status
------------------------------------------------------------------------------------------------------------------
  R1  Sea via Singapore & Suez                       2,892     43.1   51.7     62%  0.73  penalty 151
  R2  Sea via Singapore & Cape                       3,574     51.5   58.9     54%  0.15  penalty 130
* R3  Rail land bridge via Khorgos                   6,422     25.0   30.1     54%  0.56  RECOMMENDED
  R4  Sea-air via Jebel Ali / Dubai                 24,085     25.5   30.7     45%  0.37  penalty 161
  R5  Air via Frankfurt                             37,230      4.3    5.6     20%  0.16  penalty 132

Shipment SHP-037: auto parts, 12.2 t, Shanghai -> Rotterdam
Priority: balanced | No hard constraints

RECOMMENDED: R3 - Rail land bridge via Khorgos
  $6,422 | 25.0 days expected (P90 30.1) | delay prob 54% | geo risk 0.56

Why R3 over R2 (Sea via Singapore & Cape): it has geopolitical risk 0.56 vs 0.15, costs $2,848 more (+80%), but arrives 26.6 days sooner on average (25.0 vs 51.5). For a balanced shipment, the gain in expected transit time outweighs the geopolitical risk penalty (trade-off penalty 114 vs 130 points; 0 = best on every criterion).

Other options:
  R1 Sea via Singapore & Suez: penalty 151 - main weakness vs R3: it takes 18.1 days longer on average (43.1 vs 25.0).
  R4 Sea-air via Jebel Ali / Dubai: penalty 161 - main weakness vs R3: it costs $17,663 more (+275%).
  R5 Air via Frankfurt: penalty 132 - main weakness vs R3: it costs $30,808 more (+480%).

Confidence: CLOSE CALL - winner holds in 56% of priority-weight perturbations and 72% of risk-estimate perturbations.
  Break-even: switches to R5 (Air via Frankfurt) if the delay probability weight rises from 25% to 32%.
  Recommend operator review: the top options are near-equivalent under plausible priorities.
```

## Repository layout

| Path | Purpose |
|---|---|
| `src/routeopt/data.py` | Synthetic generator: hub table, 6 trade lanes, 27 route templates, shipments, planted decoys |
| `src/routeopt/simulate.py` | Monte Carlo transit model (correlated delays, estimate uncertainty, disruptions) |
| `src/routeopt/scoring.py` | Constraints → Pareto filter → weighted scoring; priority profiles |
| `src/routeopt/explain.py` | Template-based operator explanations |
| `src/routeopt/sensitivity.py` | Weight / estimate robustness, profile sweep, break-even points |
| `generate_data.py`, `recommend.py`, `evaluate.py`, `app.py` | Entry points |
| `tests/test_routeopt.py` | Unit tests (crafted cases) + dataset-wide property tests |
| `results/` | Evaluation outputs: `summary.json`, `decisions.csv`, `baselines.csv`, `explanations.txt`, figures |
| `report/` | Technical report (PDF) and its build script |
| `pyproject.toml`, `.vscode/` | pytest configuration and VS Code test discovery |

## What "optimal" means here

There is no single optimum, so it is defined per shipment by **two layers**:

1. **Hard constraints** (never traded off): deadline met with ≥75% probability (≥85% for time-critical),
   budget cap, maximum geopolitical risk. Constraints are *tiered*: if nothing is feasible, the system
   relaxes deadline/budget (minimum relative breach) but **never** the risk cap, and flags the shipment
   for escalation.
2. **Trade-off preferences**: one of four priority profiles weighting cost, expected transit time,
   delay probability and geopolitical risk.

| Profile | cost | time | delay | geo | Implied max premium per day saved |
|---|---|---|---|---|---|
| cost_first | .60 | .15 | .15 | .10 | 0.09% of cargo value |
| balanced | .35 | .25 | .25 | .15 | 0.26% of cargo value |
| time_critical | .15 | .45 | .30 | .10 | 1.07% of cargo value |
| risk_averse | .20 | .15 | .30 | .35 | 0.27% of cargo value |

**Scoring.** Penalty = Σ wᵢ · (xᵢ − bestᵢ) / unitᵢ, using fixed reference units: cost 5% of cargo value,
14 days, 20 pp delay probability, 0.30 geo risk. Unlike min-max normalisation, a $5k premium matters on a
$40k bulk load but not on a $750k pharma load, the exchange rates between criteria are explicit
(table above), and adding or removing a route can never flip the order of two others.

## Evaluation results (40 shipments, 162 routes)

| Check | Result |
|---|---|
| E1 Recommendations violating a hard constraint | **0 / 37** (3 shipments correctly flagged: no feasible route) |
| E2 Recommendations dominated by another feasible route | **0 / 40** |
| E3 Planted dominated decoys recommended | **0 / 8** (all 8 detected) |
| E4 Directional consistency across profiles | cost-first never dearer than time-critical in 36/37; time-critical never slower 37/37; risk-averse never riskier 36/37 (both exceptions are immaterial differences — see report) |
| E5 vs "cheapest feasible" rule | differs on 14 shipments; median **+$903 (0.41% of cargo value)** buys **5.1 days**, **10.3 pp** lower delay probability, **0.30** lower geo risk |
| E6 Robustness (24 trade-off decisions) | 14 robust, 3 moderate, 7 close calls flagged; mean stability 0.91 (weights) vs 0.79 (risk estimates); 100% agreement across Monte Carlo seeds |
| E7 Context sensitivity | 25% of shipments have a different correct answer under a different profile |
| E8 Rank reversal when a losing route is removed | **0 / 40** (min-max scoring: 5 / 40) |

Baselines (`results/baselines.csv`): naive "cheapest" and "fastest" rules violate a hard constraint on
40% and 42.5% of shipments respectively.

## Key assumptions

Synthetic data (rates, congestion and risk indices are plausible, not calibrated); delay events share a
route-level condition factor but routes are evaluated independently; one shipment at a time (no capacity
or consolidation); profile weights are policy inputs, not learned. See the report for the full list.
