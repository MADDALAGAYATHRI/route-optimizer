"""Build report/technical_report.pdf from results/ (run evaluate.py first)."""
import json
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import HRFlowable, Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
S = json.loads((RES / "summary.json").read_text(encoding="utf-8"))
B = pd.read_csv(RES / "baselines.csv")
D = pd.read_csv(RES / "decisions.csv")

ss = getSampleStyleSheet()
ACCENT = colors.HexColor("#1f3b63")
body = ParagraphStyle("b", parent=ss["Normal"], fontName="Helvetica", fontSize=8.8, leading=11.8,
                      spaceAfter=4, alignment=TA_JUSTIFY)
h1 = ParagraphStyle("h1", parent=body, fontName="Helvetica-Bold", fontSize=10.5, leading=13,
                    spaceBefore=8, spaceAfter=3, textColor=ACCENT, alignment=0)
title = ParagraphStyle("t", parent=body, fontName="Helvetica-Bold", fontSize=15, leading=18,
                       alignment=TA_CENTER, spaceAfter=3, textColor=ACCENT)
sub = ParagraphStyle("s", parent=body, fontSize=8.6, alignment=TA_CENTER,
                     textColor=colors.HexColor("#555555"), spaceAfter=4)
cap = ParagraphStyle("c", parent=body, fontSize=7.8, leading=9.8, textColor=colors.HexColor("#444444"),
                     spaceBefore=2, spaceAfter=6)
cell = ParagraphStyle("cell", parent=body, fontSize=7.8, leading=9.6, spaceAfter=0, alignment=0)
ex = ParagraphStyle("ex", parent=body, fontName="Helvetica", fontSize=7.8, leading=10, spaceAfter=0,
                    alignment=0)


def P(t, st=body):
    return Paragraph(t, st)


def table(rows, widths, header=True):
    t = Table([[P(str(c), cell) for c in r] for r in rows], colWidths=widths)
    style = [("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#c4ccd8")),
             ("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
             ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3)]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dfe7f2")),
                  ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f6f8fb")])]
    t.setStyle(TableStyle(style))
    return t


def example_box(lines):
    t = Table([[P(l, ex)] for l in lines], colWidths=[17.8 * cm])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f2f5f9")),
                           ("LINEBEFORE", (0, 0), (0, -1), 2, ACCENT),
                           ("LEFTPADDING", (0, 0), (-1, -1), 7), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                           ("TOPPADDING", (0, 0), (-1, -1), 1.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5),
                           ("TOPPADDING", (0, 0), (-1, 0), 5), ("BOTTOMPADDING", (0, -1), (-1, -1), 5)]))
    return KeepTogether([t])  # never split the example across pages


tr = S["E5_vs_cheapest_feasible"]
rob = S["E6_robustness"]
v = rob["verdicts"]
n_scored = S["decided_by"]["trade-off score"]
e4 = S["E4_directional_consistency"]
n4 = e4["n"]
bl = B.set_index("policy")

story = [
    P("Optimizing Shipment Routes Under Cost, Time and Risk", title),
    P("Aiqon AI Internship &ndash; Technical Report &nbsp;|&nbsp; Author: [Your Name] &nbsp;|&nbsp; "
      "Code: [GitHub link]", sub),
    HRFlowable(width="100%", thickness=0.8, color=ACCENT, spaceBefore=1, spaceAfter=4),

    P("1. Problem framing: what &ldquo;optimal&rdquo; means", h1),
    P("Route choice has no single optimum: a cheaper route is usually slower, and the fastest route may cross a "
      "congested or politically exposed chokepoint. I therefore define optimality per shipment in <b>two layers</b>. "
      "(i) <b>Hard constraints</b> that must never be traded away: meet the deadline with at least 75% probability "
      "(85% for time-critical cargo), stay within budget, and stay below a geopolitical-risk cap where the cargo "
      "requires it. (ii) <b>Preferences</b> over four criteria &ndash; cost, expected transit time, delay "
      "probability and geopolitical risk &ndash; expressed as one of four priority profiles. The recommended route "
      "is the feasible, non-dominated route with the lowest profile-weighted penalty. Separating the layers "
      "matters: a weighted score alone would let a large enough saving &lsquo;buy back&rsquo; a missed deadline "
      "or a sanctioned corridor, which no operator would accept."),

    P("2. Synthetic data and how the trade-offs were constructed", h1),
    P(f"A seeded generator (<font face='Courier'>data.py</font>) builds {S['n_shipments']} shipments with "
      f"{S['n_routes']} routes (3&ndash;5 each) on six real trade lanes (e.g. Shanghai&ndash;Rotterdam, "
      "Nhava Sheva&ndash;Hamburg, Busan&ndash;Houston). Routes are <b>assembled leg by leg</b> from a table of 35 "
      "hubs (congestion index, geopolitical index, data confidence) and four modes (sea, rail, air, truck) with "
      "per-tonne-km cost, speed, handling fee and dwell time, so cost, time and risk stay physically consistent. "
      "Each lane offers genuine tensions: a short route through a high-risk chokepoint (Suez, geo index 0.75), a "
      "safe but ~10-day-longer Cape detour, an overland rail bridge (fast-ish, high geo exposure, sparse data), "
      "sea-air hybrids, and air freight (~20&times; sea cost). Per shipment, seasonal congestion (0.8&ndash;1.3&times;) "
      "and market rates (&plusmn;20% by mode) are jittered. Cargo type (apparel, bulk, auto parts, electronics, "
      "pharma, perishables) sets weight, value per tonne and the likely profile. Deadlines are set relative to the "
      "cheapest route so they bind sometimes without forcing air freight; budgets are anchored to the cheapest "
      "route that plausibly meets the deadline, with ~20% left deliberately conflicting to test the fallback. "
      f"Finally, {S['E3_decoys']['decoys']} <b>dominated decoys</b> (same path plus an extra congested "
      "transshipment and higher cost) are planted as known-wrong answers."),

    P("3. Modelling uncertainty", h1),
    P("Each route carries delay <i>events</i> (hubs, chokepoints, long legs) with a probability and mean delay "
      "derived from congestion and geo risk. A Monte Carlo model (4,000 voyages per route) draws a <b>route-level "
      "condition factor</b> shared by all events, so a bad season hits every hub at once; independent per-event "
      "noise would average out and understate the tail. The factor&rsquo;s spread grows when a hub&rsquo;s delay "
      "statistics are thin (low confidence), so uncertain estimates produce wider P90s. Geopolitical events add "
      "rare 7&ndash;15-day disruptions (probability &prop; geo<super>2</super>). Outputs per route: expected days, "
      "P50/P90, P(delay &gt; 1 day) and P(on time) for the deadline."),

    P("4. Decision framework", h1),
    P("<b>Pipeline:</b> constraints &rarr; Pareto filter &rarr; weighted score &rarr; explanation &rarr; robustness. "
      "The Pareto filter is assumption-free: a route beaten on every criterion is discarded for <i>any</i> weights, "
      "and the explanation can say so. <b>Scoring</b> uses fixed reference units rather than min-max scaling: "
      "penalty = &Sigma; w<sub>i</sub> &middot; (x<sub>i</sub> &minus; best<sub>i</sub>) / unit<sub>i</sub>, with units of 5% of "
      "cargo value, 14 days, 20 pp delay probability and 0.30 geo risk. This was a deliberate change after "
      "diagnostics: min-max scaling capped any cost premium at 1.0, so a 20&times; air premium counted like a 2&times; "
      "one and air won for heavy, low-value cargo; it also made rankings depend on which other routes were present. "
      "Fixed units make a $5k premium significant on a $40k bulk load but negligible on a $750k pharma load, and "
      "make the exchange rate between criteria explicit and auditable:"),
    table([["Profile", "w cost", "w time", "w delay", "w geo", "Implied max premium per day saved"],
           ["cost_first", ".60", ".15", ".15", ".10", "0.09% of cargo value (&asymp; carrying cost)"],
           ["balanced", ".35", ".25", ".25", ".15", "0.26% of cargo value"],
           ["time_critical", ".15", ".45", ".30", ".10", "1.07% of cargo value (stock-out / spoilage)"],
           ["risk_averse", ".20", ".15", ".30", ".35", "0.27% of cargo value; delay + geo = 65% of weight"]],
          [2.3 * cm, 1.3 * cm, 1.3 * cm, 1.4 * cm, 1.2 * cm, 9.5 * cm]),
    Spacer(1, 4),
    P("A weighted sum was preferred over a black-box optimiser because each shipment has only 3&ndash;5 options "
      "(exhaustive evaluation is exact), and because every number in the score maps to a sentence an operator can "
      "check. <b>Constraints are tiered</b>: if no route is feasible, the system relaxes deadline and budget "
      "(minimum relative breach) but never the geo-risk cap, and flags the shipment for escalation."),

    P("5. Explaining the decision", h1),
    P("Explanations are generated from the numbers, not free text: the winner is compared with the runner-up on "
      "every <i>material</i> difference (immaterial ones, e.g. $91 on $83k of cargo, are suppressed), each "
      "eliminated route gets its reason (constraint, dominance, or main weakness), a <b>price-of-constraints</b> "
      "line shows what a hard limit costs (e.g. &ldquo;the cheapest option misses the deadline; meeting it costs "
      "$31,369 more&rdquo;), and a <b>confidence verdict</b> plus <b>break-even point</b> tell the operator how "
      "fragile the choice is. Example (SHP-037, balanced, auto parts):"),
    example_box([
        "<b>RECOMMENDED: R3 &ndash; Rail land bridge via Khorgos</b> &nbsp; $6,422 &middot; 25.0 days expected "
        "&middot; delay probability 54% &middot; geo risk 0.56",
        "<b>Why R3 over R2 (Sea via Cape):</b> it has geopolitical risk 0.56 vs 0.15 and costs $2,848 more (+80%), "
        "but arrives 26.6 days sooner on average. For a balanced shipment, the gain in transit time outweighs the "
        "geo-risk penalty.",
        "<b>Confidence: CLOSE CALL</b> &ndash; the winner holds in 56% of priority-weight and 72% of risk-estimate "
        "perturbations.",
        "<b>Break-even:</b> switches to R5 (Air via Frankfurt) if the delay-probability weight rises from 25% to 32%.",
    ]),
    Spacer(1, 6),
    KeepTogether([Image(str(RES / "fig_tradeoff.png"), width=14.4 * cm, height=7.2 * cm),
    P("<b>Figure 1.</b> Trade-off space for SHP-037. The balanced profile picks the rail bridge between cheap/slow sea "
      "and fast/expensive air; the verdict correctly flags that air is a close alternative.", cap)]),

    P("6. Evaluation", h1),
    P("Because there is no ground-truth &lsquo;best route&rsquo;, I evaluate with checks that do not simply re-use "
      "the scoring function: rule compliance, planted known-wrong answers, directional sanity across profiles, "
      "naive and rule-based baselines, robustness, and a structural property (rank reversal)."),
    table([["Check", "Result"],
           ["E1 Recommendations violating a hard constraint",
            f"<b>{S['E1_constraint_violations_in_recommendations']} / {S['n_shipments'] - S['E1_fallback_shipments']}</b>; "
            f"{S['E1_fallback_shipments']} shipments with no feasible route correctly flagged for escalation"],
           ["E2 Recommendations dominated by a feasible route",
            f"<b>{S['E2_dominated_recommendations']} / {S['n_shipments']}</b>"],
           ["E3 Planted dominated decoys recommended",
            f"<b>{S['E3_decoys']['decoy_chosen']} / {S['E3_decoys']['decoys']}</b> (all detected as dominated or infeasible)"],
           ["E4 Directional consistency across profiles",
            f"cost-first never dearer than time-critical {round(e4['cost_first_not_dearer'] * n4)}/{n4}; "
            f"time-critical never slower {round(e4['time_critical_not_slower'] * n4)}/{n4}; risk-averse never "
            f"riskier {round(e4['risk_averse_not_riskier'] * n4)}/{n4}. Both exceptions are immaterial gaps "
            "(e.g. cost-first pays $996 = 0.13 cost units to avoid geo risk 0.71)"],
           ["E5 vs &ldquo;cheapest feasible&rdquo; rule",
            f"differs on {tr['n_differ']} shipments; median <b>+${tr['median_extra_cost_usd']:,.0f} "
            f"({tr['median_extra_cost_pct_of_value']:.2f}% of cargo value)</b> buys {tr['mean_days_saved']:.1f} days, "
            f"{tr['mean_delay_pp_saved']:.1f} pp lower delay probability and {tr['mean_geo_reduction']:.2f} lower geo risk"],
           ["E6 Robustness of the " + str(n_scored) + " score-decided cases",
            f"{v.get('ROBUST', 0)} robust, {v.get('MODERATE', 0)} moderate, {v.get('CLOSE CALL', 0)} close calls "
            f"flagged; mean stability {rob['mean_weight_stability']:.2f} under weight noise vs "
            f"{rob['mean_estimate_stability']:.2f} under risk-estimate error; "
            f"{rob['mc_seed_agreement']:.0%} agreement across Monte Carlo seeds"],
           ["E7 Context sensitivity",
            f"{S['E7_context_sensitivity']['share_with_profile_dependent_winner']:.0%} of shipments have a different "
            "best route under a different profile"],
           ["E8 Rank reversal when a losing route is removed",
            f"<b>{S['E8_rank_reversal']['ours']}</b> (min-max scoring: {S['E8_rank_reversal']['minmax']})"]],
          [5.3 * cm, 11.7 * cm]),
    Spacer(1, 4),
    table([["Policy", "Mean cost", "Mean days", "Mean P90", "Delay prob", "Geo risk", "Violates constraints"]] +
          [[p, f"${r.mean_cost_usd:,.0f}", f"{r.mean_days:.1f}", f"{r.mean_p90_days:.1f}",
            f"{r.mean_delay_prob:.0%}", f"{r.mean_geo_risk:.2f}", f"{r.constraint_violation_rate:.1%}"]
           for p, r in bl.iterrows()],
          [4.4 * cm, 2.0 * cm, 1.9 * cm, 1.9 * cm, 2.0 * cm, 1.8 * cm, 3.0 * cm]),
    P("<b>Table.</b> Policy comparison over all 40 shipments. Naive single-metric rules break hard constraints on "
      "40&ndash;42.5% of shipments. The framework sits between the feasible extremes, paying slightly more than the "
      "cheapest-feasible rule for materially better reliability and risk. The 7.5% rate for the framework and the "
      "feasible-rule baselines is entirely the three shipments where no route can satisfy every constraint; the "
      "framework is the only policy that flags them.", cap),
    KeepTogether([Image(str(RES / "fig_stability.png"), width=15.6 * cm, height=15.6 / 2.77 * cm),
    P("<b>Figure 2.</b> Share of perturbations that keep the same winner. Errors in risk estimates move decisions "
      "more than plausible changes in priorities: better delay data is worth more than finer-tuned weights.", cap)]),
    P(f"<b>Observations.</b> {S['decided_by']['constraints/dominance']} of 40 decisions are settled by constraints "
      "or dominance before any weighting, which is realistic and also why the price-of-constraints line matters: "
      "an over-tight deadline is often the most expensive input. Close calls are surfaced instead of hidden, "
      "turning sensitivity analysis into an operator-facing feature."),

    P("7. Assumptions, limitations and next steps", h1),
    P("<b>Assumptions:</b> rates, congestion and geo indices are synthetic and plausible, not calibrated; delay events "
      "within a route share one condition factor but routes are independent of each other (in reality a Red Sea "
      "closure also congests the Cape route); risk indices are ordinal scores, not probabilities; profile weights "
      "and reference units are policy inputs; one shipment is routed at a time, ignoring capacity, consolidation "
      "and carrier contracts. <b>Limitations:</b> the weighted sum cannot express non-compensatory preferences "
      "beyond the hard constraints; the estimate-error model is a single multiplicative bias per route. "
      "<b>Next steps:</b> calibrate the delay model on historical schedule-reliability data (highest value per "
      "Figure 2); elicit weights from planners through pairwise choices; model correlated disruptions across "
      "routes; add in-transit inventory cost explicitly; and log operator overrides to learn where the profiles "
      "are miscalibrated."),
]


def footer(c, doc):
    c.saveState()
    grey = colors.HexColor("#777777")
    c.setStrokeColor(colors.HexColor("#c4ccd8"))
    c.setLineWidth(0.4)
    c.setFont("Helvetica", 7)
    c.setFillColor(grey)
    c.line(1.6 * cm, 1.25 * cm, A4[0] - 1.6 * cm, 1.25 * cm)
    c.drawString(1.6 * cm, 0.9 * cm, "Aiqon AI Internship \u2013 Shipment route optimisation")
    c.drawRightString(A4[0] - 1.6 * cm, 0.9 * cm, f"Page {doc.page}")
    if doc.page > 1:
        c.line(1.6 * cm, A4[1] - 1.0 * cm, A4[0] - 1.6 * cm, A4[1] - 1.0 * cm)
        c.drawString(1.6 * cm, A4[1] - 0.85 * cm, "Optimizing Shipment Routes Under Cost, Time and Risk")
    c.restoreState()


out = ROOT / "report" / "technical_report.pdf"
doc = SimpleDocTemplate(str(out), pagesize=A4, leftMargin=1.6 * cm, rightMargin=1.6 * cm,
                        topMargin=1.35 * cm, bottomMargin=1.6 * cm,
                        title="Optimizing Shipment Routes Under Cost, Time and Risk", author="[Your Name]")
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print(f"Wrote {out}")
