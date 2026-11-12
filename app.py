"""Interactive shipment route recommender presentation."""
import html
import re
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))
from routeopt import CRITERIA, PROFILES, compute_metrics, decide, explain, load_dataset
from routeopt.explain import robustness_verdict
from routeopt.scoring import CRITERION_LABEL
from routeopt.sensitivity import estimate_stability, flip_points, profile_sweep, weight_stability

st.set_page_config(page_title="Route recommender", layout="wide")
DATA = Path(__file__).parent / "data" / "shipments.json"

PALETTE = {
    "white": "#FFFFFF",
    "surface": "#FAFAFA",
    "border": "#E6E6E6",
    "grid": "#F0F0F0",
    "ink": "#1A1A1A",
    "muted": "#6B6B6B",
    "dot": "#9E9E9E",
    "outline": "#BDBDBD",
    "alert": "#B42318",
}
PROFILE_LABELS = {
    "balanced": "Balanced",
    "cost_first": "Cost first",
    "time_critical": "Time critical",
    "risk_averse": "Risk averse",
}
PROFILE_ORDER = ["balanced", "cost_first", "time_critical", "risk_averse"]
STATUS_ORDER = ["Recommended", "Least bad", "Alternative", "Dominated", "Infeasible"]
STATUS_COLORS = {
    "Recommended": (PALETTE["ink"], PALETTE["ink"], PALETTE["ink"]),
    "Least bad": (PALETTE["white"], PALETTE["ink"], PALETTE["ink"]),
    "Alternative": (PALETTE["dot"], PALETTE["dot"], PALETTE["muted"]),
    "Dominated": (PALETTE["white"], PALETTE["outline"], PALETTE["muted"]),
    "Infeasible": (PALETTE["alert"], PALETTE["alert"], PALETTE["alert"]),
}
CHOKEPOINTS = {"Suez Canal", "Panama Canal", "Cape of Good Hope"}

st.markdown(
    f"""<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');
    html, body, [class*="st-"] {{ font-family: 'IBM Plex Sans', system-ui, 'Segoe UI', sans-serif; }}
    .stApp {{ background: {PALETTE['white']}; color: {PALETTE['ink']}; }}
    [data-testid="stSidebar"] {{ background: {PALETTE['surface']}; border-right: 1px solid {PALETTE['border']}; }}
    [data-testid="stSidebar"] > div:first-child {{ background: {PALETTE['surface']}; }}
    .block-container {{ max-width: 1320px; padding-top: 3.6rem; padding-bottom: 3rem; }}
    h1, h2, h3, p, label {{ color: {PALETTE['ink']}; }}
    .page-title {{ font-size: 1.55rem; font-weight: 700; line-height: 1.25; color: {PALETTE['ink']}; }}
    .page-subtitle {{ margin-top: .35rem; margin-bottom: 1.45rem; color: {PALETTE['muted']}; font-size: .94rem; }}
    .side-heading {{ margin: .25rem 0 .45rem; font-size: 1rem; font-weight: 600; color: {PALETTE['ink']}; }}
    .side-section {{ margin-top: 1.1rem; }}
    .constraint-copy {{ color: {PALETTE['ink']}; font-size: .88rem; line-height: 1.8; }}
    .constraint-copy strong {{ font-weight: 600; }}
    .none-copy {{ color: {PALETTE['muted']}; font-size: .88rem; }}
    .recommendation {{ box-sizing: border-box; margin: .2rem 0 1.45rem; padding: 1.1rem 1.4rem;
        border: 1px solid {PALETTE['border']}; border-left: 3px solid {PALETTE['ink']}; border-radius: 8px;
        background: {PALETTE['white']}; }}
    .recommendation.fallback {{ border-left-color: {PALETTE['alert']}; }}
    .rec-top {{ display: flex; align-items: center; justify-content: space-between; gap: 1rem; }}
    .rec-eyebrow {{ color: {PALETTE['muted']}; font-size: .82rem; }}
    .verdict-pill {{ padding: .3rem .65rem; border: 1px solid {PALETTE['ink']}; border-radius: 999px;
        color: {PALETTE['ink']}; font-size: .76rem; white-space: nowrap; }}
    .verdict-pill.fallback-pill {{ border-color: {PALETTE['alert']}; color: {PALETTE['alert']}; }}
    .route-title {{ margin: .55rem 0 .9rem; color: {PALETTE['ink']}; font-size: 1.6rem; font-weight: 700; line-height: 1.25; }}
    .route-strip {{ display: flex; flex-wrap: wrap; align-items: center; gap: .4rem; margin-bottom: 1rem; }}
    .hub-chip {{ padding: .3rem .55rem; border: 1px solid {PALETTE['border']}; border-radius: 6px;
        color: {PALETTE['ink']}; background: {PALETTE['white']}; font-size: .88rem; }}
    .hub-chip.chokepoint {{ border-style: dashed; border-color: {PALETTE['outline']}; }}
    .mode-label {{ color: {PALETTE['muted']}; font-size: .72rem; white-space: nowrap; }}
    .facts {{ display: flex; flex-wrap: wrap; gap: .65rem 1.35rem; padding-top: .8rem; border-top: 1px solid {PALETTE['border']}; }}
    .fact {{ color: {PALETTE['muted']}; font-size: .84rem; }}
    .fact strong {{ color: {PALETTE['ink']}; font-weight: 600; }}
    .robustness-copy {{ margin-top: .75rem; color: {PALETTE['muted']}; font-size: .8rem; }}
    .section-heading {{ margin: .2rem 0 .55rem; color: {PALETTE['ink']}; font-size: 1.05rem; font-weight: 600; }}
    .chart-legend {{ display: flex; flex-wrap: wrap; align-items: center; gap: .45rem .9rem; margin: .1rem 0 .25rem;
        color: {PALETTE['muted']}; font-size: .76rem; }}
    .legend-item {{ display: inline-flex; align-items: center; gap: .35rem; }}
    .legend-dot {{ display: inline-block; width: 9px; height: 9px; box-sizing: border-box; border-radius: 50%; }}
    .legend-note {{ color: {PALETTE['muted']}; }}
    .explanation-box {{ padding: .95rem 1rem; border: 1px solid {PALETTE['border']}; border-radius: 8px;
        min-height: 400px; background: {PALETTE['white']}; color: {PALETTE['ink']}; font-size: .88rem; line-height: 1.5; }}
    .chart-legend {{ flex-wrap: nowrap; white-space: nowrap; }}
    .profile-grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: .6rem; margin-bottom: 2rem; }}
    .profile-card {{ min-width: 0; padding: .6rem .8rem; border: 1px solid {PALETTE['border']}; border-radius: 8px;
        background: {PALETTE['white']}; }}
    .profile-card.active {{ border-color: {PALETTE['ink']}; }}
    .profile-name {{ color: {PALETTE['muted']}; font-size: .8rem; }}
    .profile-route {{ margin-top: .25rem; color: {PALETTE['ink']}; font-size: .9rem; font-weight: 500; overflow-wrap: anywhere; }}
    @media (max-width: 800px) {{ .profile-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} }}
    [data-testid="stDataFrame"] {{ border: 0; }}
    </style>""",
    unsafe_allow_html=True,
)


@st.cache_data
def load():
    return load_dataset(DATA)


@st.cache_data
def metrics_for(sid):
    shipment = next(item for item in load() if item["shipment_id"] == sid)
    return compute_metrics(shipment)


def shipment_label(shipment_id):
    shipment = next(item for item in shipments if item["shipment_id"] == shipment_id)
    cargo = shipment["cargo_type"].replace("_", " ")
    return f"{shipment_id} ({cargo}, {shipment['origin']}-{shipment['destination']})"


def sync_shipment_query():
    selected_id = st.session_state["shipment_picker"]
    st.query_params["shipment"] = selected_id
    selected_shipment = next(item for item in shipments if item["shipment_id"] == selected_id)
    st.session_state["priority_profile"] = selected_shipment["priority_profile"]
    for criterion, value in PROFILES[selected_shipment["priority_profile"]].items():
        st.session_state[f"weight_{criterion}"] = float(value)


def reset_profile_weights():
    selected_profile = st.session_state["priority_profile"]
    for criterion, value in PROFILES[selected_profile].items():
        st.session_state[f"weight_{criterion}"] = float(value)


def status_for(route_id, route, decision):
    if route_id == decision["winner"]:
        return "Least bad" if decision["fallback"] else "Recommended"
    if route["violations"]:
        return "Infeasible"
    if route["dominated_by"]:
        return "Dominated"
    return "Alternative"


def route_strip(route):
    parts = []
    for index, hub in enumerate(route["hubs"]):
        chip_class = "hub-chip chokepoint" if hub in CHOKEPOINTS else "hub-chip"
        parts.append(f'<span class="{chip_class}">{html.escape(hub)}</span>')
        if index < len(route["legs"]):
            mode = html.escape(route["legs"][index]["mode"])
            parts.append(f'<span class="mode-label">— {mode} —</span>')
    return "".join(parts)


def style_status(value):
    color = STATUS_COLORS[value][2]
    weight = "700" if value in {"Recommended", "Least bad"} else "400"
    return f"color: {color}; font-weight: {weight};"


def style_winner_route(value, winner_name):
    return f"font-weight: 700; color: {PALETTE['ink']};" if value == winner_name else ""


shipments = load()
shipment_ids = [item["shipment_id"] for item in shipments]
requested_id = st.query_params.get("shipment")
if requested_id in shipment_ids and st.session_state.get("shipment_picker") != requested_id:
    st.session_state["shipment_picker"] = requested_id
    selected_from_url = next(item for item in shipments if item["shipment_id"] == requested_id)
    st.session_state["priority_profile"] = selected_from_url["priority_profile"]
    for criterion, value in PROFILES[selected_from_url["priority_profile"]].items():
        st.session_state[f"weight_{criterion}"] = float(value)
if "priority_profile" not in st.session_state:
    initial_id = requested_id if requested_id in shipment_ids else shipment_ids[0]
    initial_shipment = next(item for item in shipments if item["shipment_id"] == initial_id)
    st.session_state["priority_profile"] = initial_shipment["priority_profile"]
    for criterion, value in PROFILES[initial_shipment["priority_profile"]].items():
        st.session_state[f"weight_{criterion}"] = float(value)

with st.sidebar:
    st.markdown('<div class="side-heading">Shipment</div>', unsafe_allow_html=True)
    sid = st.selectbox(
        "Shipment",
        shipment_ids,
        index=0,
        format_func=shipment_label,
        label_visibility="collapsed",
        key="shipment_picker",
        on_change=sync_shipment_query,
    )
    s = next(item for item in shipments if item["shipment_id"] == sid)
    if st.query_params.get("shipment") != sid:
        st.query_params["shipment"] = sid

    st.markdown('<div class="side-heading side-section">Priorities</div>', unsafe_allow_html=True)
    profile = st.selectbox(
        "Priority profile",
        PROFILE_ORDER,
        format_func=lambda value: PROFILE_LABELS[value],
        key="priority_profile",
        on_change=reset_profile_weights,
    )
    custom = st.toggle("Set custom weights", value=False, key="custom_weights")
    weights = dict(PROFILES[profile])
    if custom:
        raw = {
            criterion: st.slider(
                CRITERION_LABEL[criterion].capitalize(),
                min_value=0.0,
                max_value=1.0,
                step=0.05,
                key=f"weight_{criterion}",
            )
            for criterion in CRITERIA
        }
        total = sum(raw.values())
        weights = ({criterion: value / total for criterion, value in raw.items()}
                   if total else {criterion: 1 / len(CRITERIA) for criterion in CRITERIA})
    run_sens = st.toggle("Check robustness", value=True, key="check_robustness")

    st.markdown('<div class="side-heading side-section">Hard constraints</div>', unsafe_allow_html=True)
    constraints = s["constraints"]
    constraint_lines = []
    if constraints["deadline_days"] is not None:
        constraint_lines.append(
            f"Arrive within <strong>{constraints['deadline_days']:g} days</strong> with "
            f"<strong>&gt;={constraints['min_on_time_prob']:.0%} certainty</strong>"
        )
    if constraints["budget_usd"] is not None:
        constraint_lines.append(f"Budget <strong>${constraints['budget_usd']:,.0f}</strong>")
    if constraints["max_geopolitical_risk"] is not None:
        constraint_lines.append(
            f"Geopolitical risk <strong>&lt;= {constraints['max_geopolitical_risk']:.2f}</strong>"
        )
    if constraint_lines:
        st.markdown(
            '<div class="constraint-copy">' + "<br>".join(constraint_lines) + "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<div class="none-copy">None for this shipment</div>', unsafe_allow_html=True)

m = metrics_for(sid)
d = decide(s, profile=profile, weights=weights, metrics=m)
sens = None
if run_sens and not d["fallback"] and len(d["candidates"]) > 1:
    sens = dict(
        weights=weight_stability(s, m, d["weights"], d["winner"]),
        estimates=estimate_stability(s, d["weights"], d["winner"]),
        flips=flip_points(s, m, d["weights"], d["winner"]),
    )

winner_id = d["winner"]
winner = d["routes"][winner_id]
winner_source = next(route for route in s["routes"] if route["route_id"] == winner_id)
if d["fallback"]:
    eyebrow = "Least-bad option. Escalate to a planner (Risk averse)"
elif custom:
    eyebrow = "Custom weights"
else:
    eyebrow = f"Recommended route ({PROFILE_LABELS[profile]})"

verdict, verdict_detail = robustness_verdict(sens)
if d["fallback"]:
    pill = '<span class="verdict-pill fallback-pill">No route meets every constraint</span>'
elif len(d["candidates"]) == 1:
    pill = '<span class="verdict-pill">Only viable option</span>'
elif verdict:
    verdict_label = {"ROBUST": "Robust", "MODERATE": "Moderate", "CLOSE CALL": "Close call"}[verdict]
    pill = f'<span class="verdict-pill">{verdict_label}</span>'
else:
    pill = ""

facts = [
    f'<span class="fact"><strong>${winner["cost_usd"]:,.0f}</strong> total cost</span>',
    f'<span class="fact"><strong>{winner["expected_days"]:.1f} days expected, '
    f'{winner["p90_days"]:.1f} at P90</strong></span>',
    f'<span class="fact"><strong>{winner["delay_prob"]:.0%}</strong> chance of delay</span>',
    f'<span class="fact"><strong>{winner["geopolitical_risk"]:.2f}</strong> geopolitical risk</span>',
]
if constraints["deadline_days"] is not None:
    facts.append(f'<span class="fact"><strong>{winner["on_time_prob"]:.0%}</strong> on time</span>')
robustness_copy = ""
if sens and verdict_detail:
    robustness_copy = (
        '<div class="robustness-copy">'
        + html.escape(verdict_detail[0].upper() + verdict_detail[1:] + ".")
        + "</div>"
    )

st.markdown(
    '<div class="page-title">Route recommender</div>'
    f'<div class="page-subtitle">{html.escape(s["shipment_id"])}: {s["weight_t"]:.1f} t of '
    f'{html.escape(s["cargo_type"].replace("_", " "))}, {html.escape(s["origin"])} to '
    f'{html.escape(s["destination"])}, cargo value ${s["cargo_value_usd"]:,.0f}</div>',
    unsafe_allow_html=True,
)
st.markdown(
    f'<div class="recommendation{" fallback" if d["fallback"] else ""}">'
    f'<div class="rec-top"><span class="rec-eyebrow">{html.escape(eyebrow)}</span>{pill}</div>'
    f'<div class="route-title">{html.escape(winner_id)}: {html.escape(winner["name"])}</div>'
    f'<div class="route-strip">{route_strip(winner_source)}</div>'
    f'<div class="facts">{"".join(facts)}</div>{robustness_copy}</div>',
    unsafe_allow_html=True,
)

chart_rows = []
for route_id, route in d["routes"].items():
    chart_rows.append({
        "ID": route_id,
        "Route": route["name"],
        "Status": status_for(route_id, route, d),
        "Cost": route["cost_usd"],
        "Days": route["expected_days"],
        "Delay": route["delay_prob"] * 100,
        "Geo risk": route["geopolitical_risk"],
    })
chart_df = pd.DataFrame(chart_rows)
present_statuses = [name for name in STATUS_ORDER if name in set(chart_df["Status"])]
legend_items = []
for name in present_statuses:
    fill, stroke, _ = STATUS_COLORS[name]
    legend_items.append(
        f'<span class="legend-item"><span class="legend-dot" style="background:{fill};'
        f'border:1.5px solid {stroke};"></span>{name}</span>'
    )
legend_html = (
    '<div class="chart-legend">' + "".join(legend_items)
    + '<span class="legend-note">Larger dot = higher chance of delay</span></div>'
)
chart_colors = [STATUS_COLORS[name][0] for name in present_statuses]
chart_strokes = [STATUS_COLORS[name][1] for name in present_statuses]
min_cost = float(chart_df["Cost"].min())
max_cost = float(chart_df["Cost"].max())
max_days = float(chart_df["Days"].max())
x_domain = [min_cost / 1.6, max_cost * 1.9]
tick_candidates = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000]
x_ticks = [value for value in tick_candidates if x_domain[0] <= value <= x_domain[1]]
x_encoding = alt.X(
    "Cost:Q",
    scale=alt.Scale(type="log", domain=x_domain, nice=False),
    axis=alt.Axis(title="Total cost (log scale)", format="$,.0f", values=x_ticks, grid=True),
)
y_encoding = alt.Y(
    "Days:Q",
    scale=alt.Scale(domain=[0, max_days * 1.15], nice=False),
    axis=alt.Axis(title="Expected transit (days)", tickCount=6, grid=True),
)
points = alt.Chart(chart_df).mark_circle(opacity=1, strokeWidth=1.5).encode(
    x=x_encoding,
    y=y_encoding,
    size=alt.Size("Delay:Q", scale=alt.Scale(range=[90, 650]), legend=None),
    fill=alt.Fill("Status:N", scale=alt.Scale(domain=present_statuses, range=chart_colors), legend=None),
    stroke=alt.Stroke("Status:N", scale=alt.Scale(domain=present_statuses, range=chart_strokes), legend=None),
    tooltip=[
        alt.Tooltip("ID:N", title="ID"),
        alt.Tooltip("Route:N", title="Route"),
        alt.Tooltip("Status:N", title="Status"),
        alt.Tooltip("Cost:Q", title="Cost", format="$,.0f"),
        alt.Tooltip("Days:Q", title="Days", format=".1f"),
        alt.Tooltip("Delay:Q", title="Delay", format=".0f"),
        alt.Tooltip("Geo risk:Q", title="Geo risk", format=".2f"),
    ],
)
labels = alt.Chart(chart_df).mark_text(
    align="left", baseline="middle", dx=13, color=PALETTE["ink"], fontSize=12,
).encode(x=x_encoding, y=y_encoding, text="ID:N")
chart = alt.layer(points, labels).properties(height=400).configure(
    background=PALETTE["white"],
).configure_view(strokeWidth=0).configure_axis(
    gridColor=PALETTE["grid"],
    domainColor=PALETTE["border"],
    labelColor=PALETTE["muted"],
    titleColor=PALETTE["muted"],
    labelFontSize=11,
    titleFontSize=12,
    titleFontWeight="normal",
    tickColor=PALETTE["border"],
)

explanation_lines = explain(d, s, sens).splitlines()
cleaned_lines = []
skip_following = False
for line in explanation_lines:
    if skip_following:
        skip_following = False
        continue
    if line.startswith(("Shipment ", "Priority:", "RECOMMENDED:")):
        if line.startswith("RECOMMENDED:"):
            skip_following = True
        continue
    if line.startswith("Confidence:"):
        continue
    cleaned_lines.append(line)
explanation_text = re.sub(r"\n{3,}", "\n\n", "\n".join(cleaned_lines)).strip()
escaped_explanation = html.escape(explanation_text).replace("\n", "<br>")

left, right = st.columns(2, gap="large")
with left:
    st.markdown('<div class="section-heading">Why this route</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="explanation-box">{escaped_explanation}</div>', unsafe_allow_html=True)
with right:
    st.markdown('<div class="section-heading">Cost vs. time</div>', unsafe_allow_html=True)
    st.markdown(legend_html, unsafe_allow_html=True)
    st.altair_chart(chart, width="stretch")

st.markdown('<div class="section-heading">All options</div>', unsafe_allow_html=True)
table_rows = []
for route_id, route in d["routes"].items():
    status = status_for(route_id, route, d)
    if status == "Recommended":
        why = ""
    elif status == "Infeasible":
        why = route["violations"][0]
    elif status == "Dominated":
        why = f"beaten on every criterion by {route['dominated_by']}"
    elif status == "Least bad":
        why = f"escalate: {'; '.join(route['violations'])}"
    else:
        why = f"penalty {route['score']:.0f} vs {winner['score']:.0f}"
    table_rows.append({
        "ID": route_id,
        "Route": route["name"],
        "Status": status,
        "Cost": route["cost_usd"],
        "Days": route["expected_days"],
        "P90": route["p90_days"],
        "Delay": route["delay_prob"] * 100,
        "Geo risk": route["geopolitical_risk"],
        "Why": why,
    })
options_df = pd.DataFrame(table_rows)
winner_name = winner["name"]
options_styler = options_df.style.map(style_status, subset=["Status"])
options_styler = options_styler.map(lambda value: style_winner_route(value, winner_name), subset=["Route"])
st.dataframe(
    options_styler,
    hide_index=True,
    width="stretch",
    height=min(440, 38 + 36 * len(options_df)),
    column_config={
        "ID": st.column_config.TextColumn("ID", width=52),
        "Route": st.column_config.TextColumn("Route", width=165),
        "Status": st.column_config.TextColumn("Status", width=118),
        "Cost": st.column_config.NumberColumn("Cost", format="$%,.0f", width=86),
        "Days": st.column_config.NumberColumn("Days", help="Expected transit days", format="%.1f", width=66),
        "P90": st.column_config.NumberColumn("P90", help="90th-percentile transit days", format="%.1f", width=62),
        "Delay": st.column_config.NumberColumn("Delay", help="Chance of more than 1 day delay", format="%.0f%%", width=70),
        "Geo risk": st.column_config.NumberColumn("Geo risk", format="%.2f", width=78),
        "Why": st.column_config.TextColumn("Why", width="large"),
    },
)

st.markdown('<div class="section-heading">Best route under each profile</div>', unsafe_allow_html=True)
sweep = profile_sweep(s, m)
profile_cards = []
for profile_key in PROFILE_ORDER:
    best_id = sweep[profile_key]
    best_route = d["routes"].get(best_id)
    if best_route is None:
        best_route = next(route for route in s["routes"] if route["route_id"] == best_id)
    active_class = " active" if not custom and profile_key == profile else ""
    profile_cards.append(
        f'<div class="profile-card{active_class}">'
        f'<div class="profile-name">{html.escape(PROFILE_LABELS[profile_key])}</div>'
        f'<div class="profile-route">{html.escape(best_id + ": " + best_route["name"])}</div>'
        "</div>"
    )
st.markdown('<div class="profile-grid">' + "".join(profile_cards) + "</div>", unsafe_allow_html=True)
