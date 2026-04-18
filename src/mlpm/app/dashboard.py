from __future__ import annotations

import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from mlpm.config.settings import settings
from mlpm.evaluation.settled import compute_settled_calibration
from mlpm.evaluation.strategy import select_champion_model
from mlpm.storage.duckdb import connect_read_only


@st.cache_data(ttl=30)
def load_table(sql: str) -> pd.DataFrame:
    conn = connect_read_only(settings().duckdb_path)
    try:
        return conn.execute(sql).fetchdf()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Background job plumbing (sidebar actions)
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]
JOBS_DIR = Path(settings().duckdb_path).resolve().parent / "dashboard_jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)

JOB_COMMANDS: dict[str, dict[str, object]] = {
    "collect-once": {
        "label": "Collect snapshot",
        "help": "Pull fresh MLB games, Kalshi & Polymarket quotes, and recompute discrepancies.",
        "args": ["collect-once"],
    },
    "sync-results": {
        "label": "Sync game results",
        "help": "Backfill recent final MLB game results into DuckDB.",
        "args": ["sync-results"],
    },
    "train-game-model": {
        "label": "Train model",
        "help": "Retrain and persist the MLB game-outcome model.",
        "args": ["train-game-model"],
    },
}


def _init_job_state() -> None:
    if "jobs" not in st.session_state:
        st.session_state["jobs"] = []


def _start_job(command_key: str) -> None:
    spec = JOB_COMMANDS[command_key]
    job_id = datetime.now().strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]
    log_path = JOBS_DIR / f"{command_key}-{job_id}.log"
    log_file = open(log_path, "w", buffering=1)
    proc = subprocess.Popen(
        [sys.executable, "-m", "mlpm.cli", *spec["args"]],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        cwd=str(PROJECT_ROOT),
    )
    st.session_state["jobs"].insert(
        0,
        {
            "id": job_id,
            "command": command_key,
            "label": spec["label"],
            "pid": proc.pid,
            "log_path": str(log_path),
            "started_at": datetime.now(),
            "finished_at": None,
            "status": "running",
            "returncode": None,
            "_proc": proc,
            "_log_file": log_file,
        },
    )


def _poll_jobs() -> None:
    cache_needs_clear = False
    for job in st.session_state.get("jobs", []):
        if job["status"] != "running":
            continue
        proc = job.get("_proc")
        if proc is None:
            continue
        rc = proc.poll()
        if rc is not None:
            job["status"] = "success" if rc == 0 else "failed"
            job["returncode"] = rc
            job["finished_at"] = datetime.now()
            log_file = job.pop("_log_file", None)
            if log_file is not None:
                try:
                    log_file.close()
                except Exception:
                    pass
            cache_needs_clear = True
    if cache_needs_clear:
        # Newly finished jobs may have changed the DB; drop cached query results.
        load_table.clear()


def _read_log_tail(path: str, n_lines: int = 40) -> str:
    try:
        with open(path, "r", errors="replace") as f:
            lines = f.readlines()
    except FileNotFoundError:
        return ""
    return "".join(lines[-n_lines:])


def _format_duration(start: datetime, end: datetime | None) -> str:
    end = end or datetime.now()
    seconds = (end - start).total_seconds()
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes}m {secs}s"


_init_job_state()
_poll_jobs()

st.set_page_config(page_title="MLPM Dashboard", layout="wide")
st.title("MLB Market Inconsistency Dashboard")

# ---------------------------------------------------------------------------
# Sidebar: action buttons + job status
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Actions")

    any_running = any(j["status"] == "running" for j in st.session_state["jobs"])

    for key, spec in JOB_COMMANDS.items():
        if st.button(spec["label"], help=spec["help"], use_container_width=True, key=f"btn_{key}"):
            _start_job(key)
            st.rerun()

    if st.button(
        "Refresh data (clear cache)",
        help="Clear cached query results and re-query DuckDB.",
        use_container_width=True,
        key="btn_refresh_cache",
    ):
        load_table.clear()
        st.rerun()

    st.divider()
    st.subheader("Job status")

    if any_running:
        st.info("A job is running. Click 'Refresh status' to poll for updates.")
    if st.button("Refresh status", use_container_width=True, key="btn_refresh_status"):
        st.rerun()

    jobs = st.session_state["jobs"]
    if not jobs:
        st.caption("No jobs launched yet this session.")
    else:
        for job in jobs[:10]:
            status = job["status"]
            icon = {"running": "⏳", "success": "✅", "failed": "❌"}.get(status, "•")
            title = f"{icon} {job['label']} — {status} ({_format_duration(job['started_at'], job['finished_at'])})"
            with st.expander(title, expanded=(status == "running")):
                st.caption(
                    f"command: `mlpm {' '.join(JOB_COMMANDS[job['command']]['args'])}`  "
                    f"| pid: {job['pid']}  "
                    f"| started: {job['started_at'].strftime('%H:%M:%S')}"
                )
                if job["returncode"] is not None:
                    st.caption(f"return code: {job['returncode']}")
                tail = _read_log_tail(job["log_path"])
                if tail:
                    st.code(tail, language="text")
                else:
                    st.caption("(no log output yet)")
                st.caption(f"log: `{job['log_path']}`")

summary = load_table(
    """
    SELECT
        COUNT(*) AS total_discrepancies,
        SUM(CASE WHEN flagged THEN 1 ELSE 0 END) AS flagged_discrepancies,
        AVG(ABS(gap_bps)) AS avg_abs_gap_bps
    FROM discrepancies_deduped
    """
)
settled_summary = load_table(
    """
    SELECT
        model_name,
        COUNT(*) AS games,
        AVG(CASE WHEN correct_prediction THEN 1.0 ELSE 0.0 END) AS accuracy,
        AVG(
            CASE
                WHEN actual_home_win = 1 THEN -LN(GREATEST(home_win_prob, 1e-6))
                ELSE -LN(GREATEST(1.0 - home_win_prob, 1e-6))
            END
        ) AS log_loss
    FROM settled_predictions_deduped
    GROUP BY model_name
    ORDER BY games DESC, model_name
    """
)
settled_windows = load_table(
    """
    WITH latest_date AS (
        SELECT MAX(game_date) AS max_game_date
        FROM settled_predictions_deduped
    ),
    base AS (
        SELECT s.*, ld.max_game_date
        FROM settled_predictions_deduped s
        CROSS JOIN latest_date ld
    )
    SELECT
        model_name,
        COUNT(*) AS all_games,
        AVG(CASE WHEN correct_prediction THEN 1.0 ELSE 0.0 END) AS all_accuracy,
        AVG(
            CASE
                WHEN actual_home_win = 1 THEN -LN(GREATEST(home_win_prob, 1e-6))
                ELSE -LN(GREATEST(1.0 - home_win_prob, 1e-6))
            END
        ) AS all_log_loss,
        SUM(CASE WHEN game_date >= max_game_date - 6 THEN 1 ELSE 0 END) AS last_7d_games,
        AVG(CASE WHEN game_date >= max_game_date - 6 THEN CASE WHEN correct_prediction THEN 1.0 ELSE 0.0 END END) AS last_7d_accuracy,
        AVG(CASE WHEN game_date >= max_game_date - 6 THEN CASE
                WHEN actual_home_win = 1 THEN -LN(GREATEST(home_win_prob, 1e-6))
                ELSE -LN(GREATEST(1.0 - home_win_prob, 1e-6))
            END END) AS last_7d_log_loss,
        SUM(CASE WHEN game_date >= max_game_date - 29 THEN 1 ELSE 0 END) AS last_30d_games,
        AVG(CASE WHEN game_date >= max_game_date - 29 THEN CASE WHEN correct_prediction THEN 1.0 ELSE 0.0 END END) AS last_30d_accuracy,
        AVG(CASE WHEN game_date >= max_game_date - 29 THEN CASE
                WHEN actual_home_win = 1 THEN -LN(GREATEST(home_win_prob, 1e-6))
                ELSE -LN(GREATEST(1.0 - home_win_prob, 1e-6))
            END END) AS last_30d_log_loss
    FROM base
    GROUP BY model_name
    ORDER BY all_games DESC, model_name
    """
)
bet_opportunities = load_table(
    """
    SELECT
        game_date,
        event_start_time,
        model_name,
        source,
        team,
        opponent_team,
        model_prob,
        market_prob,
        edge_bps,
        expected_value,
        is_actionable,
        is_champion
    FROM bet_opportunities_deduped
    ORDER BY game_date DESC, event_start_time DESC, edge_bps DESC
    LIMIT 200
    """
)
strategy_windows = load_table(
    """
    WITH all_settled AS (
        SELECT
            b.game_date,
            b.model_name,
            b.edge_bps,
            b.stake_units,
            b.implied_decimal_odds,
            CASE WHEN r.winner_team = b.team THEN TRUE ELSE FALSE END AS won_bet,
            CASE
                WHEN r.winner_team = b.team THEN (b.implied_decimal_odds - 1.0) * b.stake_units
                ELSE -1.0 * b.stake_units
            END AS realized_return_units
        FROM bet_opportunities_deduped b
        JOIN game_results_deduped r ON b.game_id = r.game_id
    ),
    latest_date AS (
        SELECT MAX(game_date) AS max_game_date FROM all_settled
    ),
    base AS (
        SELECT s.*, ld.max_game_date
        FROM all_settled s
        CROSS JOIN latest_date ld
    )
    SELECT
        model_name,
        COUNT(*) AS all_bets,
        ROUND(AVG(CAST(won_bet AS DOUBLE)), 4) AS all_win_rate,
        ROUND(SUM(realized_return_units) / NULLIF(SUM(stake_units), 0), 4) AS all_roi,
        ROUND(SUM(realized_return_units), 2) AS all_units,
        ROUND(AVG(edge_bps), 1) AS all_avg_edge_bps,
        SUM(CASE WHEN game_date >= max_game_date - 6 THEN 1 ELSE 0 END) AS last_7d_bets,
        ROUND(AVG(CASE WHEN game_date >= max_game_date - 6 THEN CAST(won_bet AS DOUBLE) END), 4) AS last_7d_win_rate,
        ROUND(
            SUM(CASE WHEN game_date >= max_game_date - 6 THEN realized_return_units ELSE 0 END) /
            NULLIF(SUM(CASE WHEN game_date >= max_game_date - 6 THEN stake_units ELSE 0 END), 0), 4
        ) AS last_7d_roi,
        SUM(CASE WHEN game_date >= max_game_date - 29 THEN 1 ELSE 0 END) AS last_30d_bets,
        ROUND(AVG(CASE WHEN game_date >= max_game_date - 29 THEN CAST(won_bet AS DOUBLE) END), 4) AS last_30d_win_rate,
        ROUND(
            SUM(CASE WHEN game_date >= max_game_date - 29 THEN realized_return_units ELSE 0 END) /
            NULLIF(SUM(CASE WHEN game_date >= max_game_date - 29 THEN stake_units ELSE 0 END), 0), 4
        ) AS last_30d_roi
    FROM base
    GROUP BY model_name
    ORDER BY all_bets DESC, model_name
    """
)
recent_settled = load_table(
    """
    SELECT
        game_date,
        model_name,
        away_team,
        home_team,
        predicted_winner,
        winner_team,
        home_win_prob,
        correct_prediction
    FROM settled_predictions_deduped
    ORDER BY game_date DESC, event_start_time DESC
    LIMIT 100
    """
)
current = load_table(
    """
    SELECT
        d.game_id,
        d.snapshot_ts,
        d.source,
        d.market_id,
        d.team,
        d.market_prob,
        d.model_prob,
        d.gap_bps,
        d.flagged
    FROM discrepancies_deduped d
    ORDER BY snapshot_ts DESC
    LIMIT 200
    """
)
model_features = load_table(
    """
    WITH latest_model AS (
        SELECT
            *,
            ROW_NUMBER() OVER (
                PARTITION BY game_id, team
                ORDER BY
                    CASE
                        WHEN model_name = 'mlb_win_histgb_v1' THEN 4
                        WHEN model_name = 'mlb_win_logreg_v2' THEN 3
                        WHEN model_name = 'mlb_win_logreg_v1' THEN 2
                        WHEN model_name = 'team_form_logit_v2' THEN 1
                        WHEN model_name = 'team_record_logit_v1' THEN 0
                        ELSE 0
                    END DESC,
                    collection_run_ts DESC,
                    snapshot_ts DESC
            ) AS rn
        FROM model_predictions_deduped
    )
    SELECT
        game_id,
        team,
        opponent_team,
        model_name,
        model_prob,
        season_win_pct,
        recent_win_pct,
        venue_win_pct,
        run_diff_per_game,
        streak,
        elo_rating,
        season_runs_scored_per_game,
        season_runs_allowed_per_game,
        recent_runs_scored_per_game,
        recent_runs_allowed_per_game,
        rest_days,
        venue_streak,
        travel_switch,
        is_doubleheader,
        starter_era,
        starter_whip,
        starter_strikeouts_per_9,
        starter_walks_per_9,
        bullpen_innings_3d,
        bullpen_pitches_3d,
        relievers_used_3d
    FROM latest_model
    WHERE rn = 1
    """
)
current_with_features = current.merge(model_features, on=["game_id", "team", "model_prob"], how="left")

st.subheader("Summary")
st.dataframe(summary, use_container_width=True)

st.subheader("Settled Predictions")
st.dataframe(settled_summary, use_container_width=True)

st.subheader("Settled Rolling Windows")
st.dataframe(settled_windows, use_container_width=True)

# ---------------------------------------------------------------------------
# Calibration view
# ---------------------------------------------------------------------------

st.subheader("Calibration (predicted vs empirical)")


@st.cache_data(ttl=30)
def _load_calibration(bins: int) -> pd.DataFrame:
    return compute_settled_calibration(bins=bins)


calibration_df = _load_calibration(bins=10)
if calibration_df.empty:
    st.info("No settled predictions yet — calibration will appear after games settle.")
else:
    cal_models = sorted(calibration_df["model_name"].dropna().unique().tolist())
    selected_cal_models = st.multiselect(
        "Models to plot",
        options=cal_models,
        default=cal_models,
        key="calibration_models",
    )
    view = calibration_df[calibration_df["model_name"].isin(selected_cal_models)].copy()
    if view.empty:
        st.caption("No data for the selected models.")
    else:
        st.caption(
            "Each point shows the average predicted home-win probability in a decile "
            "vs. the empirical home-win rate of those games. Perfectly calibrated models "
            "sit on the y=x diagonal."
        )
        ref = pd.DataFrame(
            {"avg_predicted_prob": [0.0, 1.0], "actual_home_win_rate": [0.0, 1.0], "model_name": ["perfect_calibration"] * 2}
        )
        plot_df = pd.concat(
            [view[["avg_predicted_prob", "actual_home_win_rate", "model_name"]], ref], ignore_index=True
        )
        st.scatter_chart(
            plot_df,
            x="avg_predicted_prob",
            y="actual_home_win_rate",
            color="model_name",
            height=380,
        )
        summary = (
            view.groupby("model_name")
            .apply(
                lambda g: pd.Series(
                    {
                        "buckets": int(len(g)),
                        "games": int(g["games"].sum()),
                        "mean_abs_error": float((g["abs_error"] * g["games"]).sum() / g["games"].sum()),
                        "max_abs_error": float(g["abs_error"].max()),
                    }
                )
            )
            .reset_index()
            .sort_values("mean_abs_error")
        )
        st.dataframe(summary, use_container_width=True)
        with st.expander("Per-bucket detail"):
            st.dataframe(view, use_container_width=True)

# ---------------------------------------------------------------------------
# Feature importance (latest training run)
# ---------------------------------------------------------------------------

st.subheader("Feature importance (latest training run)")
feature_importances = load_table(
    """
    SELECT
        model_name,
        feature,
        importance,
        importance_std,
        rank,
        trained_at,
        train_start_date,
        train_end_date,
        rows_train,
        rows_valid,
        method
    FROM latest_feature_importances
    ORDER BY model_name, rank
    """
)

if feature_importances.empty:
    st.info(
        "No feature importance has been persisted yet. Run 'Train model' from the sidebar "
        "to compute and store permutation importances."
    )
else:
    importance_models = sorted(feature_importances["model_name"].dropna().unique().tolist())
    latest_trained_at = feature_importances["trained_at"].max()
    st.caption(f"Latest training run: **{latest_trained_at}**")

    top_n = st.slider("Top N features per model", min_value=5, max_value=len(feature_importances["feature"].unique()), value=10, key="fi_top_n")
    selected_fi_models = st.multiselect(
        "Models",
        options=importance_models,
        default=importance_models,
        key="fi_models",
    )
    fi_view = feature_importances[feature_importances["model_name"].isin(selected_fi_models)].copy()
    fi_view = fi_view[fi_view["rank"] <= top_n]

    if fi_view.empty:
        st.caption("No importance rows to display.")
    else:
        st.bar_chart(
            fi_view,
            x="feature",
            y="importance",
            color="model_name",
            height=420,
        )
        with st.expander("Importance table"):
            st.dataframe(
                fi_view[["model_name", "rank", "feature", "importance", "importance_std"]],
                use_container_width=True,
            )

st.subheader("Current Bet Opportunities")
st.dataframe(bet_opportunities, use_container_width=True)

champion_model = select_champion_model()

st.subheader("Strategy Performance")
if champion_model:
    st.caption(f"Champion model: **{champion_model}**")

champion_history = load_table(
    """
    SELECT *
    FROM champion_history
    ORDER BY selected_at DESC
    LIMIT 20
    """
)
if not champion_history.empty:
    with st.expander("Champion decision history (latest 20)"):
        st.caption(
            "Each row is a guardrailed champion-selection decision. Actions: "
            "`switched` means the challenger's lower CI cleared the incumbent; "
            "`kept_incumbent` means it did not."
        )
        st.dataframe(
            champion_history[
                [
                    "selected_at",
                    "action",
                    "chosen_model",
                    "incumbent_model",
                    "challenger_model",
                    "challenger_bets",
                    "challenger_roi",
                    "challenger_win_rate",
                    "challenger_ci_lower",
                    "challenger_ci_upper",
                    "incumbent_bets",
                    "incumbent_roi",
                    "incumbent_win_rate",
                    "confidence",
                    "reason",
                ]
            ],
            use_container_width=True,
        )

if not strategy_windows.empty:
    display_df = strategy_windows.copy()
    display_df.insert(0, "champion", display_df["model_name"] == champion_model)

    def _highlight_champion(row: pd.Series) -> list[str]:
        if row["champion"]:
            return ["background-color: #1a3a1a; color: #90ee90"] * len(row)
        return [""] * len(row)

    styled = display_df.style.apply(_highlight_champion, axis=1).format(
        {
            "all_win_rate": "{:.1%}",
            "all_roi": "{:.1%}",
            "last_7d_win_rate": "{:.1%}",
            "last_7d_roi": "{:.1%}",
            "last_30d_win_rate": "{:.1%}",
            "last_30d_roi": "{:.1%}",
        },
        na_rep="-",
    )
    st.dataframe(styled, use_container_width=True)
else:
    st.info("No settled bet opportunities yet.")

st.subheader("Recent settled picks")
st.dataframe(recent_settled, use_container_width=True)

st.subheader("Recent discrepancies")
source_filter = st.multiselect(
    "Sources",
    options=sorted(current_with_features["source"].dropna().unique().tolist()),
    default=sorted(current_with_features["source"].dropna().unique().tolist()),
)
filtered = current_with_features[current_with_features["source"].isin(source_filter)].copy()
st.dataframe(filtered, use_container_width=True)

st.subheader("Model feature context")
feature_columns = [
    "snapshot_ts",
    "source",
    "team",
    "opponent_team",
    "market_prob",
    "model_prob",
    "gap_bps",
    "season_win_pct",
    "recent_win_pct",
    "venue_win_pct",
    "run_diff_per_game",
    "streak",
    "elo_rating",
    "season_runs_scored_per_game",
    "season_runs_allowed_per_game",
    "recent_runs_scored_per_game",
    "recent_runs_allowed_per_game",
    "rest_days",
    "venue_streak",
    "travel_switch",
    "is_doubleheader",
    "starter_era",
    "starter_whip",
    "starter_strikeouts_per_9",
    "starter_walks_per_9",
    "bullpen_innings_3d",
    "bullpen_pitches_3d",
    "relievers_used_3d",
]
st.dataframe(filtered[feature_columns], use_container_width=True)

if not filtered.empty:
    st.subheader("Gap history")
    chart_df = filtered[["snapshot_ts", "gap_bps", "team"]].copy()
    chart_df["snapshot_ts"] = pd.to_datetime(chart_df["snapshot_ts"])
    st.line_chart(chart_df, x="snapshot_ts", y="gap_bps", color="team")
