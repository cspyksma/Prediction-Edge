import { useEffect, useMemo, useState, type ReactNode } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Area, AreaChart, Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api } from "./api";
import type { Opportunity } from "./types";

export function CockpitPage() {
  const summary = useQuery({ queryKey: ["summary"], queryFn: api.summary, refetchInterval: 15000 });
  const opportunities = useQuery({ queryKey: ["opportunities"], queryFn: api.opportunities, refetchInterval: 15000 });
  const rows = opportunities.data?.items ?? [];
  const championModelName = opportunities.data?.champion_model ?? summary.data?.champion_model ?? null;
  const championRows = useMemo(() => {
    if (championModelName) {
      return rows.filter((row) => row.model_name === championModelName);
    }
    return rows.filter((row) => row.is_champion);
  }, [championModelName, rows]);
  const displayRows = championRows.length > 0 ? championRows : rows;
  const [selectedOpportunityKey, setSelectedOpportunityKey] = useState<string | null>(null);

  useEffect(() => {
    if (!selectedOpportunityKey && displayRows.length > 0) {
      setSelectedOpportunityKey(opportunityKey(displayRows[0]));
    }
  }, [displayRows, selectedOpportunityKey]);

  useEffect(() => {
    if (!selectedOpportunityKey) return;
    if (!displayRows.some((row) => opportunityKey(row) === selectedOpportunityKey)) {
      setSelectedOpportunityKey(displayRows[0] ? opportunityKey(displayRows[0]) : null);
    }
  }, [displayRows, selectedOpportunityKey]);

  const selectedOpportunity =
    displayRows.find((row) => opportunityKey(row) === selectedOpportunityKey) ?? displayRows[0] ?? null;
  const detail = useQuery({
    queryKey: ["game-detail", selectedOpportunity?.game_id],
    queryFn: () => api.gameDetail(selectedOpportunity!.game_id),
    enabled: selectedOpportunity !== null,
    refetchInterval: 15000,
  });

  const modelPressure = useMemo(() => aggregateCounts(rows, (row) => row.model_name), [rows]);
  const sourcePressure = useMemo(() => aggregateCounts(rows, (row) => row.source ?? "unknown"), [rows]);
  const gapHistory = (detail.data?.gap_history ?? []).filter((row) => {
    if (!selectedOpportunity) return false;
    return row.team === selectedOpportunity.team && row.source === selectedOpportunity.source;
  });
  const selectedFeatures =
    detail.data?.features?.find(
      (feature) =>
        feature.team === selectedOpportunity?.team && feature.model_name === selectedOpportunity?.model_name,
    ) ??
    detail.data?.features?.find((feature) => feature.team === selectedOpportunity?.team) ??
    null;

  return (
    <section className="workspace cockpit-workspace">
      <div className="market-tape panel">
        <TapeStat label="Status" value={summary.data?.stale_data ? "Feed stale" : "Market live"} tone={summary.data?.stale_data ? "warn" : "ok"} />
        <TapeStat label="Champion" value={summary.data?.champion_model ?? "Unknown"} />
        <TapeStat label="Actionables" value={String(summary.data?.actionable_bets ?? 0)} />
        <TapeStat label="Max edge" value={summary.data ? `${summary.data.max_edge_bps} bps` : "-"} tone="accent" />
        <TapeStat label="Snapshot" value={summary.data?.latest_snapshot_ts ? formatDateTime(summary.data.latest_snapshot_ts) : "-"} />
      </div>

      <div className="cockpit-grid">
        <div className="panel blotter-panel">
          <div className="panel-header">
            <div>
              <p className="panel-kicker">Primary blotter</p>
              <h3>{championRows.length > 0 ? "Champion ladder" : "Opportunity ladder"}</h3>
            </div>
            <span>
              {displayRows.length}
              {championRows.length > 0 ? ` champion rows (${opportunities.data?.total ?? 0} total)` : " actionable rows"}
            </span>
          </div>
          <OpportunityBlotter
            rows={displayRows}
            selectedOpportunityKey={selectedOpportunity ? opportunityKey(selectedOpportunity) : null}
            onSelect={setSelectedOpportunityKey}
          />
        </div>

        <div className="side-stack">
          <div className="panel ticket-panel">
            <div className="panel-header">
              <div>
                <p className="panel-kicker">Selected ticket</p>
                <h3>{selectedOpportunity ? `${selectedOpportunity.team} vs ${selectedOpportunity.opponent_team ?? "-"}` : "No selection"}</h3>
              </div>
              <span>{selectedOpportunity?.source ?? "-"}</span>
            </div>
            {selectedOpportunity ? (
              <>
                <div className="ticket-signal">
                  <div>
                    <span className="ticket-label">Signal</span>
                    <strong className="signal-long">LONG {selectedOpportunity.team}</strong>
                  </div>
                  <div className="ticket-price">
                    <span>{selectedOpportunity.model_name}</span>
                    <strong>{selectedOpportunity.edge_bps} bps</strong>
                  </div>
                </div>

                <div className="ticket-grid">
                  <MetricBlock label="Model fair" value={fmtPct(selectedOpportunity.model_prob)} />
                  <MetricBlock label="Market" value={fmtPct(selectedOpportunity.market_prob)} />
                  <MetricBlock label="Expected value" value={selectedOpportunity.expected_value == null ? "-" : fmtPct(selectedOpportunity.expected_value)} />
                  <MetricBlock label="First pitch" value={selectedOpportunity.event_start_time ? formatDateTime(selectedOpportunity.event_start_time) : "-"} />
                </div>

                <div className="mini-divider" />

                <div className="detail-list">
                  <DetailRow label="Venue" value={detail.data ? `${detail.data.away_team} @ ${detail.data.home_team}` : "-"} />
                  <DetailRow label="Quote count" value={String(detail.data?.quotes.length ?? 0)} />
                  <DetailRow label="Flagged quotes" value={String(detail.data?.quotes.filter((quote) => quote.flagged).length ?? 0)} />
                  <DetailRow label="Starter ERA" value={selectedFeatures?.starter_era != null ? selectedFeatures.starter_era.toFixed(2) : "-"} />
                  <DetailRow label="Team Elo" value={selectedFeatures?.elo_rating != null ? Math.round(selectedFeatures.elo_rating).toString() : "-"} />
                  <DetailRow label="Bullpen 3d" value={selectedFeatures?.bullpen_innings_3d != null ? `${selectedFeatures.bullpen_innings_3d.toFixed(1)} IP` : "-"} />
                </div>
              </>
            ) : (
              <p className="empty-copy">No actionable bet selected.</p>
            )}
          </div>

          <div className="panel">
            <div className="panel-header">
              <div>
                <p className="panel-kicker">Quote stack</p>
                <h3>Market microstructure</h3>
              </div>
              <span>{detail.data?.quotes.length ?? 0} quotes</span>
            </div>
            <SimpleTable
              headers={["Source", "Team", "Gap", "Mkt", "Fair"]}
              rows={(detail.data?.quotes ?? []).slice(0, 8).map((quote) => [
                quote.source ?? "-",
                quote.team,
                <span className={quote.gap_bps >= 0 ? "text-positive" : "text-negative"}>{quote.gap_bps} bps</span>,
                fmtPct(quote.market_prob),
                fmtPct(quote.model_prob),
              ])}
            />
          </div>
        </div>
      </div>

      <div className="workspace-grid cockpit-lower-grid">
        <div className="panel chart-panel">
          <div className="panel-header">
            <div>
              <p className="panel-kicker">Heat map</p>
              <h3>Model pressure</h3>
            </div>
            <span>By actionable count</span>
          </div>
          <BarPanel data={modelPressure} dataKey="count" color="#00d0a3" />
        </div>

        <div className="panel chart-panel">
          <div className="panel-header">
            <div>
              <p className="panel-kicker">Source depth</p>
              <h3>Venue flow</h3>
            </div>
            <span>By exchange</span>
          </div>
          <BarPanel data={sourcePressure} dataKey="count" color="#4fa3ff" />
        </div>

        <div className="panel chart-panel">
          <div className="panel-header">
            <div>
              <p className="panel-kicker">Trend tape</p>
              <h3>Selected gap history</h3>
            </div>
            <span>{selectedOpportunity?.team ?? "No team"}</span>
          </div>
          <GapHistoryChart data={gapHistory} />
        </div>
      </div>
    </section>
  );
}

export function ResearchPage() {
  const strategies = useQuery({ queryKey: ["strategies"], queryFn: api.strategies });
  const calibration = useQuery({ queryKey: ["calibration"], queryFn: api.calibration });
  const featureImportance = useQuery({ queryKey: ["feature-importance"], queryFn: api.featureImportance });
  const coverage = useQuery({ queryKey: ["training-coverage"], queryFn: api.trainingCoverage });
  const roster = useQuery({ queryKey: ["model-roster"], queryFn: api.modelRoster });
  const sizing = useQuery({ queryKey: ["sizing-comparison"], queryFn: api.sizingComparison });

  const rosterChart = (roster.data ?? [])
    .filter((row) => row.role !== "ensemble")
    .map((row) => ({
      name: compactLabel(row.model_name),
      score: row.log_loss == null ? 0 : Number((1 - row.log_loss).toFixed(3)),
    }));
  const strategyChart = (strategies.data ?? [])
    .slice(0, 8)
    .map((row) => ({ name: compactLabel(row.strategy_name), roi: Number((row.roi * 100).toFixed(2)) }));

  const sizingPolicyLabels: Record<string, string> = {
    flat_1u: "Flat 1u",
    edge_scaled_cap_1u: "Edge-scaled",
    fractional_kelly_25_cap_1u: "25% Kelly",
  };

  return (
    <section className="workspace workspace-grid research-grid">
      <div className="panel panel-span-2">
        <div className="panel-header">
          <div>
            <p className="panel-kicker">Training corpus</p>
            <h3>Historical data coverage</h3>
          </div>
          <span>{coverage.data ? `${coverage.data.total_games_with_prior.toLocaleString()} games with a market prior` : "-"}</span>
        </div>
        <div className="metric-grid">
          <MetricBlock label="Configured train start" value={coverage.data?.train_start_date ?? "-"} />
          <MetricBlock label="Last model trained" value={coverage.data?.latest_model_trained_at ? formatDateTime(coverage.data.latest_model_trained_at) : "not yet"} />
          <MetricBlock label="Last model span" value={coverage.data?.latest_model_train_start && coverage.data?.latest_model_train_end ? `${coverage.data.latest_model_train_start} → ${coverage.data.latest_model_train_end}` : "-"} />
          <MetricBlock label="Sources blended" value={String(coverage.data?.rows.length ?? 0)} />
        </div>
        <SimpleTable
          headers={["Source", "Span", "Games with prior"]}
          rows={(coverage.data?.rows ?? []).map((row) => [
            row.label,
            row.first_date && row.last_date ? `${row.first_date} → ${row.last_date}` : "-",
            row.games_with_prior.toLocaleString(),
          ])}
        />
      </div>

      <div className="panel chart-panel">
        <div className="panel-header">
          <div>
            <p className="panel-kicker">Leader board</p>
            <h3>Strategy stack</h3>
          </div>
          <span>{strategies.data?.length ?? 0} entries</span>
        </div>
        <BarPanel data={strategyChart} dataKey="roi" color="#00d0a3" suffix="%" />
      </div>

      <div className="panel chart-panel">
        <div className="panel-header">
          <div>
            <p className="panel-kicker">Model ladder</p>
            <h3>All contenders</h3>
          </div>
          <span>{roster.data?.filter((r) => r.role !== "ensemble").length ?? 0} models</span>
        </div>
        <BarPanel data={rosterChart} dataKey="score" color="#7c6bff" />
      </div>

      <div className="panel panel-span-2">
        <div className="panel-header">
          <div>
            <p className="panel-kicker">Model roster</p>
            <h3>Champion, challengers, ensemble</h3>
          </div>
          <span>{roster.data?.length ?? 0} entries</span>
        </div>
        <SimpleTable
          headers={["Model", "Role", "Family", "Trained", "Train span", "Settled bets", "Accuracy", "Log loss", "ROI"]}
          rows={(roster.data ?? []).map((row) => [
            compactLabel(row.model_name),
            <span className={`pill compact ${row.role === "champion" ? "ok" : row.role === "ensemble" ? "accent" : "neutral"}`}>{row.role}</span>,
            row.family,
            row.trained_at ? formatDateTime(row.trained_at) : "-",
            row.train_start_date && row.train_end_date ? `${row.train_start_date} → ${row.train_end_date}` : "-",
            String(row.settled_bets),
            row.accuracy != null ? fmtPct(row.accuracy) : "-",
            row.log_loss != null ? row.log_loss.toFixed(4) : "-",
            row.roi != null ? <span className={row.roi >= 0 ? "text-positive" : "text-negative"}>{fmtPct(row.roi)}</span> : "-",
          ])}
        />
      </div>

      <div className="panel panel-span-2">
        <div className="panel-header">
          <div>
            <p className="panel-kicker">Bet sizing</p>
            <h3>ROI by staking policy</h3>
          </div>
          <span>Best-ROI policy highlighted per row</span>
        </div>
        <SimpleTable
          headers={["Model", "Role", "Flat 1u", "Edge-scaled", "25% Kelly", "Best"]}
          rows={(sizing.data ?? []).map((row) => {
            const byKey: Record<string, typeof row.policies[number] | undefined> = {};
            for (const p of row.policies) byKey[p.policy] = p;
            const cell = (key: string) => {
              const p = byKey[key];
              if (!p || p.bets === 0) return <span className="text-neutral">-</span>;
              const cls = p.is_best
                ? "text-positive pill ok compact"
                : p.roi >= 0
                ? "text-positive"
                : "text-negative";
              return (
                <span className={cls}>
                  {fmtPct(p.roi)} <small>({p.bets} bets)</small>
                </span>
              );
            };
            return [
              compactLabel(row.model_name),
              <span className={`pill compact ${row.role === "champion" ? "ok" : row.role === "ensemble" ? "accent" : "neutral"}`}>{row.role}</span>,
              cell("flat_1u"),
              cell("edge_scaled_cap_1u"),
              cell("fractional_kelly_25_cap_1u"),
              row.best_policy ? <strong>{sizingPolicyLabels[row.best_policy] ?? row.best_policy}</strong> : <span className="text-neutral">-</span>,
            ];
          })}
        />
      </div>

      <div className="panel panel-span-2">
        <div className="panel-header">
          <div>
            <p className="panel-kicker">Top rows</p>
            <h3>Research blotter</h3>
          </div>
          <span>ROI and guardrails first</span>
        </div>
        <SimpleTable
          headers={["Strategy", "Family", "Bets", "ROI", "CI", "Pos slices", "Guardrails"]}
          rows={(strategies.data ?? []).slice(0, 12).map((row) => [
            row.strategy_name,
            row.family ?? "-",
            String(row.bets),
            <span className={row.roi >= 0 ? "text-positive" : "text-negative"}>{fmtPct(row.roi)}</span>,
            row.roi_ci_lower != null && row.roi_ci_upper != null ? `${fmtPct(row.roi_ci_lower)} to ${fmtPct(row.roi_ci_upper)}` : "-",
            row.positive_slice_rate != null ? fmtPct(row.positive_slice_rate) : "-",
            row.guardrails_passed ? <span className="pill ok compact">Pass</span> : <span className="pill warn compact">Fail</span>,
          ])}
        />
      </div>

      <div className="panel">
        <div className="panel-header">
          <div>
            <p className="panel-kicker">Calibration</p>
            <h3>Prediction fidelity</h3>
          </div>
          <span>{calibration.data?.length ?? 0} buckets</span>
        </div>
        <SimpleTable
          headers={["Model", "Bucket", "Pred", "Actual", "Error"]}
          rows={(calibration.data ?? []).slice(0, 10).map((row) => [
            compactLabel(row.model_name),
            row.bucket,
            fmtPct(row.avg_predicted_prob),
            fmtPct(row.actual_home_win_rate),
            fmtPct(row.abs_error),
          ])}
        />
      </div>

      <div className="panel">
        <div className="panel-header">
          <div>
            <p className="panel-kicker">Feature tape</p>
            <h3>Importance ladder</h3>
          </div>
          <span>{featureImportance.data?.length ?? 0} rows</span>
        </div>
        <SimpleTable
          headers={["Model", "Rank", "Feature", "Importance"]}
          rows={(featureImportance.data ?? []).slice(0, 10).map((row) => [
            compactLabel(row.model_name),
            String(row.rank),
            row.feature,
            fmtMetric(row.importance),
          ])}
        />
      </div>
    </section>
  );
}

export function OpsPage() {
  const queryClient = useQueryClient();
  const jobs = useQuery({ queryKey: ["jobs"], queryFn: api.jobs, refetchInterval: 10000 });
  const freshness = useQuery({ queryKey: ["freshness"], queryFn: api.freshness, refetchInterval: 10000 });
  const importStatus = useQuery({ queryKey: ["import-status"], queryFn: api.importStatus, refetchInterval: 30000 });
  const [selectedJob, setSelectedJob] = useState<string | null>(null);
  const jobDetail = useQuery({
    queryKey: ["job", selectedJob],
    queryFn: () => api.job(selectedJob!),
    enabled: selectedJob !== null,
    refetchInterval: selectedJob ? 5000 : false,
  });

  const launch = useMutation({
    mutationFn: api.launchJob,
    onSuccess: async (payload) => {
      setSelectedJob(payload.job.id);
      await queryClient.invalidateQueries({ queryKey: ["jobs"] });
      await queryClient.invalidateQueries({ queryKey: ["job", payload.job.id] });
      await queryClient.invalidateQueries({ queryKey: ["freshness"] });
    },
  });

  const buttons = useMemo(
    () => [
      { key: "collect-once", label: "Collect snapshot", desc: "Pull a fresh market and model tape." },
      { key: "sync-results", label: "Sync results", desc: "Backfill latest completed MLB finals." },
      { key: "train-game-model", label: "Train model", desc: "Refresh the primary prediction stack." },
    ],
    [],
  );

  return (
    <section className="workspace workspace-grid ops-grid">
      <div className="panel">
        <div className="panel-header">
          <div>
            <p className="panel-kicker">Execution rack</p>
            <h3>Operator actions</h3>
          </div>
          <span>{launch.isPending ? "Submitting..." : "Ready"}</span>
        </div>
        <div className="button-list">
          {buttons.map((button) => (
            <button key={button.key} className="terminal-button command-button" onClick={() => launch.mutate(button.key)} disabled={launch.isPending}>
              <strong>{button.label}</strong>
              <span>{button.desc}</span>
            </button>
          ))}
        </div>
      </div>

      <div className="panel">
        <div className="panel-header">
          <div>
            <p className="panel-kicker">Feed health</p>
            <h3>Freshness board</h3>
          </div>
          <span>{freshness.data?.stale_data ? "Attention" : "Healthy"}</span>
        </div>
        <div className="metric-grid">
          <MetricBlock label="Snapshot" value={freshness.data?.latest_snapshot_ts ? formatDateTime(freshness.data.latest_snapshot_ts) : "-"} />
          <MetricBlock label="Games" value={freshness.data?.latest_game_date ?? "-"} />
          <MetricBlock label="Weather" value={freshness.data?.latest_weather_game_date ?? "-"} />
          <MetricBlock label="Imports" value={freshness.data?.latest_historical_import_completed_at ? formatDateTime(freshness.data.latest_historical_import_completed_at) : "-"} />
        </div>
      </div>

      <div className="panel panel-span-2">
        <div className="panel-header">
          <div>
            <p className="panel-kicker">Runtime tape</p>
            <h3>Job monitor</h3>
          </div>
          <span>{jobs.data?.length ?? 0} tracked jobs</span>
        </div>
        <SimpleTable
          headers={["Job", "State", "Started", "Duration", "Exit"]}
          rows={(jobs.data ?? []).map((row) => [
            <button key={row.id} className="table-link" onClick={() => setSelectedJob(row.id)}>
              {row.label}
            </button>,
            <span className={row.status === "success" ? "text-positive" : row.status === "failed" ? "text-negative" : "text-neutral"}>{row.status}</span>,
            formatDateTime(row.started_at),
            `${Math.round(row.duration_seconds)}s`,
            row.returncode == null ? "-" : String(row.returncode),
          ])}
        />
      </div>

      <div className="panel">
        <div className="panel-header">
          <div>
            <p className="panel-kicker">Import matrix</p>
            <h3>Historical coverage</h3>
          </div>
          <span>{importStatus.data?.length ?? 0} sources</span>
        </div>
        <SimpleTable
          headers={["Source", "Runs", "Rows", "Pregame", "Last complete"]}
          rows={(importStatus.data ?? []).map((row) => [
            row.source,
            String(row.import_runs),
            String(row.normalized_rows ?? "-"),
            String(row.games_with_pregame_quotes ?? "-"),
            row.last_completed_at ? formatDateTime(row.last_completed_at) : "-",
          ])}
        />
      </div>

      <div className="panel">
        <div className="panel-header">
          <div>
            <p className="panel-kicker">Selected log</p>
            <h3>Execution output</h3>
          </div>
          <span>{selectedJob ?? "No job selected"}</span>
        </div>
        <pre className="log-tail">{jobDetail.data?.log_tail ?? "Select a job to inspect execution output."}</pre>
      </div>
    </section>
  );
}

function OpportunityBlotter({
  rows,
  selectedOpportunityKey,
  onSelect,
}: {
  rows: Opportunity[];
  selectedOpportunityKey: string | null;
  onSelect: (opportunityKey: string) => void;
}) {
  return (
    <div className="blotter-wrap">
      <table className="blotter-table">
        <thead>
          <tr>
            {["#", "Signal", "Matchup", "Source", "Model", "Edge", "Mkt", "Fair", "Start"].map((header, index) => (
              <th key={`${header}-${index}`}>{header}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, index) => {
            const rowKey = opportunityKey(row);
            const selected = rowKey === selectedOpportunityKey;
            return (
              <tr key={`${rowKey}-${index}`} className={selected ? "selected-row" : undefined} onClick={() => onSelect(rowKey)}>
                <td className="mono">{index + 1}</td>
                <td><span className="signal-pill">LONG</span></td>
                <td>
                  <div className="matchup-cell">
                    <strong>{row.team}</strong>
                    <span>vs {row.opponent_team ?? "-"}</span>
                  </div>
                </td>
                <td>{row.source ?? "-"}</td>
                <td>{compactLabel(row.model_name)}</td>
                <td className="mono text-positive">{row.edge_bps}</td>
                <td className="mono">{fmtPct(row.market_prob)}</td>
                <td className="mono">{fmtPct(row.model_prob)}</td>
                <td className="mono">{row.event_start_time ? formatShortTime(row.event_start_time) : "-"}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function opportunityKey(row: Opportunity): string {
  return [row.game_id, row.team, row.model_name, row.source ?? "", row.market_id ?? ""].join("|");
}

function BarPanel({
  data,
  dataKey,
  color,
  suffix = "",
}: {
  data: Array<{ name: string; [key: string]: string | number }>;
  dataKey: string;
  color: string;
  suffix?: string;
}) {
  if (data.length === 0) {
    return <p className="empty-copy">No data available.</p>;
  }

  return (
    <div className="chart-shell">
      <ResponsiveContainer width="100%" height={220}>
        <BarChart data={data} layout="vertical" margin={{ left: 12, right: 12, top: 8, bottom: 8 }}>
          <CartesianGrid stroke="rgba(255,255,255,0.06)" horizontal={false} />
          <XAxis type="number" stroke="#6d83a8" tickLine={false} axisLine={false} />
          <YAxis type="category" dataKey="name" stroke="#9eb2d2" width={88} tickLine={false} axisLine={false} />
          <Tooltip contentStyle={{ background: "#09101b", border: "1px solid rgba(255,255,255,0.12)" }} formatter={(value: number) => `${value}${suffix}`} />
          <Bar dataKey={dataKey} radius={[0, 8, 8, 0]} fill={color} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

function GapHistoryChart({ data }: { data: Array<{ snapshot_ts: string | null; gap_bps: number; team?: string }> }) {
  if (data.length === 0) {
    return <p className="empty-copy">No gap history for the selected market.</p>;
  }
  const chartData = data
    .slice(-24)
    .map((row) => ({ time: row.snapshot_ts ? new Date(row.snapshot_ts).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" }) : "-", gap: row.gap_bps }));

  return (
    <div className="chart-shell">
      <ResponsiveContainer width="100%" height={220}>
        <AreaChart data={chartData} margin={{ left: 8, right: 8, top: 8, bottom: 8 }}>
          <defs>
            <linearGradient id="gapFill" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#00d0a3" stopOpacity={0.45} />
              <stop offset="95%" stopColor="#00d0a3" stopOpacity={0.03} />
            </linearGradient>
          </defs>
          <CartesianGrid stroke="rgba(255,255,255,0.06)" vertical={false} />
          <XAxis dataKey="time" stroke="#6d83a8" tickLine={false} axisLine={false} />
          <YAxis stroke="#6d83a8" tickLine={false} axisLine={false} />
          <Tooltip contentStyle={{ background: "#09101b", border: "1px solid rgba(255,255,255,0.12)" }} />
          <Area type="monotone" dataKey="gap" stroke="#00d0a3" fill="url(#gapFill)" strokeWidth={2} />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}

function SimpleTable({ headers, rows }: { headers: string[]; rows: Array<Array<ReactNode>> }) {
  return (
    <div className="table-wrap">
      <table className="terminal-table">
        <thead>
          <tr>
            {headers.map((header, index) => (
              <th key={`${header}-${index}`}>{header}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, index) => (
            <tr key={index}>
              {row.map((cell, cellIndex) => (
                <td key={cellIndex}>{cell}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function TapeStat({ label, value, tone = "neutral" }: { label: string; value: string; tone?: "neutral" | "ok" | "warn" | "accent" }) {
  return (
    <div className={`tape-stat ${tone}`}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function MetricBlock({ label, value }: { label: string; value: string }) {
  return (
    <div className="metric-block">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function DetailRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="detail-row">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function aggregateCounts(rows: Opportunity[], keyFn: (row: Opportunity) => string) {
  const counts = new Map<string, number>();
  for (const row of rows) {
    const key = compactLabel(keyFn(row));
    counts.set(key, (counts.get(key) ?? 0) + 1);
  }
  return Array.from(counts.entries())
    .map(([name, count]) => ({ name, count }))
    .sort((left, right) => right.count - left.count)
    .slice(0, 8);
}

function compactLabel(value: string) {
  return value
    .replace("mlb_win_", "")
    .replace("_v1", "")
    .replace("_v2", "")
    .replace("_rbf", "")
    .replace(/_/g, " ");
}

function formatDateTime(value: string) {
  return new Date(value).toLocaleString([], { month: "numeric", day: "numeric", hour: "numeric", minute: "2-digit" });
}

function formatShortTime(value: string) {
  return new Date(value).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
}

function fmtPct(value: number | null | undefined) {
  if (value == null || Number.isNaN(value)) return "-";
  return `${(value * 100).toFixed(1)}%`;
}

function fmtMetric(value: number | null | undefined) {
  if (value == null || Number.isNaN(value)) return "-";
  return value.toFixed(4);
}
