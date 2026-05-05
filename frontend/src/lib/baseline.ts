export interface GaitSession {
  id: string;
  timestamp: string;
  profile: string;
  features: Record<string, number>;
  injuryRisk: number;
}

export interface BaselineStats {
  mean: Record<string, number>;
  std: Record<string, number>;
  sessions: number;
}

const BASELINE_KEY = "shoealls_baseline_sessions";
const MAX_SESSIONS = 10;

export function saveSession(session: GaitSession): void {
  try {
    const sessions = getSessions();
    const updated = [session, ...sessions].slice(0, MAX_SESSIONS);
    localStorage.setItem(BASELINE_KEY, JSON.stringify(updated));
  } catch {}
}

export function getSessions(): GaitSession[] {
  try {
    return JSON.parse(localStorage.getItem(BASELINE_KEY) ?? "[]");
  } catch {
    return [];
  }
}

export function computeBaseline(sessions: GaitSession[]): BaselineStats | null {
  if (sessions.length < 3) return null;
  const keys = Object.keys(sessions[0].features);
  const mean: Record<string, number> = {};
  const std: Record<string, number> = {};

  for (const key of keys) {
    const vals = sessions
      .map((s) => s.features[key])
      .filter((v) => typeof v === "number" && isFinite(v));
    if (vals.length === 0) continue;
    const m = vals.reduce((a, b) => a + b, 0) / vals.length;
    mean[key] = m;
    std[key] = Math.sqrt(vals.reduce((a, b) => a + (b - m) ** 2, 0) / vals.length) || 0.001;
  }

  return { mean, std, sessions: sessions.length };
}

export function deviationZ(
  value: number,
  baseline: BaselineStats,
  key: string
): number | null {
  const m = baseline.mean[key];
  const s = baseline.std[key];
  if (m == null || s == null) return null;
  return (value - m) / s;
}
