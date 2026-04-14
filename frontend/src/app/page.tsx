"use client";

import { useState, useCallback } from "react";
import Sidebar from "@/components/Sidebar";
import { ResultCard, ProgressBar } from "@/components/ResultCard";
import { api, AnalyzeResponse, SampleResponse } from "@/lib/api";

const PROFILES = ["normal", "parkinsons", "stroke", "fall_risk"] as const;
type Profile = (typeof PROFILES)[number];

const PROFILE_KR: Record<Profile, string> = {
  normal: "정상 보행",
  parkinsons: "파킨슨",
  stroke: "뇌졸중",
  fall_risk: "낙상 위험",
};

// ── 색상 팔레트 ──────────────────────────────────────────────────────
const C = {
  blue:   "#3B82F6",
  green:  "#10B981",
  amber:  "#F59E0B",
  red:    "#EF4444",
  purple: "#AF65FA",
};

export default function Dashboard() {
  const [profile, setProfile] = useState<Profile>("parkinsons");
  const [sample, setSample] = useState<SampleResponse | null>(null);
  const [result, setResult] = useState<AnalyzeResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const runAnalysis = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const s = await api.sample(profile);
      setSample(s);
      const r = await api.analyze(s.sensor_data, s.features);
      setResult(r);
    } catch (e) {
      setError(e instanceof Error ? e.message : "분석 실패");
    } finally {
      setLoading(false);
    }
  }, [profile]);

  const cls = result?.classification;
  const dis = result?.disease_risk;
  const inj = result?.injury_risk;
  const rsn = result?.reasoning;

  return (
    <div className="flex h-screen bg-bg overflow-hidden">
      <Sidebar active={0} />

      {/* ── 메인 ── */}
      <main className="flex-1 flex flex-col overflow-hidden">
        {/* 탑바 */}
        <header className="bg-surface border-b border-border px-8 py-4 flex items-center justify-between shrink-0">
          <h1 className="text-textPri font-semibold text-xl">보행 분석 대시보드</h1>
          <div className="flex items-center gap-3">
            {/* 프로파일 선택 */}
            <select
              value={profile}
              onChange={(e) => setProfile(e.target.value as Profile)}
              className="bg-card border border-border text-textSec text-[13px] rounded-lg px-3 py-1.5 focus:outline-none focus:border-blue"
            >
              {PROFILES.map((p) => (
                <option key={p} value={p}>
                  {PROFILE_KR[p]}
                </option>
              ))}
            </select>

            {cls?.is_demo_mode !== false && (
              <span className="px-2.5 py-0.5 rounded-full text-[11px] font-medium text-amber bg-amber/20">
                데모 모드
              </span>
            )}
            <span className="px-2.5 py-0.5 rounded-full text-[11px] font-medium text-blue bg-blue/20">
              v0.1.0 MVP
            </span>

            <button
              onClick={runAnalysis}
              disabled={loading}
              className="bg-blue hover:bg-blue/80 disabled:opacity-50 text-white font-semibold text-[13px] px-5 py-2 rounded-lg transition-colors"
            >
              {loading ? "분석 중…" : "분석 시작"}
            </button>
          </div>
        </header>

        {/* 스크롤 영역 */}
        <div className="flex-1 overflow-y-auto p-8 space-y-6">
          {/* 오류 */}
          {error && (
            <div className="bg-red/10 border border-red/30 text-red rounded-xl px-5 py-4 text-[14px]">
              오류: {error}
            </div>
          )}

          {/* 센서 입력 요약 */}
          <section>
            <h2 className="text-textSec text-[14px] font-semibold mb-3">센서 입력 데이터</h2>
            <div className="grid grid-cols-3 gap-4">
              {[
                { title: "IMU 센서", sub: "가속도 + 자이로스코프 6ch", val: "128 프레임", color: C.blue },
                { title: "족저압 센서", sub: "16 × 8 그리드", val: sample ? "로드됨" : "대기 중", color: C.green },
                { title: "스켈레톤", sub: "17 관절 × 3D 좌표", val: "128 프레임", color: C.purple },
              ].map((card) => (
                <div key={card.title} className="bg-card rounded-xl p-4 flex flex-col gap-1">
                  <div className="flex items-center gap-2">
                    <span className="w-2.5 h-2.5 rounded-full" style={{ background: card.color }} />
                    <span className="text-textPri font-semibold text-[14px]">{card.title}</span>
                  </div>
                  <span className="text-textSec text-[11px]">{card.sub}</span>
                  <span className="text-[13px] font-semibold mt-1" style={{ color: card.color }}>
                    {card.val}
                  </span>
                </div>
              ))}
            </div>
          </section>

          {/* 결과 없음 안내 */}
          {!result && !loading && (
            <div className="bg-card rounded-2xl p-12 text-center">
              <div className="text-textMuted text-[14px]">
                보행 프로파일을 선택하고 <strong className="text-textSec">분석 시작</strong>을 누르세요.
              </div>
            </div>
          )}

          {/* 로딩 */}
          {loading && (
            <div className="bg-card rounded-2xl p-12 text-center">
              <div className="inline-block w-8 h-8 border-2 border-blue border-t-transparent rounded-full animate-spin mb-3" />
              <div className="text-textSec text-[14px]">AI 분석 중…</div>
            </div>
          )}

          {/* 결과 카드 그리드 */}
          {result && (
            <>
              <h2 className="text-textSec text-[14px] font-semibold">분석 결과</h2>
              <div className="grid grid-cols-2 gap-5">
                {/* ── 보행 패턴 분류 ── */}
                <ResultCard
                  title="보행 패턴 분류"
                  badge="보행 분류"
                  accentColor={C.blue}
                  isDemo={cls?.is_demo_mode}
                >
                  <div className="flex items-start gap-5">
                    {/* 원형 수치 */}
                    <div className="w-28 h-28 shrink-0 rounded-full bg-surface flex flex-col items-center justify-center border-2 border-blue/40">
                      <span className="text-blue text-[11px] font-medium">{cls?.prediction_kr}</span>
                      <span className="text-textPri font-bold text-xl">
                        {cls ? (cls.confidence * 100).toFixed(1) + "%" : "--"}
                      </span>
                    </div>
                    {/* 확률 바 */}
                    <div className="flex-1 space-y-3">
                      {cls &&
                        Object.entries(cls.class_probabilities)
                          .sort((a, b) => b[1] - a[1])
                          .map(([name, prob]) => (
                            <ProgressBar
                              key={name}
                              pct={prob}
                              color={C.blue}
                              label={name}
                              valueLabel={(prob * 100).toFixed(1) + "%"}
                            />
                          ))}
                    </div>
                  </div>
                </ResultCard>

                {/* ── 질환 위험도 ── */}
                <ResultCard title="질환 위험 예측" badge="질환 위험도" accentColor={C.green}>
                  {dis && (
                    <>
                      <div className="text-green text-[13px] font-semibold mb-3">
                        ML 예측: {dis.ml_prediction_kr}{" "}
                        {(dis.ml_confidence * 100).toFixed(1)}%
                      </div>
                      <div className="space-y-3">
                        {dis.ml_top3.map((d) => (
                          <ProgressBar
                            key={d.disease}
                            pct={d.probability}
                            color={
                              d.probability > 0.5 ? C.red
                              : d.probability > 0.2 ? C.amber
                              : C.green
                            }
                            label={d.disease_kr}
                            valueLabel={(d.probability * 100).toFixed(1) + "%"}
                          />
                        ))}
                      </div>
                      {dis.anomaly_biomarkers.length > 0 && (
                        <div className="mt-3 text-textMuted text-[11px]">
                          이상 감지: {dis.anomaly_biomarkers.slice(0, 3).join(" / ")}
                        </div>
                      )}
                    </>
                  )}
                </ResultCard>

                {/* ── 부상 위험 ── */}
                <ResultCard title="부상 위험 예측" badge="부상 예측" accentColor={C.amber}>
                  {inj && (
                    <div className="flex gap-5">
                      {/* 종합 점수 */}
                      <div className="shrink-0">
                        <div className="text-amber font-bold text-4xl">
                          {(inj.overall_risk_score * 100).toFixed(1)}%
                        </div>
                        <div className="text-textSec text-[11px] mt-1">종합 위험도</div>
                        <span
                          className="mt-2 inline-block px-2.5 py-0.5 rounded-full text-[11px] font-medium"
                          style={{
                            color: inj.risk_level === "HIGH" ? C.red : C.amber,
                            background: `${inj.risk_level === "HIGH" ? C.red : C.amber}28`,
                          }}
                        >
                          {inj.risk_level_kr}
                        </span>
                      </div>
                      {/* 부상 바 */}
                      <div className="flex-1 space-y-3">
                        {inj.top_injuries.slice(0, 3).map((injury) => (
                          <ProgressBar
                            key={injury.injury}
                            pct={injury.probability}
                            color={injury.probability > 0.4 ? C.red : C.amber}
                            label={injury.injury_kr}
                            valueLabel={(injury.probability * 100).toFixed(1) + "%"}
                          />
                        ))}
                        <div className="text-textMuted text-[11px] mt-1">{inj.injury_timeline}</div>
                      </div>
                    </div>
                  )}
                </ResultCard>

                {/* ── Chain-of-Reasoning ── */}
                <ResultCard title="Chain-of-Reasoning" badge="AI 추론" accentColor={C.purple}>
                  {rsn && (
                    <>
                      <div className="space-y-2 mb-3">
                        {rsn.reasoning_trace.map((step, i) => (
                          <div key={step.step} className="flex items-start gap-3">
                            <div className="flex flex-col items-center mt-1">
                              <div
                                className="w-2 h-2 rounded-full"
                                style={{ background: i === rsn.reasoning_trace.length - 1 ? C.purple : "#32425B" }}
                              />
                              {i < rsn.reasoning_trace.length - 1 && (
                                <div className="w-0.5 h-5 bg-border mt-0.5" />
                              )}
                            </div>
                            <div className="flex-1 flex justify-between text-[12px]">
                              <span className="text-textMuted">{step.label}</span>
                              <span
                                className="font-semibold"
                                style={{ color: i === rsn.reasoning_trace.length - 1 ? C.purple : "#94A3B8" }}
                              >
                                {step.prediction_kr}
                              </span>
                              <span className="text-textSec">
                                {(step.confidence * 100).toFixed(1)}%
                              </span>
                            </div>
                          </div>
                        ))}
                      </div>
                      <div className="text-textMuted text-[11px]">
                        불확실성: {(rsn.uncertainty * 100).toFixed(1)}% &nbsp;·&nbsp; 근거 강도:{" "}
                        {(rsn.evidence_strength * 100).toFixed(1)}%
                        {cls?.is_demo_mode && " · 데모 모드"}
                      </div>
                    </>
                  )}
                </ResultCard>
              </div>
            </>
          )}
        </div>

        {/* 하단 바 */}
        <footer className="bg-surface border-t border-border px-8 py-2.5 text-textMuted text-[11px] flex justify-between shrink-0">
          <span>Shoealls Gait Analysis API · POST /api/v1/analyze · {cls?.is_demo_mode ? "데모 모드" : "모델 로드됨"}</span>
          <span>© 2026 Shoealls</span>
        </footer>
      </main>
    </div>
  );
}
