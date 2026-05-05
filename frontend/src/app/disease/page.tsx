"use client";

import { useState, useCallback } from "react";
import Sidebar from "@/components/Sidebar";
import { ResultCard, ProgressBar } from "@/components/ResultCard";
import { api, SampleResponse, DiseaseRiskResponse } from "@/lib/api";

const PROFILES = ["normal", "parkinsons", "stroke", "fall_risk"] as const;
type Profile = (typeof PROFILES)[number];

const PROFILE_KR: Record<Profile, string> = {
  normal: "정상 보행",
  parkinsons: "파킨슨",
  stroke: "뇌졸중",
  fall_risk: "낙상 위험",
};

const C = {
  blue:   "#3B82F6",
  green:  "#10B981",
  amber:  "#F59E0B",
  red:    "#EF4444",
  purple: "#AF65FA",
};

function riskColor(prob: number) {
  if (prob > 0.5) return C.red;
  if (prob > 0.25) return C.amber;
  return C.green;
}

export default function DiseasePage() {
  const [profile, setProfile] = useState<Profile>("parkinsons");
  const [sample, setSample] = useState<SampleResponse | null>(null);
  const [result, setResult] = useState<DiseaseRiskResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const run = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const s = await api.sample(profile);
      setSample(s);
      const r = await api.analyze(s.sensor_data, s.features);
      setResult(r.disease_risk);
    } catch (e) {
      setError(e instanceof Error ? e.message : "분석 실패");
    } finally {
      setLoading(false);
    }
  }, [profile]);

  return (
    <div className="flex h-screen bg-bg overflow-hidden">
      <Sidebar />
      <main className="flex-1 flex flex-col overflow-hidden">
        <header className="bg-surface border-b border-border px-8 py-4 flex items-center justify-between shrink-0">
          <h1 className="text-textPri font-semibold text-xl">질환 위험 예측</h1>
          <div className="flex items-center gap-3">
            <select
              value={profile}
              onChange={(e) => setProfile(e.target.value as Profile)}
              className="bg-card border border-border text-textSec text-[13px] rounded-lg px-3 py-1.5 focus:outline-none focus:border-blue"
            >
              {PROFILES.map((p) => <option key={p} value={p}>{PROFILE_KR[p]}</option>)}
            </select>
            <button
              onClick={run}
              disabled={loading}
              className="bg-blue hover:bg-blue/80 disabled:opacity-50 text-white font-semibold text-[13px] px-5 py-2 rounded-lg transition-colors"
            >
              {loading ? "분석 중…" : "위험도 분석"}
            </button>
          </div>
        </header>

        <div className="flex-1 overflow-y-auto p-8 space-y-6">
          {error && (
            <div className="bg-red/10 border border-red/30 text-red rounded-xl px-5 py-4 text-[14px]">
              오류: {error}
            </div>
          )}

          {loading && (
            <div className="bg-card rounded-2xl p-12 text-center">
              <div className="inline-block w-8 h-8 border-2 border-blue border-t-transparent rounded-full animate-spin mb-3" />
              <div className="text-textSec text-[14px]">질환 위험도 예측 중…</div>
            </div>
          )}

          {result && (
            <>
              {/* ML 예측 요약 */}
              <div className="grid grid-cols-3 gap-4">
                <div className="bg-card rounded-xl p-4 col-span-1">
                  <div className="text-textSec text-[12px] mb-1">ML 예측 질환</div>
                  <div className="text-textPri font-bold text-[18px]">{result.ml_prediction_kr}</div>
                  <div className="text-green text-[13px] mt-1">{(result.ml_confidence * 100).toFixed(1)}% 신뢰도</div>
                </div>
                <div className="bg-card rounded-xl p-4 col-span-2">
                  <div className="text-textSec text-[12px] mb-2">상위 3개 질환 위험도</div>
                  <div className="space-y-2">
                    {result.ml_top3.map((d) => (
                      <ProgressBar
                        key={d.name_kr}
                        pct={d.probability}
                        color={riskColor(d.probability)}
                        label={d.name_kr}
                        valueLabel={(d.probability * 100).toFixed(1) + "%"}
                      />
                    ))}
                  </div>
                </div>
              </div>

              {/* 이상 바이오마커 */}
              {result.abnormal_biomarkers && result.abnormal_biomarkers.length > 0 && (
                <ResultCard title="이상 감지 바이오마커" badge="바이오마커" accentColor={C.amber}>
                  <div className="flex flex-wrap gap-2">
                    {result.abnormal_biomarkers.map((b) => (
                      <span
                        key={b}
                        className="px-2.5 py-1 rounded-lg text-[12px] font-medium"
                        style={{ color: C.amber, background: `${C.amber}20` }}
                      >
                        {b}
                      </span>
                    ))}
                  </div>
                </ResultCard>
              )}

              {/* 상세 질환 목록 */}
              {result.top_diseases && result.top_diseases.length > 0 && (
                <ResultCard title="질환별 상세 위험도" badge="상세 분석" accentColor={C.green}>
                  <div className="space-y-4">
                    {result.top_diseases.map((d) => (
                      <div key={d.disease} className="border-b border-border/40 pb-4 last:border-0 last:pb-0">
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-textPri font-semibold text-[14px]">{d.disease_kr}</span>
                          <span
                            className="px-2 py-0.5 rounded-full text-[11px] font-medium"
                            style={{ color: riskColor(d.risk_score), background: `${riskColor(d.risk_score)}20` }}
                          >
                            {d.severity}
                          </span>
                        </div>
                        <ProgressBar
                          pct={d.risk_score}
                          color={riskColor(d.risk_score)}
                          label=""
                          valueLabel={(d.risk_score * 100).toFixed(1) + "%"}
                        />
                        {d.key_signs && d.key_signs.length > 0 && (
                          <div className="text-textMuted text-[11px] mt-1">
                            주요 징후: {d.key_signs.slice(0, 3).join(" · ")}
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </ResultCard>
              )}
            </>
          )}

          {!result && !loading && (
            <div className="bg-card rounded-2xl p-12 text-center">
              <div className="text-textMuted text-[14px]">
                프로파일을 선택하고 <strong className="text-textSec">위험도 분석</strong>을 눌러 질환 위험을 예측하세요.
              </div>
              <div className="mt-3 text-textMuted text-[12px]">11개 질환 클래스 · 45개 바이오마커 분석</div>
            </div>
          )}
        </div>

        <footer className="bg-surface border-t border-border px-8 py-2.5 text-textMuted text-[11px] flex justify-between shrink-0">
          <span>POST /api/v1/analyze · disease_risk 섹션</span>
          <span>© 2026 Shoealls</span>
        </footer>
      </main>
    </div>
  );
}
