"use client";

import { useCallback, useEffect, useState } from "react";
import Sidebar from "@/components/Sidebar";
import { generateMockSensorData, MockProfile, PROFILE_LABELS } from "@/lib/mockSensorData";

const PROFILES: MockProfile[] = ["normal", "parkinsons", "stroke", "fall_risk"];

interface ReasoningStep {
  icon: string;
  title: string;
  subtitle: string;
  evidence: string[];
  confidence: number;
  color: string;
}

const STEPS_BY_PROFILE: Record<MockProfile, ReasoningStep[]> = {
  parkinsons: [
    {
      icon: "🔍", title: "이상 감지", subtitle: "Step 1 · Anomaly Detection",
      evidence: ["보행 속도 0.68 m/s — 정상 대비 44% 저하", "케이던스 144 spm — 빠른 소보 패턴 감지", "보폭 규칙성 54% — 불규칙 리듬 감지"],
      confidence: 92, color: "#EF4444",
    },
    {
      icon: "📊", title: "근거 수집", subtitle: "Step 2 · Cross-modal Evidence",
      evidence: ["IMU: 4–6 Hz 미세 떨림 신호 감지 (tremor)", "족저압: 전족부 과부하 — 발 끌림(shuffling) 패턴", "좌우 비대칭 지수 0.79 — 편측 감소 감지"],
      confidence: 89, color: "#F59E0B",
    },
    {
      icon: "🧠", title: "감별 진단", subtitle: "Step 3 · Differential Diagnosis",
      evidence: ["파킨슨 보행 패턴 일치도 87.3%", "뇌졸중 패턴 제외 — 편측 비대칭 미충족", "정상 노화성 보행 저하와 감별 — 떨림 주파수 이상"],
      confidence: 87, color: "#AF65FA",
    },
    {
      icon: "⚕️", title: "최종 판정", subtitle: "Step 4 · Calibrated Assessment",
      evidence: ["복합 위험도 HIGH · 신뢰도 87%", "파킨슨 전조 보행 징후 감지 — 의료기관 방문 권고", "6개월 추적 관찰 + 신경과 전문의 상담 권고"],
      confidence: 87, color: "#EF4444",
    },
  ],
  stroke: [
    {
      icon: "🔍", title: "이상 감지", subtitle: "Step 1 · Anomaly Detection",
      evidence: ["보행 속도 0.52 m/s — 정상 대비 57% 저하", "좌우 대칭 0.58 — 심각한 편측 비대칭", "CoP 흔들림 0.086 — 동적 불안정"],
      confidence: 95, color: "#EF4444",
    },
    {
      icon: "📊", title: "근거 수집", subtitle: "Step 2 · Cross-modal Evidence",
      evidence: ["족저압: 우측 하중 68% — 편측 체중 지지", "IMU: 발 들어올림 각도 좌우 31% 차이", "골반 회전 비대칭 패턴 감지"],
      confidence: 93, color: "#F59E0B",
    },
    {
      icon: "🧠", title: "감별 진단", subtitle: "Step 3 · Differential Diagnosis",
      evidence: ["뇌졸중 후 편마비 패턴 일치도 91.2%", "파킨슨 패턴 제외 — 편측 특이성 불일치", "정형외과적 원인 배제 — 족저압 분포 근거"],
      confidence: 91, color: "#AF65FA",
    },
    {
      icon: "⚕️", title: "최종 판정", subtitle: "Step 4 · Calibrated Assessment",
      evidence: ["복합 위험도 HIGH · 신뢰도 91%", "뇌혈관 후유증 보행 패턴 — 재활 평가 필요", "낙상 위험 매우 높음 — 즉각 물리치료 권고"],
      confidence: 91, color: "#EF4444",
    },
  ],
  fall_risk: [
    {
      icon: "🔍", title: "이상 감지", subtitle: "Step 1 · Anomaly Detection",
      evidence: ["CoP 흔들림 0.092 — 기준치 2.6배 초과", "보폭 규칙성 56% — 리듬 불안정", "보행 속도 0.72 m/s — 경계 수준"],
      confidence: 88, color: "#F59E0B",
    },
    {
      icon: "📊", title: "근거 수집", subtitle: "Step 2 · Cross-modal Evidence",
      evidence: ["족저압: 외측 하중 불균형 — 발목 불안정", "IMU: 가속도 변동성 높음 — 균형 반응 저하", "ML 지수 0.15 — 내외측 체중 이동 과도"],
      confidence: 84, color: "#F59E0B",
    },
    {
      icon: "🧠", title: "감별 진단", subtitle: "Step 3 · Differential Diagnosis",
      evidence: ["근력 저하 기반 낙상 위험 패턴 일치", "전정기관 장애 동반 가능성 — CoP 패턴", "발목 인대 불안정 vs. 중추성 균형 장애 감별 필요"],
      confidence: 82, color: "#AF65FA",
    },
    {
      icon: "⚕️", title: "최종 판정", subtitle: "Step 4 · Calibrated Assessment",
      evidence: ["낙상 위험도 ELEVATED · 신뢰도 82%", "낙상 예방 훈련 + 발목 강화 운동 권고", "가정 내 낙상 위험 환경 점검 필요"],
      confidence: 82, color: "#F59E0B",
    },
  ],
  normal: [
    {
      icon: "🔍", title: "이상 감지", subtitle: "Step 1 · Anomaly Detection",
      evidence: ["보행 속도 1.22 m/s — 정상 범위", "케이던스 112 spm — 적정 범위", "보폭 규칙성 88% — 양호"],
      confidence: 96, color: "#10B981",
    },
    {
      icon: "📊", title: "근거 수집", subtitle: "Step 2 · Cross-modal Evidence",
      evidence: ["IMU: 규칙적 사인파 패턴 — 정상 보행 사이클", "족저압: 발뒤꿈치-발가락 균형 전이 정상", "좌우 대칭 93% — 이상 없음"],
      confidence: 97, color: "#10B981",
    },
    {
      icon: "🧠", title: "감별 진단", subtitle: "Step 3 · Differential Diagnosis",
      evidence: ["신경계 이상 패턴 미감지", "근골격계 부하 불균형 없음", "모든 바이오마커 정상 범위 내"],
      confidence: 98, color: "#10B981",
    },
    {
      icon: "⚕️", title: "최종 판정", subtitle: "Step 4 · Calibrated Assessment",
      evidence: ["위험도 LOW · 신뢰도 97%", "정상 보행 패턴 유지", "정기 모니터링 권고 (3개월 주기)"],
      confidence: 97, color: "#10B981",
    },
  ],
};

export default function ReasoningPage() {
  const [profile, setProfile] = useState<MockProfile>("parkinsons");
  const [visibleSteps, setVisibleSteps] = useState<number>(0);
  const [running, setRunning] = useState(false);

  const run = useCallback(async () => {
    setRunning(true);
    setVisibleSteps(0);
    generateMockSensorData(profile);
    for (let i = 1; i <= 4; i++) {
      await new Promise((res) => setTimeout(res, 900));
      setVisibleSteps(i);
    }
    setRunning(false);
  }, [profile]);

  // Auto-run on mount
  useEffect(() => { run(); }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const steps = STEPS_BY_PROFILE[profile];
  const verdict = steps[3];
  const isHighRisk = profile !== "normal";

  return (
    <div className="flex h-screen bg-bg overflow-hidden">
      <Sidebar />
      <main className="flex-1 flex flex-col overflow-hidden">
        <header className="bg-surface border-b border-border px-8 py-4 flex items-center justify-between shrink-0">
          <div>
            <h1 className="text-textPri font-semibold text-xl">Chain-of-Reasoning AI 추론</h1>
            <p className="text-textMuted text-[12px] mt-0.5">
              4단계 추론 체인 · 이상 감지 → 근거 수집 → 감별 진단 → 신뢰도 보정
            </p>
          </div>
          <div className="flex items-center gap-3">
            <select
              value={profile}
              onChange={(e) => setProfile(e.target.value as MockProfile)}
              className="bg-card border border-border text-textSec text-[13px] rounded-lg px-3 py-1.5"
            >
              {PROFILES.map((p) => <option key={p} value={p}>{PROFILE_LABELS[p]}</option>)}
            </select>
            <button
              onClick={run}
              disabled={running}
              className="bg-purple hover:bg-purple/80 disabled:opacity-50 text-white font-semibold text-[13px] px-5 py-2 rounded-lg transition-colors"
            >
              {running ? "추론 중…" : "추론 재실행"}
            </button>
          </div>
        </header>

        <div className="flex-1 overflow-y-auto p-8">
          <div className="max-w-3xl mx-auto space-y-4">
            {steps.map((step, idx) => {
              const visible = visibleSteps > idx;
              return (
                <div
                  key={idx}
                  className="rounded-xl border overflow-hidden transition-all duration-500"
                  style={{
                    borderColor: visible ? `${step.color}40` : "#32425B",
                    opacity: visible ? 1 : 0.25,
                    transform: visible ? "translateY(0)" : "translateY(12px)",
                  }}
                >
                  <div
                    className="px-5 py-3 flex items-center justify-between"
                    style={{ background: visible ? `${step.color}18` : "transparent" }}
                  >
                    <div className="flex items-center gap-3">
                      <span className="text-2xl">{step.icon}</span>
                      <div>
                        <span className="text-textPri font-semibold text-[14px]">{step.title}</span>
                        <span className="ml-2 text-textMuted text-[11px]">{step.subtitle}</span>
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      {visible && running && idx === visibleSteps - 1 && (
                        <span className="w-2 h-2 rounded-full animate-pulse" style={{ background: step.color }} />
                      )}
                      <span
                        className="px-3 py-0.5 rounded-full text-[12px] font-bold"
                        style={{ background: `${step.color}25`, color: step.color }}
                      >
                        {step.confidence}%
                      </span>
                    </div>
                  </div>
                  <div className="px-5 py-3 bg-card space-y-2">
                    {step.evidence.map((e, i) => (
                      <div key={i} className="flex items-start gap-2">
                        <span className="text-textMuted text-[12px] mt-0.5">·</span>
                        <span className="text-textSec text-[13px]">{e}</span>
                      </div>
                    ))}
                  </div>
                </div>
              );
            })}

            {/* Final verdict banner */}
            {visibleSteps >= 4 && (
              <div
                className="rounded-xl p-5 mt-2 border transition-all duration-700"
                style={{
                  borderColor: verdict.color,
                  background: isHighRisk
                    ? `linear-gradient(135deg, ${verdict.color}22, ${verdict.color}08)`
                    : "rgba(16,185,129,0.08)",
                }}
              >
                <div className="flex items-center justify-between">
                  <div>
                    <div className="text-textPri font-bold text-[16px]">
                      {isHighRisk ? "⚠️ 보행 이상 징후 감지" : "✅ 정상 보행 패턴"}
                    </div>
                    <div className="text-textSec text-[13px] mt-1">
                      {verdict.evidence[1]}
                    </div>
                    <div className="text-textMuted text-[11px] mt-2">
                      본 결과는 보행 이상 징후 참고용이며, 의학적 진단이 아닙니다.
                    </div>
                  </div>
                  <div
                    className="text-[28px] font-black px-5 py-3 rounded-xl"
                    style={{ background: `${verdict.color}20`, color: verdict.color }}
                  >
                    {verdict.confidence}%
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}
