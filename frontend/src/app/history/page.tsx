"use client";

import { useCallback, useEffect, useState } from "react";
import { FallRiskAlert } from "@/components/FallRiskAlert";
import Sidebar from "@/components/Sidebar";
import { buildGaitReport } from "@/lib/gaitAnalysis";
import { generateMockSensorData, MockProfile, PROFILE_LABELS } from "@/lib/mockSensorData";
import type { GaitReport } from "@/types/sensor";

const PROFILES: MockProfile[] = ["normal", "parkinsons", "stroke", "fall_risk"];
const STORAGE_KEY = "shoealls_gait_history";

interface HistoryEntry {
  id: string;
  timestamp: string;
  profile: MockProfile;
  report: GaitReport;
}

function loadHistory(): HistoryEntry[] {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY) ?? "[]") as HistoryEntry[];
  } catch {
    return [];
  }
}

export default function HistoryPage() {
  const [profile, setProfile] = useState<MockProfile>("normal");
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [selected, setSelected] = useState<HistoryEntry | null>(null);

  useEffect(() => {
    setHistory(loadHistory());
  }, []);

  const runAndSave = useCallback(() => {
    const mock = generateMockSensorData(profile);
    const entry: HistoryEntry = {
      id: crypto.randomUUID(),
      timestamp: new Date().toLocaleString("ko-KR"),
      profile,
      report: buildGaitReport(mock.features, mock.frames),
    };
    const next = [entry, ...history].slice(0, 20);
    setHistory(next);
    setSelected(entry);
    localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
  }, [history, profile]);

  return (
    <div className="flex h-screen bg-bg overflow-hidden">
      <Sidebar />
      <main className="flex-1 flex flex-col overflow-hidden">
        <header className="bg-surface border-b border-border px-8 py-4 flex items-center justify-between shrink-0">
          <h1 className="text-textPri font-semibold text-xl">분석 이력</h1>
          <div className="flex items-center gap-3">
            <select value={profile} onChange={(event) => setProfile(event.target.value as MockProfile)} className="bg-card border border-border text-textSec text-[13px] rounded-lg px-3 py-1.5">
              {PROFILES.map((item) => <option key={item} value={item}>{PROFILE_LABELS[item]}</option>)}
            </select>
            <button onClick={runAndSave} className="bg-blue hover:bg-blue/80 text-white font-semibold text-[13px] px-5 py-2 rounded-lg">
              분석 저장
            </button>
          </div>
        </header>
        <div className="flex-1 flex overflow-hidden">
          <aside className="w-80 shrink-0 border-r border-border overflow-y-auto p-4 space-y-2">
            {history.length === 0 && <div className="text-center py-12 text-textMuted text-[13px]">저장된 분석 이력이 없습니다.</div>}
            {history.map((entry) => (
              <button key={entry.id} onClick={() => setSelected(entry)} className="w-full text-left bg-card hover:border-blue/40 border border-border rounded-xl p-3">
                <div className="flex items-center justify-between">
                  <span className="text-textPri font-semibold text-[13px]">{PROFILE_LABELS[entry.profile]}</span>
                  <span className="text-amber text-[12px]">{Math.round(entry.report.fallRisk.score * 100)}%</span>
                </div>
                <div className="text-textMuted text-[11px] mt-1">{entry.timestamp}</div>
              </button>
            ))}
          </aside>
          <section className="flex-1 overflow-y-auto p-6">
            {selected ? <FallRiskAlert risk={selected.report.fallRisk} /> : <div className="h-full flex items-center justify-center text-textMuted text-[14px]">왼쪽 목록에서 분석을 선택하세요.</div>}
          </section>
        </div>
      </main>
    </div>
  );
}
