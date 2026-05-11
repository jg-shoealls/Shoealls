"use client";

import { useEffect, useRef, useState } from "react";
import { LineChart, Line, YAxis, ResponsiveContainer, ReferenceLine } from "recharts";
import type { MockProfile } from "@/lib/mockSensorData";

const MAX_POINTS = 80;

interface WavePoint { t: number; az: number; gx: number }

const PROFILE_CONFIG: Record<MockProfile, { amp: number; tremorHz: number; noiseScale: number; label: string; color: string }> = {
  normal:     { amp: 1.0,  tremorHz: 0,    noiseScale: 0.08, label: "정상 보행 신호",          color: "#10B981" },
  parkinsons: { amp: 0.42, tremorHz: 4.5,  noiseScale: 0.14, label: "미세 떨림 감지 (4-6 Hz)", color: "#EF4444" },
  stroke:     { amp: 0.65, tremorHz: 0,    noiseScale: 0.22, label: "불규칙 패턴 감지",         color: "#F59E0B" },
  fall_risk:  { amp: 0.70, tremorHz: 1.8,  noiseScale: 0.18, label: "불안정 신호 감지",         color: "#F59E0B" },
};

function nextPoint(t: number, profile: MockProfile): WavePoint {
  const { amp, tremorHz, noiseScale } = PROFILE_CONFIG[profile];
  const step = t * 0.26;
  const tremor = tremorHz > 0 ? Math.sin(t * tremorHz * 0.18) * 0.18 : 0;
  const noise = (Math.random() - 0.5) * noiseScale;
  return {
    t,
    az: parseFloat((Math.sin(step) * amp + tremor + noise + 0.9).toFixed(3)),
    gx: parseFloat((Math.sin(step + 1.3) * amp * 0.38 + tremor * 0.4 + noise * 0.6).toFixed(3)),
  };
}

export default function RealtimeWaveform({ profile }: { profile: MockProfile }) {
  const [data, setData] = useState<WavePoint[]>(() =>
    Array.from({ length: MAX_POINTS }, (_, i) => nextPoint(i, profile))
  );
  const tRef = useRef(MAX_POINTS);
  const profileRef = useRef(profile);

  useEffect(() => { profileRef.current = profile; }, [profile]);

  useEffect(() => {
    const id = setInterval(() => {
      tRef.current += 1;
      const pt = nextPoint(tRef.current, profileRef.current);
      setData((prev) => [...prev.slice(1), pt]);
    }, 80);
    return () => clearInterval(id);
  }, []);

  const cfg = PROFILE_CONFIG[profile];

  return (
    <div className="bg-card rounded-xl border border-border p-4">
      <div className="flex items-center justify-between mb-3">
        <div>
          <span className="text-textPri font-semibold text-[13px]">IMU 실시간 가속도 신호</span>
          <span className="ml-2 text-[11px] text-textMuted">ax/gz · 30 Hz</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full animate-pulse" style={{ background: cfg.color }} />
          <span className="text-[11px] font-medium" style={{ color: cfg.color }}>{cfg.label}</span>
        </div>
      </div>
      <div className="h-[120px]">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data}>
            <YAxis domain={[0.2, 1.6]} hide />
            <ReferenceLine y={0.9} stroke="#32425B" strokeDasharray="4 4" />
            <Line
              type="monotone" dataKey="az" name="가속도 Z"
              stroke={cfg.color} strokeWidth={1.5} dot={false} isAnimationActive={false}
            />
            <Line
              type="monotone" dataKey="gx" name="각속도 X"
              stroke="#AF65FA" strokeWidth={1} dot={false} isAnimationActive={false} strokeOpacity={0.7}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <div className="flex gap-4 mt-2">
        <span className="flex items-center gap-1.5 text-[10px] text-textMuted">
          <span className="w-4 h-0.5 rounded" style={{ background: cfg.color }} /> 가속도 Z축
        </span>
        <span className="flex items-center gap-1.5 text-[10px] text-textMuted">
          <span className="w-4 h-0.5 rounded bg-purple" /> 각속도 X축
        </span>
      </div>
    </div>
  );
}
