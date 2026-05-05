"use client";

import { AnalyzeResponse } from "@/lib/api";

interface Props {
  result: AnalyzeResponse | null;
  profile: string;
}

function buildReport(result: AnalyzeResponse, profile: string): string {
  const timestamp = new Date().toLocaleString("ko-KR");

  const cls = result.classify;
  const dis = result.disease_risk;
  const inj = result.injury_risk;
  const rsn = result.reasoning;

  // Severity color helper (inline for report)
  const riskColor = (score: number): string => {
    if (score < 0.3) return "#10B981";
    if (score < 0.6) return "#F59E0B";
    return "#EF4444";
  };

  // Disease risk rows
  const diseaseRows = dis.top_diseases
    .map((d) => {
      const pct = (d.risk_score * 100).toFixed(1);
      const color = riskColor(d.risk_score);
      return `
        <div class="risk-row">
          <div class="risk-label">
            <span class="risk-name">${d.disease_kr}</span>
            <span class="severity-badge" style="color:${color};background:${color}22">${d.severity}</span>
          </div>
          <div class="progress-track">
            <div class="progress-fill" style="width:${Math.min(d.risk_score * 100, 100)}%;background:${color}"></div>
          </div>
          <span class="risk-pct" style="color:${color}">${pct}%</span>
        </div>
        ${d.key_signs.length > 0 ? `<p class="signs">주요 증거: ${d.key_signs.slice(0, 3).join(" · ")}</p>` : ""}
      `;
    })
    .join("");

  // ML top3 rows for disease
  const mlTop3Rows = dis.ml_top3
    .map((d) => {
      const pct = (d.probability * 100).toFixed(1);
      const color = riskColor(d.probability);
      return `
        <div class="risk-row">
          <span class="risk-name">${d.name_kr}</span>
          <div class="progress-track">
            <div class="progress-fill" style="width:${Math.min(d.probability * 100, 100)}%;background:${color}"></div>
          </div>
          <span class="risk-pct" style="color:${color}">${pct}%</span>
        </div>
      `;
    })
    .join("");

  // Injury top3 rows
  const injuryRows = inj.top3
    .map((i) => {
      const pct = (i.probability * 100).toFixed(1);
      const color = riskColor(i.probability);
      return `
        <div class="risk-row">
          <span class="risk-name">${i.name_kr}</span>
          <div class="progress-track">
            <div class="progress-fill" style="width:${Math.min(i.probability * 100, 100)}%;background:${color}"></div>
          </div>
          <span class="risk-pct" style="color:${color}">${pct}%</span>
        </div>
      `;
    })
    .join("");

  // Priority actions list
  const actions = inj.priority_actions
    .map((a) => `<li>${a}</li>`)
    .join("");

  // Reasoning trace
  const traceRows = rsn.reasoning_trace
    .map(
      (step) => `
      <tr>
        <td>${step.step}</td>
        <td>${step.label}</td>
        <td>${step.prediction_kr}</td>
        <td>${(step.probability * 100).toFixed(1)}%</td>
      </tr>
    `
    )
    .join("");

  // Anomaly findings
  const anomalyEntries = Object.entries(rsn.anomaly_findings)
    .filter(([, findings]) => findings.length > 0)
    .map(
      ([modality, findings]) =>
        `<li><strong>${modality}</strong>: ${findings.join(", ")}</li>`
    )
    .join("");

  const injRiskColor = riskColor(inj.combined_risk_score);

  return `<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Shoealls 보행 분석 리포트</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans KR",
        sans-serif;
      background: #f8fafc;
      color: #1e293b;
      line-height: 1.6;
      padding: 32px 16px 64px;
    }

    .wrapper {
      max-width: 780px;
      margin: 0 auto;
    }

    /* Header */
    .report-header {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      margin-bottom: 32px;
      padding-bottom: 20px;
      border-bottom: 2px solid #e2e8f0;
    }

    .logo-row {
      display: flex;
      align-items: center;
      gap: 12px;
    }

    .logo-mark {
      width: 44px;
      height: 44px;
      background: #3B82F6;
      border-radius: 12px;
      display: flex;
      align-items: center;
      justify-content: center;
      color: #fff;
      font-weight: 800;
      font-size: 22px;
    }

    .report-title {
      font-size: 22px;
      font-weight: 700;
      color: #0f172a;
    }

    .report-subtitle {
      font-size: 12px;
      color: #64748b;
      margin-top: 2px;
    }

    .header-meta {
      text-align: right;
      font-size: 12px;
      color: #64748b;
      line-height: 1.8;
    }

    /* Print button */
    .print-btn {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      margin-bottom: 28px;
      padding: 8px 20px;
      background: #3B82F6;
      color: #fff;
      border: none;
      border-radius: 8px;
      font-size: 13px;
      font-weight: 600;
      cursor: pointer;
      transition: background 0.2s;
    }

    .print-btn:hover { background: #2563eb; }

    @media print {
      .print-btn { display: none; }
      body { background: #fff; padding: 16px; }
    }

    /* Section */
    .section {
      background: #fff;
      border: 1px solid #e2e8f0;
      border-radius: 14px;
      overflow: hidden;
      margin-bottom: 20px;
    }

    .section-header {
      display: flex;
      align-items: center;
      gap: 10px;
      padding: 14px 20px;
      border-bottom: 1px solid #e2e8f0;
    }

    .section-dot {
      width: 10px;
      height: 10px;
      border-radius: 50%;
      flex-shrink: 0;
    }

    .section-title {
      font-size: 15px;
      font-weight: 700;
      color: #0f172a;
    }

    .section-badge {
      margin-left: auto;
      padding: 2px 10px;
      border-radius: 20px;
      font-size: 11px;
      font-weight: 600;
    }

    .section-body {
      padding: 20px;
    }

    /* Classify */
    .classify-grid {
      display: grid;
      grid-template-columns: 120px 1fr;
      gap: 20px;
      align-items: center;
    }

    .classify-circle {
      width: 110px;
      height: 110px;
      border-radius: 50%;
      border: 3px solid #3B82F626;
      background: #f1f5f9;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      text-align: center;
    }

    .classify-circle .cls-label {
      font-size: 12px;
      color: #3B82F6;
      font-weight: 600;
      padding: 0 8px;
    }

    .classify-circle .cls-pct {
      font-size: 26px;
      font-weight: 800;
      color: #0f172a;
    }

    /* Risk rows */
    .risk-row {
      display: flex;
      align-items: center;
      gap: 10px;
      margin-bottom: 10px;
    }

    .risk-label {
      display: flex;
      align-items: center;
      gap: 6px;
      flex-shrink: 0;
      width: 160px;
    }

    .risk-name {
      font-size: 13px;
      color: #334155;
      font-weight: 500;
      min-width: 80px;
    }

    .severity-badge {
      font-size: 10px;
      font-weight: 600;
      padding: 1px 7px;
      border-radius: 10px;
      white-space: nowrap;
    }

    .progress-track {
      flex: 1;
      height: 7px;
      background: #e2e8f0;
      border-radius: 4px;
      overflow: hidden;
    }

    .progress-fill {
      height: 100%;
      border-radius: 4px;
      transition: width 0.4s ease;
    }

    .risk-pct {
      font-size: 12px;
      font-weight: 700;
      width: 44px;
      text-align: right;
      flex-shrink: 0;
    }

    .signs {
      font-size: 11px;
      color: #94a3b8;
      margin: -4px 0 10px 170px;
    }

    /* Sub heading */
    .sub-heading {
      font-size: 12px;
      font-weight: 700;
      color: #64748b;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      margin-bottom: 12px;
      margin-top: 20px;
    }

    .sub-heading:first-child { margin-top: 0; }

    /* Injury summary */
    .inj-summary {
      display: flex;
      align-items: center;
      gap: 20px;
      margin-bottom: 20px;
      padding: 14px 18px;
      background: #f8fafc;
      border-radius: 10px;
      border: 1px solid #e2e8f0;
    }

    .inj-score {
      font-size: 36px;
      font-weight: 800;
    }

    .inj-meta {
      font-size: 12px;
      color: #64748b;
    }

    .inj-grade {
      display: inline-block;
      padding: 2px 10px;
      border-radius: 20px;
      font-size: 12px;
      font-weight: 700;
      margin-top: 4px;
    }

    .timeline-note {
      font-size: 12px;
      color: #64748b;
      margin-bottom: 14px;
    }

    /* Actions list */
    .actions-list {
      list-style: none;
      padding: 0;
      margin: 0;
    }

    .actions-list li {
      position: relative;
      padding-left: 18px;
      font-size: 13px;
      color: #334155;
      margin-bottom: 6px;
      line-height: 1.5;
    }

    .actions-list li::before {
      content: "→";
      position: absolute;
      left: 0;
      color: #F59E0B;
      font-weight: 700;
    }

    /* Reasoning trace table */
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
    }

    th {
      background: #f1f5f9;
      padding: 8px 10px;
      text-align: left;
      color: #64748b;
      font-weight: 600;
      border-bottom: 1px solid #e2e8f0;
    }

    td {
      padding: 8px 10px;
      border-bottom: 1px solid #f1f5f9;
      color: #334155;
    }

    tr:last-child td { border-bottom: none; }

    /* Report text */
    .report-text {
      font-size: 13px;
      color: #334155;
      line-height: 1.8;
      white-space: pre-wrap;
      background: #f8fafc;
      border-radius: 8px;
      padding: 16px;
      border: 1px solid #e2e8f0;
    }

    /* Anomaly list */
    .anomaly-list {
      list-style: none;
      padding: 0;
      margin: 16px 0 0;
    }

    .anomaly-list li {
      font-size: 12px;
      color: #64748b;
      margin-bottom: 4px;
      padding-left: 14px;
      position: relative;
    }

    .anomaly-list li::before {
      content: "•";
      position: absolute;
      left: 0;
      color: #AF65FA;
    }

    /* Meta row */
    .meta-row {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-top: 12px;
      font-size: 11px;
      color: #94a3b8;
    }

    .meta-chip {
      background: #f1f5f9;
      border-radius: 6px;
      padding: 3px 9px;
    }

    /* Footer */
    .report-footer {
      margin-top: 32px;
      padding-top: 16px;
      border-top: 1px solid #e2e8f0;
      font-size: 11px;
      color: #94a3b8;
      text-align: center;
    }
  </style>
</head>
<body>
  <div class="wrapper">

    <!-- Header -->
    <div class="report-header">
      <div class="logo-row">
        <div class="logo-mark">S</div>
        <div>
          <div class="report-title">Shoealls 보행 분석 리포트</div>
          <div class="report-subtitle">Gait AI · 멀티모달 센서 기반 분석</div>
        </div>
      </div>
      <div class="header-meta">
        <div>${timestamp}</div>
        <div>프로파일: ${profile}</div>
        ${cls.is_demo_mode ? '<div style="color:#F59E0B;font-weight:600">데모 모드</div>' : ""}
      </div>
    </div>

    <!-- Print button -->
    <button class="print-btn" onclick="window.print()">
      &#128438; 인쇄 / PDF 저장
    </button>

    <!-- Section 1: 보행 분류 -->
    <div class="section">
      <div class="section-header">
        <div class="section-dot" style="background:#3B82F6"></div>
        <span class="section-title">보행 분류</span>
        <span class="section-badge" style="color:#3B82F6;background:#3B82F620">보행 분류</span>
      </div>
      <div class="section-body">
        <div class="classify-grid">
          <div class="classify-circle">
            <span class="cls-label">${cls.prediction_kr}</span>
            <span class="cls-pct">${(cls.confidence * 100).toFixed(1)}%</span>
          </div>
          <div>
            <p class="sub-heading">클래스별 확률</p>
            ${Object.entries(cls.class_probabilities)
              .sort((a, b) => b[1] - a[1])
              .map(([name, prob]) => {
                const pct = (prob * 100).toFixed(1);
                const color = riskColor(prob > 0.5 ? prob : 0);
                const barColor = "#3B82F6";
                const KR: Record<string, string> = {
                  normal: "정상 보행",
                  antalgic: "절뚝거림",
                  ataxic: "운동실조",
                  parkinsonian: "파킨슨",
                };
                return `
                  <div class="risk-row">
                    <span class="risk-name">${KR[name] ?? name}</span>
                    <div class="progress-track">
                      <div class="progress-fill" style="width:${Math.min(prob * 100, 100)}%;background:${barColor}"></div>
                    </div>
                    <span class="risk-pct" style="color:${barColor}">${pct}%</span>
                  </div>
                `;
              })
              .join("")}
          </div>
        </div>
      </div>
    </div>

    <!-- Section 2: 질환 위험도 -->
    <div class="section">
      <div class="section-header">
        <div class="section-dot" style="background:#10B981"></div>
        <span class="section-title">질환 위험도</span>
        <span class="section-badge" style="color:#10B981;background:#10B98120">질환 위험도</span>
      </div>
      <div class="section-body">
        <p class="sub-heading">ML 예측: ${dis.ml_prediction_kr} · ${(dis.ml_confidence * 100).toFixed(1)}%</p>
        ${mlTop3Rows}
        ${dis.top_diseases.length > 0 ? `<p class="sub-heading" style="margin-top:20px">규칙 기반 질환별 위험도</p>${diseaseRows}` : ""}
        ${
          dis.abnormal_biomarkers && dis.abnormal_biomarkers.length > 0
            ? `<p style="font-size:11px;color:#94a3b8;margin-top:8px">이상 감지 바이오마커: ${dis.abnormal_biomarkers.join(" / ")}</p>`
            : ""
        }
      </div>
    </div>

    <!-- Section 3: 부상 위험도 -->
    <div class="section">
      <div class="section-header">
        <div class="section-dot" style="background:#F59E0B"></div>
        <span class="section-title">부상 위험도</span>
        <span class="section-badge" style="color:#F59E0B;background:#F59E0B20">부상 예측</span>
      </div>
      <div class="section-body">
        <div class="inj-summary">
          <div>
            <div class="inj-score" style="color:${injRiskColor}">${(inj.combined_risk_score * 100).toFixed(1)}%</div>
            <div class="inj-meta">종합 위험도</div>
            <div class="inj-grade" style="color:${injRiskColor};background:${injRiskColor}22">${inj.combined_risk_grade}</div>
          </div>
          <div>
            <div style="font-size:13px;font-weight:600;color:#334155">${inj.predicted_injury_kr}</div>
            <div class="inj-meta">주요 예측 부상</div>
            <div class="inj-meta" style="margin-top:4px">${inj.timeline}</div>
          </div>
        </div>

        <p class="sub-heading">부상 부위별 위험도 (Top 3)</p>
        ${injuryRows}

        ${
          inj.priority_actions.length > 0
            ? `<p class="sub-heading" style="margin-top:20px">우선 권고 조치</p>
               <ul class="actions-list">${actions}</ul>`
            : ""
        }
      </div>
    </div>

    <!-- Section 4: AI 리포트 -->
    <div class="section">
      <div class="section-header">
        <div class="section-dot" style="background:#AF65FA"></div>
        <span class="section-title">AI 리포트</span>
        <span class="section-badge" style="color:#AF65FA;background:#AF65FA20">Chain-of-Reasoning</span>
      </div>
      <div class="section-body">
        <p class="sub-heading">추론 단계 (Reasoning Trace)</p>
        <table>
          <thead>
            <tr>
              <th>단계</th>
              <th>레이블</th>
              <th>예측</th>
              <th>확률</th>
            </tr>
          </thead>
          <tbody>
            ${traceRows}
          </tbody>
        </table>

        <p class="sub-heading" style="margin-top:20px">종합 소견</p>
        <div class="report-text">${rsn.report_kr || "(소견 없음)"}</div>

        ${
          anomalyEntries
            ? `<ul class="anomaly-list">${anomalyEntries}</ul>`
            : ""
        }

        <div class="meta-row">
          <span class="meta-chip">불확실성 ${(rsn.uncertainty * 100).toFixed(1)}%</span>
          <span class="meta-chip">근거 강도 ${(rsn.evidence_strength * 100).toFixed(1)}%</span>
          <span class="meta-chip">최종 예측 ${rsn.final_prediction_kr}</span>
          <span class="meta-chip">신뢰도 ${(rsn.confidence * 100).toFixed(1)}%</span>
          ${rsn.is_demo_mode ? '<span class="meta-chip" style="color:#F59E0B">데모 모드</span>' : ""}
        </div>
      </div>
    </div>

    <!-- Footer -->
    <div class="report-footer">
      Shoealls Gait AI · 이 리포트는 AI 분석 결과이며 의학적 진단을 대체하지 않습니다.
      <br/>생성 시각: ${timestamp}
    </div>

  </div>
</body>
</html>`;
}

export default function ExportButton({ result, profile }: Props) {
  function handleExport() {
    if (!result) return;

    const html = buildReport(result, profile);
    const win = window.open("", "_blank");
    if (!win) return;

    win.document.open();
    win.document.write(html);
    win.document.close();
  }

  return (
    <button
      onClick={handleExport}
      disabled={result === null}
      title="보행 분석 리포트를 새 창에서 열어 PDF로 저장하세요"
      aria-label="리포트 저장"
      className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-card border border-border text-textSec hover:text-textPri disabled:opacity-40 disabled:cursor-not-allowed transition-colors text-[13px] font-medium"
    >
      <span className="text-[13px] leading-none select-none">⬇</span>
      리포트 저장
    </button>
  );
}
