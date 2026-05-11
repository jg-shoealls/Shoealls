import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any

class GaitTrendAnalyzer:
    """
    InfluxDB 또는 전처리된 CSV 데이터를 기반으로 보행 패턴의 장기 추세를 분석합니다.
    의료진이 사용자의 보행 능력 변화를 시각화할 수 있도록 데이터를 가공합니다.
    """
    
    @staticmethod
    def get_weekly_summary(df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        주간 보행 지표 요약 (최근 7일)
        """
        if df.empty:
            return []
            
        # 타임스탬프 기준 정렬
        df = df.sort_values('fe_timestamp')
        
        # 날짜별 그룹화
        df['date'] = df['fe_timestamp'].dt.date
        daily_avg = df.groupby('date').agg({
            'speed': 'mean',
            'tilt': 'mean',
            'fe_event_value': 'mean'
        }).reset_index()
        
        # 마지막 7일 데이터
        summary = []
        for _, row in daily_avg.tail(7).iterrows():
            summary.append({
                "date": row['date'].strftime("%m/%d"),
                "speed": round(float(row['speed']), 3),
                "stability": round(100 - (float(row['tilt']) * 10), 1), # 임의의 안정성 점수화
                "activity": int(len(df[df['date'] == row['date']])) # 활동량 (로그 수)
            })
            
        return summary

    @staticmethod
    def detect_anomalous_trends(df: pd.DataFrame) -> List[str]:
        """
        장기적인 보행 기능 저하 패턴 감지 (예: 속도 지속 감소)
        """
        findings = []
        if len(df) < 100:
            return findings

        # 이동 평균 계산
        df['speed_ma'] = df['speed'].rolling(window=50).mean()
        
        recent_speed = df['speed_ma'].iloc[-1]
        past_speed = df['speed_ma'].iloc[-50]
        
        if recent_speed < past_speed * 0.9:
            findings.append("최근 보행 속도가 과거 대비 10% 이상 감소했습니다. (근감소증 또는 인지 저하 전조 가능성)")
            
        return findings
