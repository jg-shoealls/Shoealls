import pandas as pd
from typing import List
from fastapi import APIRouter, HTTPException, Query
from src.analysis.trend_tracker import GaitTrendAnalyzer

router = APIRouter(prefix="/api/v1/trends", tags=["Monitoring"])

@router.get("/summary")
async def get_gait_trends(user_id: str = "default"):
    """
    사용자의 최근 7일간 보행 지표 추세를 반환합니다.
    (실제 구현에서는 DB 쿼리 필요, 여기서는 CES.csv 기반 샘플링)
    """
    try:
        # DB 연동 전까지는 CES.csv를 mock DB로 활용
        df = pd.read_csv('CES.csv', sep='\t')
        df['fe_timestamp'] = pd.to_datetime(df['fe_timestamp'])
        
        # JSON 파싱 (speed 추출)
        import json
        def parse_speed(raw):
            try: return json.loads(raw).get('speed', 0.0)
            except: return 0.0
        
        def parse_tilt(raw):
            try: return json.loads(raw).get('tilt', 0.0)
            except: return 0.0
            
        df['speed'] = df['fe_raw_data'].apply(parse_speed)
        df['tilt'] = df['fe_raw_data'].apply(parse_tilt)
        
        summary = GaitTrendAnalyzer.get_weekly_summary(df)
        findings = GaitTrendAnalyzer.detect_anomalous_trends(df)
        
        return {
            "user_id": user_id,
            "weekly_data": summary,
            "clinical_findings": findings
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
