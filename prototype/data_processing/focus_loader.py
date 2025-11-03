# -*- coding: utf-8 -*-
"""
FOCUS 데이터 로더
CSV 파일을 읽고 전처리하는 모듈
"""

import pandas as pd
import yaml
import os
from pathlib import Path


class FocusDataLoader:
    """FOCUS 데이터 로더"""
    
    def __init__(self, config_path='config/focus_config.yaml'):
        """
        초기화
        
        Args:
            config_path: 설정 파일 경로
        """
        # 설정 로드
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.data_path = self.config['data']['raw_path']
        self.df = None
    
    
    def load(self, use_sample=False):
        """
        데이터 로드
        
        Args:
            use_sample: True면 샘플 데이터 사용
        
        Returns:
            DataFrame: 로드된 데이터
        """
        if use_sample:
            path = self.config['data']['sample_path']
        else:
            path = self.data_path
        
        print("="*100)
        print(f"🔄 데이터 로딩: {path}")
        print("="*100)
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"❌ 파일을 찾을 수 없습니다: {path}")
        
        # CSV 로드
        self.df = pd.read_csv(path, low_memory=False)
        
        # 날짜 컬럼 변환
        self._convert_date_columns()
        
        print(f"✅ 로드 완료!")
        print(f"   📊 총 레코드: {len(self.df):,} 건")
        print(f"   📋 총 컬럼: {len(self.df.columns)} 개")
        print(f"   💾 메모리: {self.df.memory_usage(deep=True).sum() / 1024**2:.1f} MB\n")
        
        return self.df
    
    
    def _convert_date_columns(self):
        """날짜 컬럼 자동 변환"""
        date_keywords = ['date', 'period', 'time']
        
        for col in self.df.columns:
            if any(keyword in col.lower() for keyword in date_keywords):
                try:
                    self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
                except:
                    pass
    
    
    def get_schema(self):
        """스키마 정보 반환"""
        schema = []
        
        for col in self.df.columns:
            schema.append({
                'column': col,
                'dtype': str(self.df[col].dtype),
                'null_count': int(self.df[col].isna().sum()),
                'null_pct': float(self.df[col].isna().sum() / len(self.df) * 100)
            })
        
        return pd.DataFrame(schema)
    
    
    def get_summary(self):
        """데이터 요약 통계"""
        summary = {
            'total_records': len(self.df),
            'total_columns': len(self.df.columns),
            'memory_mb': self.df.memory_usage(deep=True).sum() / 1024**2
        }
        
        # 비용 통계
        cost_cols = [col for col in self.df.columns if 'cost' in col.lower()]
        if cost_cols:
            cost_col = cost_cols[0]
            summary['total_cost'] = float(self.df[cost_col].sum())
            summary['avg_cost'] = float(self.df[cost_col].mean())
        
        return summary