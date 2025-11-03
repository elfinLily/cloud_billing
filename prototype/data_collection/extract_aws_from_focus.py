"""
FOCUS 데이터에서 AWS 데이터만 추출
"""

import pandas as pd
import yaml
from pathlib import Path


class AWSExtractor:
    """AWS 데이터 추출기"""
    
    def __init__(self, config_path='config/focus_config.yaml'):
        """초기화"""
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.input_path = self.config['data']['raw_path']
        self.output_path = self.config['data']['aws_focus_output']
    
    
    def extract(self):
        """AWS 데이터 추출"""
        # 로드
        df = pd.read_csv(self.input_path)
        
        # AWS 필터링
        aws_df = df[df['ProviderName'] == 'AWS'].copy()
        
        # 저장
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
        aws_df.to_csv(self.output_path, index=False)
        
        print(f"✅ AWS 데이터 추출: {len(aws_df):,}건")
        
        return aws_df


if __name__ == "__main__":
    extractor = AWSExtractor()
    extractor.extract()