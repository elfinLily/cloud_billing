"""
processed 폴더의 모든 FOCUS 데이터 분석

1. data/processed/ 폴더의 모든 .csv 파일 로드
2. 각 파일별로 2패턴 탐지 (과다 프로비저닝, 미사용 리소스)
3. CloudProvider, PatternType 컬럼 추가
4. 하나의 .csv.gz로 통합 저장
"""

import pandas as pd
import yaml
from pathlib import Path
import logging

from focus_patterns import OverProvisioningDetector, UnusedResourceDetector


class AllFocusAnalyzer:
    """전체 FOCUS 데이터 분석기"""
    
    def __init__(self, config_path='config/focus_config.yaml'):
        """
        초기화
        
        Args:
            config_path: 설정 파일 경로
        """
        # 설정 로드
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 경로 설정
        self.focus_folder = Path(self.config['data']['focus_folder'])
        self.output_path = Path(self.config['data']['detected_patterns_output'])
        
        # 로깅 설정
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # 결과 저장
        self.all_results = []
    
    
    def find_focus_files(self):
        """
        processed 폴더의 모든 CSV 파일 찾기
        
        Returns:
            list: CSV 파일 경로 리스트
        """
        print("="*100)
        print("🔍 FOCUS 파일 검색 중...")
        print("="*100)
        print(f"   📂 폴더: {self.focus_folder}")
        
        csv_files = list(self.focus_folder.glob('*.csv'))
        
        print(f"\n✅ 발견된 파일: {len(csv_files)}개")
        for i, file in enumerate(csv_files, 1):
            print(f"   {i}. {file.name}")
        
        print("\n" + "="*100)
        
        return csv_files
    
    
    def detect_cloud_provider(self, df, filename):
        """
        CloudProvider 자동 감지
        
        Args:
            df: DataFrame
            filename: 파일명
        
        Returns:
            str: 'AWS' or 'GCP' or 'Unknown'
        """
        # 1. 파일명에서 추출
        filename_lower = filename.lower()
        if 'aws' in filename_lower:
            return 'AWS'
        elif 'gcp' in filename_lower or 'google' in filename_lower:
            return 'GCP'
        
        # 2. ProviderName 컬럼에서 추출
        if 'ProviderName' in df.columns:
            providers = df['ProviderName'].unique()
            if len(providers) == 1:
                provider = providers[0]
                if 'AWS' in provider or 'Amazon' in provider:
                    return 'AWS'
                elif 'Google' in provider or 'GCP' in provider:
                    return 'GCP'
        
        return 'Unknown'
    
    
    def analyze_file(self, file_path):
        """
        단일 파일 분석
        
        Args:
            file_path: CSV 파일 경로
        
        Returns:
            dict: {'over_provisioned': DataFrame, 'unused': DataFrame, 'provider': str}
        """
        print("\n" + "="*100)
        print(f"📊 분석 시작: {file_path.name}")
        print("="*100)
        
        # 데이터 로드
        df = pd.read_csv(file_path, low_memory=False)
        print(f"   ✅ 로드 완료: {len(df):,}건")
        
        # CloudProvider 감지
        provider = self.detect_cloud_provider(df, file_path.name)
        print(f"   ☁️  Provider: {provider}")
        
        # 패턴 1: 과다 프로비저닝
        detector1 = OverProvisioningDetector(df, self.config)
        over_prov = detector1.detect()
        
        if len(over_prov) > 0:
            over_prov['CloudProvider'] = provider
            over_prov['PatternType'] = 'OverProvisioning'
        
        # 패턴 2: 미사용 리소스
        detector2 = UnusedResourceDetector(df, self.config)
        unused = detector2.detect()
        
        if len(unused) > 0:
            unused['CloudProvider'] = provider
            unused['PatternType'] = 'Unused'
        
        return {
            'over_provisioned': over_prov,
            'unused': unused,
            'provider': provider
        }
    
    
    def merge_results(self):
        """
        모든 결과 병합
        
        Returns:
            DataFrame: 통합된 탐지 결과
        """
        print("\n" + "="*100)
        print("🔗 결과 병합 중...")
        print("="*100)
        
        all_patterns = []
        
        for result in self.all_results:
            if len(result['over_provisioned']) > 0:
                all_patterns.append(result['over_provisioned'])
            
            if len(result['unused']) > 0:
                all_patterns.append(result['unused'])
        
        if len(all_patterns) == 0:
            print("⚠️  탐지된 패턴이 없습니다!")
            return pd.DataFrame()
        
        # 병합
        merged = pd.concat(all_patterns, ignore_index=True)
        
        print(f"✅ 병합 완료: {len(merged):,}건")
        print(f"\n📊 CloudProvider별:")
        for provider, count in merged['CloudProvider'].value_counts().items():
            print(f"   • {provider}: {count:,}건")
        
        print(f"\n📊 PatternType별:")
        for pattern, count in merged['PatternType'].value_counts().items():
            print(f"   • {pattern}: {count:,}건")
        
        print("\n" + "="*100)
        
        return merged
    
    
    def save_results(self, df):
        """
        결과 저장 (.csv.gz)
        
        Args:
            df: 저장할 DataFrame
        """
        if len(df) == 0:
            print("⚠️  저장할 데이터가 없습니다.")
            return
        
        print("\n" + "="*100)
        print("💾 결과 저장 중...")
        print("="*100)
        
        # 디렉토리 생성
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 저장 (압축)
        df.to_csv(
            self.output_path,
            index=False,
            encoding='utf-8-sig',
            compression='gzip'
        )
        
        size_mb = self.output_path.stat().st_size / 1024**2
        
        print(f"\n✅ 저장 완료!")
        print(f"   📂 경로: {self.output_path}")
        print(f"   📊 레코드: {len(df):,}건")
        print(f"   💾 크기: {size_mb:.1f} MB")
        print("\n" + "="*100)
    
    
    def print_summary(self, df):
        """
        최종 요약 통계
        
        Args:
            df: 통합 결과 DataFrame
        """
        if len(df) == 0:
            return
        
        print("\n" + "="*100)
        print("📊 최종 분석 요약")
        print("="*100)
        
        print(f"\n✅ 총 탐지 건수: {len(df):,}건")
        
        # CloudProvider × PatternType 교차표
        print(f"\n📊 CloudProvider × PatternType:")
        crosstab = pd.crosstab(df['CloudProvider'], df['PatternType'], margins=True)
        print(crosstab)
        
        # 비용 통계
        cost_cols = [col for col in df.columns if 'cost' in col.lower() or 'savings' in col.lower()]
        
        if cost_cols:
            print(f"\n💰 비용 통계:")
            for provider in df['CloudProvider'].unique():
                if provider == 'All':
                    continue
                
                provider_df = df[df['CloudProvider'] == provider]
                
                # 과다 프로비저닝
                over_prov = provider_df[provider_df['PatternType'] == 'OverProvisioning']
                if len(over_prov) > 0 and 'PotentialSavings' in over_prov.columns:
                    savings = over_prov['PotentialSavings'].sum()
                    print(f"   • {provider} 과다프로비저닝 절감 가능: ${savings:,.2f}/월")
                
                # 미사용 리소스
                unused = provider_df[provider_df['PatternType'] == 'Unused']
                if len(unused) > 0 and 'WastedCost' in unused.columns:
                    waste = unused['WastedCost'].sum()
                    print(f"   • {provider} 미사용 리소스 낭비: ${waste:,.2f}/월")
        
        print("\n" + "="*100)
    
    
    def run(self):
        """
        전체 분석 실행
        
        Returns:
            DataFrame: 통합 결과
        """
        # 1. 파일 찾기
        csv_files = self.find_focus_files()
        
        if len(csv_files) == 0:
            print("❌ 분석할 파일이 없습니다!")
            return pd.DataFrame()
        
        # 2. 각 파일 분석
        for file_path in csv_files:
            result = self.analyze_file(file_path)
            self.all_results.append(result)
        
        # 3. 결과 병합
        merged_df = self.merge_results()
        
        # 4. 저장
        self.save_results(merged_df)
        
        # 5. 요약
        self.print_summary(merged_df)
        
        print("\n✅ 모든 분석 완료!")
        
        return merged_df


# ==================== 메인 실행 ====================
if __name__ == "__main__":
    
    print("="*100)
    print("🚀 전체 FOCUS 데이터 분석 시작")
    print("="*100)
    
    analyzer = AllFocusAnalyzer()
    results = analyzer.run()
    
    print("\n" + "="*100)
    print("🎉 완료!")
    print("="*100)