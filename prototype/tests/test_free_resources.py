"""
비용 미청구 리소스 조회 테스트
"""

import pandas as pd
import sys
from pathlib import Path

# 프로젝트 루트 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

# 데이터 로드
CSV_PATH = 'data/raw/focus_data_table.csv'

print("="*100)
print("🔄 데이터 로딩...")
print("="*100)

df = pd.read_csv(CSV_PATH, low_memory=False)

print(f"✅ 로드 완료: {len(df):,}건\n")

# ==================== 비용 미청구 리소스 조회 ====================

print("="*100)
print("🆓 비용 미청구 리소스 조회")
print("="*100)

# BilledCost == 0인 리소스
free_resources = df[df['BilledCost'] == 0].copy()

# 기본 통계
print(f"\n📊 기본 통계:")
print(f"   • 전체 리소스: {len(df):,}건")
print(f"   • 비용 청구 (>0): {len(df[df['BilledCost'] > 0]):,}건")
print(f"   • 비용 미청구 (=0): {len(free_resources):,}건 ({len(free_resources)/len(df)*100:.1f}%)")

if len(free_resources) == 0:
    print("\n❌ 비용 미청구 리소스가 없습니다.")
    sys.exit(0)

# ChargeDescription 분석
print(f"\n" + "="*100)
print("📝 ChargeDescription 분석 (왜 비용이 0인가?)")
print("="*100)

descriptions = free_resources['ChargeDescription'].value_counts()

print(f"\nTop 20개:")
for i, (desc, count) in enumerate(descriptions.head(20).items(), 1):
    pct = count / len(free_resources) * 100
    # 긴 description 잘라서 표시
    desc_short = desc[:90] + "..." if len(desc) > 90 else desc
    print(f"{i:3d}. [{count:6,}건 | {pct:5.1f}%] {desc_short}")

# 키워드 검색
print(f"\n" + "="*100)
print("🔍 무료 키워드 검색")
print("="*100)

keywords = {
    'free tier': '프리티어',
    'free': '무료',
    '$0.00': '$0.00',
    'no charge': '무료',
    'included': '포함됨',
    'credit': '크레딧',
    'promotional': '프로모션',
    'trial': '체험판',
}

descriptions_lower = free_resources['ChargeDescription'].str.lower()

print(f"\n키워드 매칭 결과:")
for keyword, label in keywords.items():
    matches = descriptions_lower.str.contains(keyword, na=False).sum()
    if matches > 0:
        pct = matches / len(free_resources) * 100
        print(f"   • '{keyword:15s}' ({label:10s}): {matches:6,}건 ({pct:5.1f}%)")

# 서비스별 분포
print(f"\n" + "="*100)
print("📊 서비스별 분포")
print("="*100)

service_counts = free_resources['ServiceName'].value_counts().head(10)

print(f"\nTop 10 서비스:")
for i, (service, count) in enumerate(service_counts.items(), 1):
    pct = count / len(free_resources) * 100
    service_short = service[:50] + "..." if len(service) > 50 else service
    print(f"{i:3d}. [{count:6,}건 | {pct:5.1f}%] {service_short}")

# 리소스 타입별
print(f"\n" + "="*100)
print("📦 리소스 타입별 분포")
print("="*100)

type_counts = free_resources['ResourceType'].value_counts()

for rtype, count in type_counts.items():
    pct = count / len(free_resources) * 100
    print(f"   • {rtype:20s}: {count:6,}건 ({pct:5.1f}%)")

# ConsumedQuantity 확인
print(f"\n" + "="*100)
print("📈 사용량 (ConsumedQuantity) 분석")
print("="*100)

print(f"\n통계:")
print(f"   • 평균: {free_resources['ConsumedQuantity'].mean():.6f}")
print(f"   • 중앙값: {free_resources['ConsumedQuantity'].median():.6f}")
print(f"   • 최소: {free_resources['ConsumedQuantity'].min():.6f}")
print(f"   • 최대: {free_resources['ConsumedQuantity'].max():.6f}")
print(f"   • 표준편차: {free_resources['ConsumedQuantity'].std():.6f}")

# 사용량도 0인 경우
zero_usage = free_resources[free_resources['ConsumedQuantity'] == 0]
print(f"\n⚠️ 비용 0 + 사용량 0: {len(zero_usage):,}건 ({len(zero_usage)/len(free_resources)*100:.1f}%)")

# 사용량은 있는데 비용 0
nonzero_usage = free_resources[free_resources['ConsumedQuantity'] > 0]
print(f"✅ 비용 0 + 사용량 있음: {len(nonzero_usage):,}건 ({len(nonzero_usage)/len(free_resources)*100:.1f}%)")

# 샘플 데이터
print(f"\n" + "="*100)
print("📋 샘플 데이터 (처음 15개)")
print("="*100)

sample_cols = ['ResourceId', 'ServiceName', 'ChargeDescription', 'ConsumedQuantity', 'ConsumedUnit']
sample = free_resources[sample_cols].head(15)

pd.set_option('display.max_colwidth', 70)
pd.set_option('display.width', None)
pd.set_option('display.float_format', lambda x: f'{x:.6f}')

print("\n" + sample.to_string(index=False))

# CSV 저장 (선택)
print(f"\n" + "="*100)
print("💾 결과 저장")
print("="*100)

output_path = 'results/free_resources_analysis.csv'
free_resources.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"✅ 저장 완료: {output_path}")

print("\n" + "="*100)
print("✅ 분석 완료!")
print("="*100)