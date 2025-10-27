"""
미사용 리소스 탐지 테스트 (명확한 조건)

조건:
1. EffectiveCost != 0 일 때: CommitmentDiscountStatus = 'Unused'
2. EffectiveCost == 0 일 때: BilledCost = 0 AND (ConsumedQuantity = 0 OR null)
"""

import pandas as pd
import sys
from pathlib import Path

# 데이터 로드
CSV_PATH = 'data/raw/focus_data_table.csv'

print("="*100)
print("🔄 데이터 로딩...")
print("="*100)

df = pd.read_csv(CSV_PATH, low_memory=False)
print(f"✅ 로드 완료: {len(df):,}건\n")

# ==================== 미사용 리소스 탐지 ====================

print("="*100)
print("🔍 미사용 리소스 탐지")
print("="*100)

print("\n📌 탐지 조건:")
print("   1. EffectiveCost != 0 → CommitmentDiscountStatus = 'Unused'")
print("   2. EffectiveCost == 0 → BilledCost = 0 AND (ConsumedQuantity = 0 or null)")

unused_all = []

# ========== 조건 1: EffectiveCost != 0 & Commitment Unused ==========
print(f"\n" + "-"*100)
print("📌 조건 1: EffectiveCost != 0 & CommitmentDiscountStatus = 'Unused'")
print("-"*100)

if 'EffectiveCost' not in df.columns:
    print("❌ EffectiveCost 컬럼 없음")
    condition1 = pd.DataFrame()
elif 'CommitmentDiscountStatus' not in df.columns:
    print("❌ CommitmentDiscountStatus 컬럼 없음")
    condition1 = pd.DataFrame()
else:
    condition1 = df[
        (df['EffectiveCost'] != 0) &
        (df['CommitmentDiscountStatus'].str.lower() == 'unused')
    ].copy()
    
    if len(condition1) > 0:
        condition1['UnusedReason'] = 'Commitment-Unused'
        condition1['WastedCost'] = condition1['EffectiveCost']
        unused_all.append(condition1)
        
        print(f"🚨 발견: {len(condition1):,}건")
        print(f"💸 낭비 비용: ${condition1['EffectiveCost'].sum():,.2f}")
        
        # Commitment 타입별
        if 'CommitmentDiscountType' in condition1.columns:
            print(f"\n📊 Commitment 타입별:")
            for ctype, group in condition1.groupby('CommitmentDiscountType'):
                count = len(group)
                cost = group['EffectiveCost'].sum()
                print(f"   • {ctype:20s}: {count:6,}건 | ${cost:,.2f}")
    else:
        print("✅ 없음")

# ========== 조건 2: EffectiveCost == 0 & BilledCost == 0 & (ConsumedQuantity == 0 or null) ==========
print(f"\n" + "-"*100)
print("📌 조건 2: EffectiveCost = 0 & BilledCost = 0 & (ConsumedQuantity = 0 or null)")
print("-"*100)

if 'EffectiveCost' not in df.columns:
    print("❌ EffectiveCost 컬럼 없음")
    condition2 = pd.DataFrame()
elif 'BilledCost' not in df.columns:
    print("❌ BilledCost 컬럼 없음")
    condition2 = pd.DataFrame()
elif 'ConsumedQuantity' not in df.columns:
    print("❌ ConsumedQuantity 컬럼 없음")
    condition2 = pd.DataFrame()
else:
    condition2 = df[
        (df['EffectiveCost'] == 0) &
        (df['BilledCost'] == 0) &
        ((df['ConsumedQuantity'] == 0) | (df['ConsumedQuantity'].isna()))
    ].copy()
    
    if len(condition2) > 0:
        condition2['UnusedReason'] = 'Zero-Cost-Zero-Usage'
        condition2['WastedCost'] = 0  # 비용은 0이지만 리소스는 존재
        unused_all.append(condition2)
        
        print(f"🚨 발견: {len(condition2):,}건")
        print(f"⚠️ 비용은 0이지만 불필요한 리소스로 추정")
        
        # ConsumedQuantity 상태별
        null_count = condition2['ConsumedQuantity'].isna().sum()
        zero_count = (condition2['ConsumedQuantity'] == 0).sum()
        
        print(f"\n📊 사용량 상태:")
        print(f"   • null: {null_count:,}건")
        print(f"   • 0: {zero_count:,}건")
        
        # 서비스별
        print(f"\n📊 서비스별 Top 5:")
        for service, count in condition2['ServiceName'].value_counts().head(5).items():
            pct = count / len(condition2) * 100
            print(f"   • {service[:50]}: {count:,}건 ({pct:.1f}%)")
    else:
        print("✅ 없음")

# ========== 결과 통합 ==========
print(f"\n" + "="*100)
print("📊 최종 결과")
print("="*100)

if len(unused_all) == 0:
    print("\n✅ 미사용 리소스를 찾을 수 없습니다!")
    print("   모든 리소스가 적절히 사용되고 있습니다.")
else:
    # 통합
    result = pd.concat(unused_all, ignore_index=True)
    
    # 중복 제거
    if 'ResourceId' in result.columns:
        before = len(result)
        result = result.drop_duplicates(subset=['ResourceId'])
        if before > len(result):
            print(f"\n⚠️ 중복 제거: {before - len(result):,}건")
    
    print(f"\n✅ 총 미사용 리소스: {len(result):,}건")
    
    # 조건별 통계
    print(f"\n📊 조건별 분포:")
    for reason in result['UnusedReason'].unique():
        subset = result[result['UnusedReason'] == reason]
        count = len(subset)
        pct = count / len(result) * 100
        cost = subset['WastedCost'].sum()
        print(f"   • {reason:25s}: {count:7,}건 ({pct:5.1f}%) | ${cost:,.2f}")
    
    total_waste = result['WastedCost'].sum()
    print(f"\n💰 총 낭비 비용: ${total_waste:,.2f}/월")
    if total_waste > 0:
        print(f"💰 연간 낭비: ${total_waste * 12:,.2f}")
    
    # 조건 1 상위 10개
    if len(condition1) > 0:
        print(f"\n" + "-"*100)
        print("📈 조건 1 (Commitment Unused) 상위 10개:")
        print("-"*100)
        
        display_cols = ['ResourceId', 'ServiceName', 'CommitmentDiscountType', 
                       'CommitmentDiscountStatus', 'EffectiveCost', 'BilledCost']
        available = [col for col in display_cols if col in condition1.columns]
        
        top10_c1 = condition1.nlargest(10, 'EffectiveCost')[available]
        
        pd.set_option('display.max_colwidth', 40)
        pd.set_option('display.float_format', lambda x: f'{x:.6f}' if abs(x) < 0.01 else f'{x:.2f}')
        
        print(top10_c1.to_string(index=False))
    
    # 조건 2 샘플 10개
    if len(condition2) > 0:
        print(f"\n" + "-"*100)
        print("📋 조건 2 (Zero Cost & Zero Usage) 샘플 10개:")
        print("-"*100)
        
        display_cols = ['ResourceId', 'ServiceName', 'ResourceType',
                       'EffectiveCost', 'BilledCost', 'ConsumedQuantity']
        available = [col for col in display_cols if col in condition2.columns]
        
        sample_c2 = condition2[available].head(10)
        print(sample_c2.to_string(index=False))
    
    # CSV 저장
    print(f"\n" + "="*100)
    print("💾 결과 저장")
    print("="*100)
    
    output_path = 'results/unused_resources_detected.csv'
    result.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"✅ {output_path}")

print("\n" + "="*100)
print("✅ 분석 완료!")
print("="*100)