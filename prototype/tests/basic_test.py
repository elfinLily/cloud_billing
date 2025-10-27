import pandas as pd

# CSV 읽기
# df = pd.read_csv('FOCUS-Sample-Data/FOCUS-1.0/focus_data_table.csv')

# # 날짜 필터링 (ChargePeriodStart, ChargePeriodEnd가 날짜 형식이어야 함)
# df['ChargePeriodStart'] = pd.to_datetime(df['ChargePeriodStart'])
# df['ChargePeriodEnd'] = pd.to_datetime(df['ChargePeriodEnd'])

# # 필터 적용
# start_date = '2024-09-01'
# end_date = '2024-12-31'

# filtered = df[
#     (df['ChargePeriodStart'] >= start_date) & 
#     (df['ChargePeriodEnd'] < end_date)
# ]

# # GROUP BY와 SUM
# result = filtered.groupby([
#     'ProviderName',
#     'PublisherName',
#     'InvoiceIssuerName'
# ]).agg({
#     'BilledCost': 'sum'
# }).reset_index()

# # 컬럼 이름 변경
# result.rename(columns={'BilledCost': 'TotalBilledCost'}, inplace=True)

# # 정렬
# result = result.sort_values('TotalBilledCost', ascending=False)

# print("="*100)
# print("📊 쿼리 결과")
# print("="*100)
# print(f"\n총 {len(result):,}건 조회됨\n")
# print(result.to_string(index=False))

# 컬럼명 읽기
df = pd.read_csv('./data/raw/focus_data_table.csv')

# 컬럼 목록
print("="*80)
print("📋 전체 컬럼 목록")
print("="*80)
for i, col in enumerate(df.columns, 1):
    print(f"{i:3d}. {col}")

# 각 컬럼의 샘플 값
print("\n" + "="*80)
print("📊 각 컬럼의 샘플 값 (처음 3개)")
print("="*80)
for col in df.columns:
    sample = df[col].dropna().head(3).tolist()
    print(f"{col:40s}: {sample}")