import os
from datetime import datetime, timedelta
import json
from dotenv import load_dotenv
load_dotenv(dotenv_path='.env')

# GCP 테스트
def test_gcp():
    print("\n" + "="*50)
    print("🔍 GCP Billing API 테스트")
    print("="*50)
    
    try:
        from google.cloud import billing_v1
        from google.oauth2 import service_account
        
        # 자격증명 로드
        credentials = service_account.Credentials.from_service_account_file(
            'prototype/config/gcp_key.json'
        )
        
        # 클라이언트 생성
        client = billing_v1.CloudBillingClient(credentials=credentials)
        
        print(f"✅ GCP 연결 성공")
        print(f"   Project ID: {credentials.project_id}")
        print(f"   Service Account: {credentials.service_account_email}")
        
        return True
        
    except Exception as e:
        print(f"❌ GCP 연결 실패: {e}")
        return False


# AWS 테스트
def test_aws():
    print("\n" + "="*50)
    print("🔍 AWS Cost Explorer API 테스트")
    print("="*50)
    
    try:
        import boto3
        
        # AWS 클라이언트 생성
        ce_client = boto3.client('ce', region_name='us-east-1')
        sts_client = boto3.client('sts')
        
        # 자격증명 확인
        identity = sts_client.get_caller_identity()
        print(f"✅ AWS 연결 성공")
        print(f"   Account ID: {identity['Account']}")
        print(f"   User: {identity['Arn']}")
        
        # 실제 데이터 조회 테스트
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=7)
        
        response = ce_client.get_cost_and_usage(
            TimePeriod={
                'Start': start_date.strftime('%Y-%m-%d'),
                'End': end_date.strftime('%Y-%m-%d')
            },
            Granularity='DAILY',
            Metrics=['UnblendedCost'],
            GroupBy=[
                {
                    'Type': 'DIMENSION',
                    'Key': 'SERVICE'
                }
            ]
        )
        
        print(f"✅ 비용 데이터 조회 성공")
        print(f"   조회 기간: {start_date} ~ {end_date}")
        print(f"   데이터 포인트 수: {len(response['ResultsByTime'])}")
        
        return True
        
    except Exception as e:
        print(f"❌ AWS 연결 실패: {e}")
        return False


# Gemini API 테스트
def test_gemini():
    print("\n" + "="*50)
    print("🔍 Gemini API 테스트")
    print("="*50)
    
    try:
        import google.generativeai as genai
        
        # API 키 설정
        gemini_key = os.getenv('GEMINI_API_KEY')
        if not gemini_key:
            print("❌ GEMINI_API_KEY 환경변수 미설정")
            return False
        
        genai.configure(api_key=gemini_key)
        
        # 간단한 테스트 호출
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content("비용 분석 시스템이라고 말해줘")
        
        print(f"✅ Gemini 연결 성공")
        print(f"   모델: gemini-2.5-flash")
        print(f"   응답: {response.text[:50]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Gemini 연결 실패: {e}")
        return False


# Claude API 테스트
def test_claude():
    print("\n" + "="*50)
    print("🔍 Claude API 테스트")
    print("="*50)
    
    try:
        from anthropic import Anthropic
        
        # API 키 설정
        claude_key = os.getenv('CLAUDE_API_KEY')
        if not claude_key:
            print("❌ CLAUDE_API_KEY 환경변수 미설정")
            return False
        
        client = Anthropic(api_key=claude_key)
        
        # 간단한 테스트 호출
        message = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=100,
            messages=[
                {"role": "user", "content": "비용 분석 시스템이라고 말해줘"}
            ]
        )
        
        print(f"✅ Claude 연결 성공")
        print(f"   모델: claude-3-5-haiku-20241022")
        print(f"   응답: {message.content[0].text[:50]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Claude 연결 실패: {e}")
        return False


# 메인 실행
if __name__ == "__main__":
    print("\n🚀 멀티클라우드 FinOps 시스템 - API 연결 테스트")
    print("="*50)
    
    results = {
        'GCP': test_gcp(),
        'AWS': test_aws(),
        'Gemini': test_gemini(),
        'Claude': test_claude()
    }
    
    print("\n" + "="*50)
    print("📊 테스트 결과 요약")
    print("="*50)
    
    for api_name, success in results.items():
        status = "✅ 성공" if success else "❌ 실패"
        print(f"{api_name:15} {status}")
    
    all_success = all(results.values())
    
    print("\n" + "="*50)
    if all_success:
        print("✅ 모든 API 연결 테스트 완료!")
        print("   다음 단계: API 수집 스크립트 작성")
    else:
        print("❌ 일부 API 연결 실패")
        print("   설정을 다시 확인하세요")
    print("="*50 + "\n")