"""
API 키 유효성 디버깅
"""
import os
from dotenv import load_dotenv

load_dotenv(dotenv_path='.env')

print("="*60)
print("🔍 API 키 디버깅")
print("="*60)

# 1. 키 존재 확인
print("\n1️⃣  API 키 존재 확인")
gemini_key = os.getenv('GEMINI_API_KEY')
claude_key = os.getenv('CLAUDE_API_KEY')

print(f"   Gemini 키: {'✅ 있음' if gemini_key else '❌ 없음'}")
print(f"   Claude 키: {'✅ 있음' if claude_key else '❌ 없음'}")

# 2. 키 형식 확인
print("\n2️⃣  API 키 형식 확인")
if gemini_key:
    print(f"   Gemini 시작: {gemini_key[:10]}")
    print(f"   Gemini 길이: {len(gemini_key)} 자")
    
if claude_key:
    print(f"   Claude 시작: {claude_key[:10]}")
    print(f"   Claude 길이: {len(claude_key)} 자")

# 3. Gemini 상세 테스트
print("\n3️⃣  Gemini 상세 테스트")
try:
    import google.generativeai as genai
    genai.configure(api_key=gemini_key)
    
    # 사용 가능한 모델 확인
    print("   사용 가능한 모델:")
    models = genai.list_models()
    for model in models:
        if 'generateContent' in model.supported_generation_methods:
            print(f"     - {model.name}")
            
except Exception as e:
    print(f"   ❌ 에러: {e}")

# 4. Claude 상세 테스트
print("\n4️⃣  Claude 상세 테스트")
try:
    from anthropic import Anthropic
    
    client = Anthropic(api_key=claude_key)
    
    # 기본 정보 확인
    print(f"   ✅ Claude 클라이언트 생성 성공")
    
    # 간단한 호출 (모델명 없이 테스트)
    # - 이건 에러날 것이지만 API 키가 유효한지 확인 가능
    
    models_to_test = [
        'claude-3-opus-20250219',      # 최신
        'claude-3-5-sonnet-20241022',  # 기존
        'claude-3-5-haiku-20241022',   # 빠른 버전
        'claude-3-sonnet-20240229',    # 구 버전
    ]
    
    print("="*60)
    print("🔍 Claude 사용 가능한 모델 확인")
    print("="*60)
    
    for model in models_to_test:
        print(f"\n테스트: {model}")
        try:
            message = client.messages.create(
                model=model,
                max_tokens=10,
                messages=[{"role": "user", "content": "hi"}]
            )
            print(f"   ✅ 작동함! 이 모델을 사용하세요")
            break
        except Exception as e:
            error_msg = str(e)
            if "not found" in error_msg.lower() or "404" in error_msg:
                print(f"   ❌ 모델 없음")
            else:
                print(f"   ⚠️  다른 에러: {e}")
    
    print("\n" + "="*60)



    #####
    
except Exception as e:
    print(f"   ❌ 에러: {e}")

print("\n" + "="*60)