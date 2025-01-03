# TALLRec: An Effective and Efficient Tuning Framework to Align Large Language Model with Recommendation

## Abstract
- LLM의 강점을 사용하여 추천시스템에 적용하지만 사전학습이나 추천시스템의 특성간의 차이점으로 큰 성능을 발휘하지 못함
- 그 간격을 채우고자 TALLRec을 제안함.

## Introduction
- LLM의 광범위한 지식과 일반화는 높은 성능을 자랑하지만, 추천시스템에서는 서브적으로만 활용
- 이전 LLM을 병합한 시도는 In-context에 의존
    - Open AI의 API를 이용한 toolformer
    - 기존 RecSys에서 받은 후보군을 re-ranking
    - 답변을 거부하거나 무조건 긍정적인 답변을 도출
- 왜 위와 같이 실패하였는가?
    - LLM의 언어 학습과 추천의 간극
    - 제한된 용량으로 추천 시스템의 후보군이 누락
- LLM의 강력함(지식,일반화)을 지니며 추천 시스템에 적용하기 위해 가벼운 튜닝 프레임워크인 TALLRec을 제안
- TALLRec의 설정
    - LLaMA-7B 모델과 LoRA를 활용
    - few-shot 세팅으로 적은 양으로도 튜닝

## TALLRec
### Preliminary
- instructiong tuning은 LLM의 중요한 요소이며 다음과 같은 요소를 걸침
    - 작업을 정의하고 자연어로 작업 명령문 작성
    - 작업의 입력과 출력을 자연어로 작성하여 정의
    - 튜닝 샘플에 대한 작업 명령문과 작업 입력을 통합하여 명령문 입력을 생성하고 작업 출력에 대한 명령문 출력으로 사용
    - 명령문 입력과 출력의 쌍으로 수행
- TALLRec의 처리 절차
    - 사용자의 과거 상호작용으로 작업 명령문을 Yes or No로 작성하도록 지시
    - 사용자를 긍정, 부정으로 나누고 상호작용은 시간 순서대로, 간단한 텍스트 설명문 포함
    - 작업 명령문과 작업 입력을 통하여 명령문 입력 생성
    - 명령문 출력

### TALLRec Framework
- TALLRec은 2가지의 튜닝(alpaca tuning과 rec-tuning)과 backbone 단계가 존재
    - alpaca tuning : LLM의 일반화 능력을 향상시키는 단계, 제공되는 self-instruct 데이터를 사용
    - rec-tuning : 명령문을 튜닝 패턴을 모방하며, 추천 시스템에 알맞게 튜닝
- 최신 언어 모델은 많은 파라미터를 가지고 있어서 학습에 많은 시간이 소요. 그래서 LoRA를 사용하여 일부 파라미터만 튜닝하여 많은 시간을 줄임
- 데이터의 이슈, 모델의 접근성 등을 고려하여 오픈소스인 LLaMA를 채택

