# Contrastive Learning for Sequential Recommendation

## ABSTRACT
- 순차 추천은 과거의 기록을 통해 동적으로 변화하는 관심도를 추정할 수 있음에 큰 강점을 보인다.
- 그러나, 순차 예측 작업은 많은 파라미터를 동반해야 해서 데이터가 sparse하면 학습에 어려움을 가진다
- CL4SRec은 위와 같은 문제를 해결하기 위해 CV에서 유망한 학습방법인 contrastive learning을 선택하였다.
- CL4SRec은 전통적인 순서 예측과 contrastive learning 프레임워크를 사용하여 self-supervision을 도출하여 사용자 패턴을 기존보다 더 잘 잡음
- 또한, data augmentation을 위해 3가지 접근법을 사용하여 4가지의 데이터 셋으로 검증을 하였다.

## I. INTRODUCT
- 사용자의 과거 상호작용을 바탕으로 추론하는 것은 순차 추천의 핵심 문제
- 최근 이러한 문제를 위하여 deep-learning을 사용하였으며 향상을 불러왔지만, sparse한 데이터에서는 약하다는 단점을 지님
- 최근, self-supervised learning이 CV와 NLP에서 화두가 되며 이는 label이 없는 데이터에서 데이터간의 상관성을 추출하는 것
- 이에 영감을 받아서 순차 추천시스템에도 적용한 것이 CL4SRec
- 사용자 모델을 학습하기 위해 GPT와 같은 강력한 모델을 사용할 수 있지만 2가지 이유로 거절
    - RecSys는 사전학습을 위한 데이터가 없음
    - 기존 예측 기반 self-supervised는 학습과 목표의 데이터가 같기 때문에 추천에는 도움이 안됨
- 위 언급된 문제때문에 추천 시스템에서는 연구가 진행되지 않고 item level의 표현 향상에 사용됨
- 기존 연구는 item의 특성으로 item을 표현하는것에 반해, ID를 이용한 유저 행동으로 더 나은 순차 예측을 진행
- CL4SRec는 contrastive learning loss를 통해 동일 사용자에 대한 latent space를 극대화하여 사용자를 표현
- CL4SRec은 오직 사용자의 상호작용 기반으로 사용자를 표현함
- cropping, masking, reordering으로 사용자의 다른 관점을 추측

## II. RELATED WORK
### 2.1 Sequential Recommendation
- 