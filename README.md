# Kaist-pedestrian-detection
- ssd-halfway 방식 시도
- conv4_3에서의 feature map을 합쳐서 학습
- 근데 rgb image와 thermal image가 같은 layer를 사용하도록 코드를 작성하였습니다. (학습을 돌린 이후에나 발견했습니다....정말 제 자신이 이해가 가지 않는군요...? 반성합니다. 미래의 저는 이런 실수하면 싸다구 맞아야 합니다.)
- rgb, thermal가 다른 layer를 사용하는 코드는 branch 4_3을 확인해주세요
- 근데 역시나 4_3처럼 성능이 개개개개개개개후집니다. (엉엉엉)
