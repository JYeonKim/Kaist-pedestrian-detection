# Kaist-pedestrian-detection
- conv4_3까지 하나의 layer로 rgb, thermal를 학습했기 때문에 
- 새롭게 thermal을 위한 layer를 init 해주었고 그 layer를 가지고 학습을 해주었음
- 역시나 weight는 pretrained된 weight, bais를 가지도록 넣어주었음 (rgb layer와 thermal layer의 weight가 같도록 초기화 해줌)
- 근데 성능이 개후지게 나옴....에러도 발생함! (도대체 왜 엉엉ㅜㅠㅠㅠ) (저는 꼼꼼하게 공부하고 있지 않은가 봅니다...반성합니다,.......)
