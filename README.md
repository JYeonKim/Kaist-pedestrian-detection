# Kaist-pedestrian-detection
- 성능이 이전보다 좋게 나오지 않아서, 어디에서 정확히 성능이 잘 안나오는지 분석 해봤습니다
- 우선 이전보다 훨씬더 사람이 아닌 object를 더 잡는 경향이 있습니다 (FP가 너무 높습니다. 영상으로 확인하면 이전보다 많다는 것을 체감할 수 있습니다)
- crowded한 pedestrian을 잘 detect하지 못합니다. (제가 논문의 방법을 잘못 적용한 것일까요?)
![image](https://user-images.githubusercontent.com/46176600/182750376-e6195464-dbdc-4c9b-bee3-8c6ce2a941b6.png)
- 이전에 crowded한 것과 관련해서 잘 detect하지 못한 이미지인데, 여전히 잘 detect하지 못하는 것을 확인할 수 있습니다. 분명히 문제가 있군요
