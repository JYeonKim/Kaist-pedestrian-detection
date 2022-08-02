# Kaist-pedestrian-detection
- ssd-h 최종본입니다.
- detect_rgb_thermal.py의 코드가 조금 변경 필요하지만 현재는 괜찮을거 같네요
간단히 분석도 적어봅니다.
![image](https://user-images.githubusercontent.com/46176600/182305978-3576c9a1-c707-4aa9-b045-036da2e57efd.png)
- 정량적으로는 앞전과 비교했을 때, night 성능이 전체적으로 올랐습니다. thermal보다 더 오를것이라고는 생각 못했는데 (rgb에서 night가 굉장히 안좋기 때문에 rgb와 합치면 thermal의 17보다는 낮게 나올 것이라 생각했음) 14로 3더 성능이 좋게 나왔습니다. day도 전체적으로 높게 나왔는데 thermal와 rgb 모두 원래는 34~35퍼 였는데 지금은 29퍼로 약 5퍼정도 성능이 향샹되었습니다.

![image](https://user-images.githubusercontent.com/46176600/182306063-ec7ee283-9055-410f-9017-88a3285921a5.png)
- 예전에 detect하지 못한 것도 지금은 잘합니다.

![image](https://user-images.githubusercontent.com/46176600/182306176-4524f07f-4b24-47f7-93fb-b5c830a63d7e.png)
- 하지만 여전히 pedestrian이 아닌 것도 detect하는 모습을 보입니다.

![image](https://user-images.githubusercontent.com/46176600/182306230-b0c2b129-8549-4c2e-a0eb-8768f6b7d04c.png)
- 사람들이 많을 때도 detect을 잘하지만, 그렇지 않은 때도 있습니다. 

![image](https://user-images.githubusercontent.com/46176600/182306282-2454fb6a-4349-4a06-8cce-22be22694b1b.png)
![image](https://user-images.githubusercontent.com/46176600/182306312-605dc37b-adc9-4b3a-b76d-9a272f193f04.png)
- 'A SSD-based Crowded Pedestrian Detection Method' 논문을 통해 해결 방법을 참고하고자 합니다.

- 참고 이미지는 논문 이미지 중 하나 입니다.
![image](https://user-images.githubusercontent.com/46176600/182306617-c64810ec-29e2-4709-9359-836c94c1901b.png)
