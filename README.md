# [Naver AI Hackathon 2018](https://github.com/naver/ai-hackathon-2018)

### **Team kozistr - Member : 김형찬(kozistr)**

---

## tl;dr

I participated in [Naver A.I Hackathon 2018] and 
**ranked 4th/13th(over 200 teams total)** as an individual participant (Team : kozistr)

And also i uploaded **summary docs** with **the codes**.

### Final LeaderBoard

> 네이버 지식iN 질문 유사도 예측 (결선)

![kin_leaderboard](_images/kin_final_lb.jpg)

> 네이버 영화 평점 예측 (결선)

![movie_leaderboard](_images/movie_final_lb.jpg)

### Result

*Stage* | *Mission*    | *Metric* | *Score* | *Rank*
:-----: | :----------: | :------: | :-----: | :---:
phase 1 | kin          | acc      |         |
phase 1 | movie-review | mse      |         |
phase 2 | kin          | acc      |         |
phase 2 | movie-review | mse      |         |
 final  | kin          | acc      | 0.8115  | 4th
 final  | movie-review | mse      | 0.0310  | 13th

## Models

Soon~

## Summary!

Here's **summary docs**! ![Summary](_refs/kozistr-naver_ai_hackathon_2018_report.pdf)


## Hyper-Parameters

### 네이버 지식iN 질문 유사도 예측

|           Name            |   Value        |          Note                 |
| :-----------------------: | :------------: | :---------------------------: |
|          Epochs           |      100       | 70 ~ 80 에서 converge          |
|       Learning Rate       |      1e-3      | exponential decay (rate 0.95) |
|        Batch Size         |     64/128     | 본선에서는 128                  |
|       DropOut Rate        |      0.7       | 0.7 is the best               |
|      Char Embedding       |      378       | 378 ~ 400 good                |
|      CNN kernel size      | 10, 9, 7, 5, 3 | 10 이하에서 찾음                |
|      CNN filter size      |      256       | 256 ~ good                    |
|         FC Unit           |     1024       | 1024 good                    |
|        Optimizer          |     Adam       | Adam, NAdam, SGD ~            |
...

### 네이버 영화 평점 예측

|           Name            |   Value        |          Note                 |
| :-----------------------: | :------------: | :---------------------------: |
|          Epochs           |      30        | 20 ~ 30 에서 converge          |
|       Learning Rate       |     2e-4       | lr 에 엄청나게 민감             |
|        Batch Size         |      128       | 128 ~                         |
|       DropOut Rate        |      0.6       | 0.6 is the best               |
|      Char Embedding       |      128       | 128 ~ 256 good                |
|      CNN kernel size      |    3, 5, 7     | 10 이하에서 찾음                |
|      CNN filter size      |      256       | 256 ~ good                    |
|         FC Unit           |      512       | 512 good                      |
|        Optimizer          |     Adam       | Adam, SGD ~                   |
...

## Author

HyeongChan Kim ([@kozistr](http://kozistr.tech), kozistr@gmail.com)
