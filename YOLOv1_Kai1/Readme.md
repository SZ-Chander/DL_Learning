# YOLO v1-Kaiについて
## 説明
簡単にいうと、そのコードは、YOLO v1アルゴリズムの再現です。YOLO v1の論文は「[You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640)」です。  
この再現コードは、100％のYOLO v1ではなく、分類器の部分はResNet34となりました。しかし、メインの部分は変わらないです。
## 使い方
### Train
　Trainとは、モデルを訓練することです。  
　[Train.py](Train.py)のmain部分に、*setupPath*変数の值をご希望のJsonファイルのPathに変更して、そのまま運行すればいいです。  
　訓練は結構計算量や時間をかかるので、GPUの利用は推奨です。GPUの使い方は、*device*を"cuda:0"又は"mps"などに設定することです。

**_＊Jsonの書き方と各変量の意味は、変量名をご参考ください。わからない場合ではissuesでご提出ください。_**
### Test
　Testとは、訓練済みのモデルの性能をテストすることです。
　[Test.py](Test.py)のmain部分に、テストの目標分類を*cls*で順番に設定して、*val_dataset*でテストリストを設定して、最後は*model.load_state_dict*のところにテストしたいのモデルのPathを設定して、そのままで使えるです。
### Pred
　Predとは、訓練済みのモデルを利用して、画像を推測することです。  
　使い方は、上記のTestとほぼ一緒です。ただし、コードは[Pred.py](Pred.py)になるです。そして、インプットのはリストではなっく、その代わりに、*imgPath*で具体な画像のPathを設定して使えるです。