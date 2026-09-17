### Initial Training Results

- Model: YOLO26n
- Dataset: Helmet Detection
- Epochs: 30
- Image Size: 640
- Batch Size: 8

- Precision: 0.737
- Recall: 0.49
- mAP50: 0.61
- mAP50-95: 0.38

### Evaluating helment detector

- The class helmet is predicted correctly 20 times
- The class no helmet predicted correctly 11 times
- Class helmet is never confused with no helmet directly
- helmet was treated as background 13 times
- no helmet was treated as background 18 times
- 7 times backgorund objects were predicted as helmet and 5 times background was predicted as no helmet
yes no helmet is comparatively weaker 