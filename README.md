# voc_ssd
Single Shot MultiBox Detector implementation for PASCAL VOC dataset

Work in progress. Please come again later.

#### Notes on theoretical fields of views of different VGG layers
- Block 1-1 - 3x3
- Block 1-2 - 5x5
- Block 1-pooling - 6x6

- Block 2-1 - 10x10
- Block 2-2 - 14x14
- Block 2-pooling - 15x15

- Block 3-1 - 24x24
- Block 3-2 - 32x32
- Block 3-3 - 39x39
- Block 3-pooling - 44x44

Sample fields of views after Block 3-pooling:  
| 1 - 44 | 9 - 52 | 17 - 60 | 25 - 68 | 33 - 76 | 41 - 84 | 49 - 92 |
