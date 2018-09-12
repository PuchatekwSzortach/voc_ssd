# voc_ssd
Single Shot MultiBox Detector implementation for PASCAL VOC dataset

Work in progress. Please come again later.

#### Notes on theoretical fields of views of different VGG layers
- Block 1-1 - 3x3
- Block 1-2 - 5x5
- Block 1-pooling - 6x6

- Block 2-1 - 10x10
- Block 2-2 - 14x14
- Block 2-pooling - 16x16

- Block 3-1 - 24x24
- Block 3-2 - 32x32
- Block 3-3 - 40x40
- Block 3-pooling - 44x44

#### Sample fields of view for different layers of VGG network

Sample fields of views after Block 1-pooling:  
| 1 - 6 | 3 - 8 | 5 - 10 | 7 - 12 | 9 - 14 | 11 - 16 | 13 - 18 |

Sample fields of views after Block 2-pooling:  
| 1 - 16 | 5 - 20 | 9 - 24 | 13 - 28 | 17 - 32 | 21 - 36 | 25 - 40 |

Sample fields of views after Block 3-pooling:  
| 1 - 44 | 9 - 52 | 17 - 60 | 25 - 68 | 33 - 76 | 41 - 84 | 49 - 92 |

Sample fields of views after Block 4-pooling:  
| 1 - 100 | 17 - 116 | 33 - 132 | 49 - 148 | 65 - 164 |

Sample fields of views after Block 5-pooling:  
| 1 - 212 | 33 - 224 | 65 - 276 | 97 - 308 | 129 - 340 |



#### Sample fields of view for different layers of ours VGGish SSD network.
Assumes each prediction head is made of two 3x3 convolutions.

Sample fields of view after Block 2-head - based on Block 2-pooling:  
| 1 - 32 | 5 - 36 | 9 - 40 | 13 - 44 | 17 - 40 | 21 - 52 |  
Single cell field of view: 32x32

Sample fields of view after Block 3-head - based on Block 3-pooling:  
| 1 - 76 | 9 - 84 | 17 - 92 | 25 - 100 | 33 - 108 | 41 - 116 |  
Single cell field of view: 76x76

Sample fields of view after Block 4-head - based on Block 4-pooling:  
| 1 - 164 | 17 - 180 | 33 - 196 | 49 - 212 | 65 - 228 |  
Single cell field of view: 164x164

Sample fields of view after Block 5-head - based on Block 5-pooling:  
| 1 - 340 | 33 - 372 | 65 - 404 | 97 - 436 | 129 - 468 |  
Single cell field of view: 340x340