File:           README.md

Description: This project explores the use of computer vision techniques—applied to OHLC (Open, High, Low, Close) candlestick images—to predict stock price movements. By generating and classifying OHLC image windows, the pipeline aims to compare traditional models like logistic regression with deep learning models such as ResNet. The goal is to assess whether visual representations of price data can provide predictive signals, particularly over a rolling window of `180` trading days.

Author:         Morgan Cooper
Created:        2025-09-09
Updated:        2025-09-09

#######################################################
#######################################################

Logistic Regression Baseline Results: 

Test accuracy                           51.0%

TEST BINNING STATS (sorted by confidence):
   bin_pctile  avg_proba  accuracy  num_total  num_won
0        100%   0.707896  0.475410         61       29
1         95%   0.621833  0.573770         61       35
2         90%   0.596748  0.540984         61       33
3         85%   0.578408  0.655738         61       40
4         80%   0.562541  0.475410         61       29
5         75%   0.551194  0.540984         61       33
6         70%   0.541677  0.557377         61       34
7         65%   0.533391  0.459016         61       28
8         60%   0.523076  0.639344         61       39
9         55%   0.514067  0.557377         61       34
10        50%   0.506228  0.491803         61       30
11        45%   0.497628  0.409836         61       39
12        40%   0.487158  0.540984         61       28
13        35%   0.477051  0.459016         61       33
14        30%   0.465191  0.491803         61       31
15        25%   0.451894  0.459016         61       33
16        20%   0.436488  0.409836         61       36
17        15%   0.410594  0.491803         61       31
18        10%   0.376550  0.524590         61       29
19         5%   0.286116  0.508197         61       30

#######################################################
#######################################################

ResNet18 Image Classifier Results:

--- first 3 layers frozen ---

EPOCH: 1
  [batch   178] train loss: 0.708
  Validation loss: 0.708, acc: 51.31%

EPOCH: 2
  [batch   178] train loss: 0.654
  Validation loss: 0.713, acc: 51.72%

EPOCH: 3
  [batch   178] train loss: 0.610
  Validation loss: 0.725, acc: 50.98%

EPOCH: 4
  [batch   178] train loss: 0.554
  Validation loss: 0.745, acc: 50.98%

EPOCH: 5
  [batch   178] train loss: 0.480
  Validation loss: 0.788, acc: 50.82%
  
Test loss: 0.757, acc: 51.31%

TEST BINNING STATS (sorted by confidence):
   bin_pctile  avg_proba  accuracy  num_total  num_won
0        100%   0.882781  0.622951         61       38
1         95%   0.828487  0.573770         61       35
2         90%   0.787654  0.672131         61       41
3         85%   0.751530  0.524590         61       32
4         80%   0.727821  0.508197         61       31
5         75%   0.703885  0.573770         61       35
6         70%   0.682702  0.491803         61       30
7         65%   0.663052  0.524590         61       32
8         60%   0.640611  0.377049         61       23
9         55%   0.619378  0.524590         61       32
10        50%   0.593092  0.540984         61       33
11        45%   0.565520  0.491803         61       30
12        40%   0.540797  0.639344         61       39
13        35%   0.513276  0.475410         61       29
14        30%   0.488912  0.393443         61       31
15        25%   0.459321  0.475410         61       32
16        20%   0.423791  0.606557         61       24
17        15%   0.378673  0.409836         61       36
18        10%   0.336442  0.426230         61       35
19         5%   0.251519  0.409836         61       36

#######################################################
#######################################################

ResNet18 Image Classifier Results:

--- first 2 layers frozen ---

EPOCH: 1
  [batch   178] train loss: 0.709
  Validation loss: 0.698, acc: 52.54%

EPOCH: 2
  [batch   178] train loss: 0.640
  Validation loss: 0.705, acc: 51.80%

EPOCH: 3
  [batch   178] train loss: 0.574
  Validation loss: 0.724, acc: 52.62%

EPOCH: 4
  [batch   178] train loss: 0.479
  Validation loss: 0.744, acc: 51.89%

EPOCH: 5
  [batch   178] train loss: 0.351
  Validation loss: 0.809, acc: 52.46%

Test loss: 0.789, acc: 51.89%

TEST BINNING STATS (sorted by confidence):
   bin_pctile  avg_proba  accuracy  num_total  num_won
0        100%   0.912854  0.573770         61       35
1         95%   0.853400  0.573770         61       35
2         90%   0.815630  0.508197         61       31
3         85%   0.786635  0.590164         61       36
4         80%   0.753566  0.491803         61       30
5         75%   0.721030  0.491803         61       30
6         70%   0.684107  0.491803         61       30
7         65%   0.656040  0.606557         61       37
8         60%   0.624148  0.655738         61       40
9         55%   0.589702  0.475410         61       29
10        50%   0.555693  0.524590         61       32
11        45%   0.526524  0.557377         61       34
12        40%   0.497380  0.459016         61       34
13        35%   0.454656  0.409836         61       36
14        30%   0.416937  0.442623         61       34
15        25%   0.377146  0.540984         61       28
16        20%   0.335147  0.393443         61       37
17        15%   0.278175  0.573770         61       26
18        10%   0.217075  0.540984         61       28
19         5%   0.129057  0.475410         61       32

#######################################################
#######################################################

ResNet18 Image Classifier Results:

--- first layer frozen ---

EPOCH: 1
  [batch   178] train loss: 0.708
  Validation loss: 0.703, acc: 52.30%

EPOCH: 2
  [batch   178] train loss: 0.632
  Validation loss: 0.713, acc: 52.21%

EPOCH: 3
  [batch   178] train loss: 0.553
  Validation loss: 0.749, acc: 51.89%

EPOCH: 4
  [batch   178] train loss: 0.450
  Validation loss: 0.768, acc: 52.87%

EPOCH: 5
  [batch   178] train loss: 0.312
  Validation loss: 0.832, acc: 52.38%

Test loss: 0.842, acc: 51.39%


TEST BINNING STATS (sorted by confidence):
   bin_pctile  avg_proba  accuracy  num_total  num_won
0        100%   0.938317  0.639344         61       39
1         95%   0.877873  0.491803         61       30
2         90%   0.827877  0.491803         61       30
3         85%   0.792608  0.573770         61       35
4         80%   0.751660  0.573770         61       35
5         75%   0.715475  0.508197         61       31
6         70%   0.681837  0.557377         61       34
7         65%   0.641285  0.590164         61       36
8         60%   0.599447  0.622951         61       38
9         55%   0.564294  0.524590         61       32
10        50%   0.530017  0.475410         61       29
11        45%   0.495278  0.426230         61       29
12        40%   0.451542  0.524590         61       29
13        35%   0.405662  0.442623         61       34
14        30%   0.366279  0.508197         61       30
15        25%   0.324904  0.557377         61       27
16        20%   0.275897  0.459016         61       33
17        15%   0.226221  0.459016         61       33
18        10%   0.164919  0.524590         61       29
19         5%   0.097220  0.327869         61       41

#######################################################
#######################################################

ResNet50 Image Classifier Results: 

--- first 3 layers frozen ---

EPOCH: 1
  [batch   178] train loss: 0.694
  Validation loss: 0.691, acc: 51.97%

EPOCH: 2
  [batch   178] train loss: 0.680
  Validation loss: 0.691, acc: 51.56%

EPOCH: 3
  [batch   178] train loss: 0.663
  Validation loss: 0.692, acc: 53.61%

EPOCH: 4
  [batch   178] train loss: 0.634
  Validation loss: 0.697, acc: 52.38%

EPOCH: 5
  [batch   178] train loss: 0.574
  Validation loss: 0.711, acc: 53.85%

Test loss: 0.739, acc: 47.05%

TEST BINNING STATS (sorted by confidence):
   bin_pctile  avg_proba  accuracy  num_total  num_won
0        100%   0.774065  0.540984         61       33
1         95%   0.667172  0.409836         61       25
2         90%   0.627836  0.475410         61       29
3         85%   0.592171  0.639344         61       39
4         80%   0.563529  0.557377         61       34
5         75%   0.543927  0.377049         61       23
6         70%   0.525875  0.491803         61       30
7         65%   0.508698  0.573770         61       35
8         60%   0.494187  0.491803         61       31
9         55%   0.479026  0.377049         61       38
10        50%   0.459196  0.426230         61       35
11        45%   0.444234  0.393443         61       37
12        40%   0.427253  0.426230         61       35
13        35%   0.410426  0.426230         61       35
14        30%   0.392691  0.442623         61       34
15        25%   0.372954  0.377049         61       38
16        20%   0.353904  0.442623         61       34
17        15%   0.333429  0.475410         61       32
18        10%   0.299902  0.426230         61       35
19         5%   0.242984  0.639344         61       22