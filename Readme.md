# Effective Transformer

Effective Transformer is built on top of the NVIDIA open sourced project [FasterTransformer](https://github.com/NVIDIA/DeepLearningExamples/tree/master/FasterTransformer/v1) with many advanced optimizations.
Our experiments show Effective Transformer can significantly reduce the execution time and memory consumption, especially for large batch size cases.

## Running BERT without Padding

When using BERT to encode a batch of input sequences, we usually treat the input batch as a matrix whose column number equals to the maximum length of all sequences.
NVIDIA [FasterTransformer](https://github.com/NVIDIA/DeepLearningExamples/tree/master/FasterTransformer/v1) can process cases that all sequences have roughly the same length very efficiently.
However, if the lengths of sequences in the same batch vary a lot, padding them into the same length means a big waste of both memory and computation resources.

Consider the following case

``` python
bert_input = [["Hi"], ["Picking"], ["The", "seed", "of", "Job's", "tears"]]
bert_tokens = [[1], [2], [3,4,5,6,7]]
bert_tokens_padded = [[1, 0, 0, 0, 0], [2, 0, 0, 0, 0], [3, 4, 5, 6, 7]]
bert_tokens_mask = [[1, 0, 0, 0, 0], [1, 0, 0, 0, 0], [1, 1, 1, 1, 1]]
```

this input includes 3 sequences and the maximum length is 5. If we just simply treat it as a 3x5 matrix, only 7 out of 15 values are meaningful.

In Effective Transformer, we still take the input batch as a padded matrix but padding values will be dynamically removed and restored during different calculation stages.

By calculating the prefix sum of the input [mask matrix](https://github.com/google-research/bert/blob/master/modeling.py#L115), we can access real inputs in each sequence in a matrix with no padding values.
The following figure illustrates how to access valid inputs and dynamically remove and restore padding values during the calculation.
All valid inputs are colored in green while padding values are colored in gray.

<img src="./images/1.png" width="50%" height="50%">


## Environment requirements

* CMake >= 3.12
* gcc >= 6
* CUDA 10.0
* Python >= 3.5
* Tensorflow 1.15.x


## Features

* dynamic batch size
* inference with float32 and float16

## Performance

BERT-Base, layers=12, head_num=12, hidden_size=64

Intel(R) Xeon(R) Platinum 8260 CPU @ 2.40GHz

serquence length generated by

``` python
avg_seq_len = np.random.randint(
    low = 2 * avg_seq_len - max_seq_len,
    high = max_seq_len + 1,
    size = (batch_size),
    dtype = np.int32)
```

### Tesla V100, float16, maximum sequence length=32, average serquence length≈20
| batch_size | XLA (in ms)| Faster Transformer (in ms) | Speedup over XLA | Effective Transformer (in ms) | Speedup over XLA
|:-------------:|:-------------:|:---------:|:-----------:|:-----------:|:-----------:|
| 100  | 15.08 | 10.39 | 1.45 | 8.75 | 1.72 |
| 200  | 28.08 | 19.64 | 1.43 | 15.32 | 1.83 |
| 300  | 41.37 | 29.65 | 1.40 | 22.18 | 1.86 |
| 400  | 53.65 | 38.52 | 1.39 | 28.31 | 1.89 |
| 500  | 66.86 | 48.13 | 1.39 | 33.08 | 2.02 |
| 1000  | 131.46 | 95.01 | 1.38 | 64.34 | 2.04 |

### Tesla V100, float16, maximum sequence length=64, average serquence length≈40
| batch_size | XLA (in ms)| Faster Transformer (in ms) | Speedup over XLA | Effective Transformer (in ms) | Speedup over XLA
|:-------------:|:-------------:|:---------:|:-----------:|:-----------:|:-----------:|
| 100  | 28.31 | 20.27 | 1.40 | 16.03 | 1.77 |
| 200  | 54.47 | 40.08 | 1.36 | 30.15 | 1.81 |
| 300  | 80.53 | 59.11 | 1.36 | 41.27 | 1.95 |
| 400  | 106.5 | 78.38 | 1.36 | 54.12 | 1.97 |
| 500  | 132.35 | 98.03 | 1.37 | 65.92 | 2.01 |
| 1000  | 261.18 | 190.91 | 1.38 | 133.61 | 1.95 |

### Tesla V100, float32, maximum sequence length=64, average serquence length≈40
| batch_size | XLA (in ms)| Faster Transformer (in ms) | Speedup over XLA | Effective Transformer (in ms) | Speedup over XLA
|:-------------:|:-------------:|:---------:|:-----------:|:-----------:|:-----------:|
| 100  | 103.13 | 98.52 | 1.05 | 67.45 | 1.53 |
| 200  | 207.40 | 198.86 | 1.04 | 125.44 | 1.65 |
| 300  | 304.99 | 290.55 | 1.05 | 197.07 | 1.55 |
| 400  | 405.98 | 386.04 | 1.05 | 247.39 | 1.64 |
| 500  | 516.88 | 496.90 | 1.04 | 325.37 | 1.59 |

### Tesla T4, float16, maximum sequence length=32, average serquence length≈20
| batch_size | XLA (in ms)| FasterTransformer (in ms) | Speedup over XLA | EffectiveTransformer (in ms) | Speedup over XLA |
|:----------:|:----------:|:---------:|:-----------:|:-----------:|:-----------:|
| 100  | 44.94 | 35.07 | 1.28 | 28.63 | 1.57 |
| 200  | 90.09 | 67.08 | 1.34 | 53.84 | 1.67 |
| 300  | 136.88 | 100.96 | 1.35 | 82.74 | 1.65 |
| 400  | 184.80 | 133.13 | 1.39 | 109.09 | 1.69 |
| 500  | 242.79 | 166.54 | 1.46 | 136.66 | 1.78 |

### Tesla T4, float16, maximum sequence length=64, average serquence length≈40
| batch_size | XLA (in ms)| FasterTransformer (in ms) | Speedup over XLA | EffectiveTransformer (in ms) | Speedup over XLA |
|:----------:|:----------:|:---------:|:-----------:|:-----------:|:-----------:|
| 100  | 87.23 | 65.86 | 1.30 | 52.01 | 1.68 |
| 200  | 176.91 | 138.53 | 1.34 | 108.33 | 1.63 |
| 300  | 261.25 | 204.99 | 1.36 | 157.84 | 1.65 |
| 400  | 355.34 | 272.96 | 1.33 | 202.61 | 1.75 |
| 500  | 452.62 | 343.89 | 1.33 | 250.78 | 1.80 |

## Run demo

Using python prebuilt packege requires python3.5+ tensorflow1.15.x cuda10.0, tested on debian9.

```
$ cd effective_transformer
$ pip install -e python

$ python benchmark.py --help
usage: benchmark.py [-h] [-c CONFIG] [-p {fp32,fp16}] [-b BATCH_SIZE]
                    [-m MAX_SEQ_LENGTH] [-a AVG_SEQ_LENGTH]

Bert performance measuring sample.

optional arguments:
  -h, --help            show this help message and exit
  -c CONFIG, --config CONFIG
                        Bert config file.
  -p {fp32,fp16}, --precision {fp32,fp16}
                        Weight precision.
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        Batch size.
  -m MAX_SEQ_LENGTH, --max_seq_length MAX_SEQ_LENGTH
                        Max sequence length.
  -a AVG_SEQ_LENGTH, --avg_seq_length AVG_SEQ_LENGTH
                        Average sequence length.
```

## Build from source
`TF_PATH : path to libtensorflow_framework.so`
```
$ mkdir build && cd build
$ cmake -DTF_PATH=/your/path/to/pythonx.x/site-packages/tensorflow_core/ ..
$ make
$ cp lib/libtf_effectivetransformer.so ../python/effective_transformer/libtf_effectivetransformer.so.1.15
```
