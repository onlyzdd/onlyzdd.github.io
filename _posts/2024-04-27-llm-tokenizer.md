---
title: 大模型基础组件——分词器
date: 2024-04-27 00:00:00
author: onlyzdd
categories: [自然语言处理, 分词器, 大模型]
tags: [nlp, tokenizer, llm]
math: true
---

## 为什么分词

将复杂的文本数据分解为机器可管理的语义单元，再结合词嵌入的方式，帮助语言模型理解文本。

### 分词粒度

- Word
  - 原则：基于空格和规则分词
  - 优点：词的边界和含义得以保留
  - 缺点：词表大；稀有词学不好；OOV 问题；无法处理词形关系
- Char：
  - 原则：基于字符分词
  - 优点：词表小；更适合用于中文
  - 缺点：对于英语，无法承载丰富的语义，不利于模型学习；序列太长
- Subword：
  - 原则：常用词不应该被分为更小的子词，但罕见词应该被分解为有意义的子词
  - 优点：较好地平衡词表大小与语义承载能力
  - 缺点：需要学习子词的拆分或合并规则

### 不同任务对分词的要求

- NLU：
  - 任务：文本分类等
  - 模型：BERT 系
  - 分词：能表达语义即可，不要求还原能力，如大小写、重音符号、空格（广义）等不影响语义的可被格式化掉
- NLG：
  - 任务：文本生成等
  - 模型：GPT 系
  - 分词：需能表达语义、无损复原输入，即 Decode(Encode(text)) = text

> 广义空格是指 ` `、`\t`、`\r`、`\n` 等。

### 分词流水线

**1. Normalization**

标准化，对原始输入进行一些处理，使其更干净。常见的操作包括移除重音符号、大小写转换、Unicode 标准化等，而大模型中，一般不执行这些标准化。

**2. Pre-tokenization**

预分词，将输入转换为相对小的单元，再进行分词。例如，根据空格、标点符号先预分词。

**3. Encoder**

编码器，使用模型对预分词的结果进行子词分割。

**4. Post-processing**

后处理，对模型分词的结果做进一步转换。一般使用模板，常见的有 BERT 模板、Chat 模板等。

**5. Decoder**

解码器，将生成模型输出的 ID 序列转换为文本。

> Huggingface Tokenizers 中列举了分词过程中的主要组件，见 [Components][tokenizers-components]。

## 常见子词分割算法

### BPE

#### 算法过程

即 Byte Pair Encoding，最早是一个用来做数据压缩的算法。BPE 算法的目标是学习词表及词的合并规则，该算法将文本作为 Unicode 字符序列，单个字符作为最基础的子词，通过计算相邻的子词 Pair 的词频决定是否应该被合并。

在一次迭代过程中，BPE 向词表中加入一个新词，并添加相关合并规则，合并规则的添加顺序等价于其在编码时的优先级。

在编码阶段，BPE 将输入作为字符序列，然后按照**词表和合并规则**，逐渐合并子词进而得到编码序列。

![BPE token learner](/assets/img/20240427/bpe.png)

> 为以防止跨边界的子词出现，BPE 一般要求将文本先切分成单词。

#### 训练过程示例

1. 语料库：`{"hug": 10, "pug": 5, "pun": 12, "bun": 4, "hugs": 5}`
2. 初始化词表为字符集合：`{"b", "g", "h", "n", "p", "s", "u"}`
3. 如果词表已达预设大小，则退出；否则使用当前词表将语料库划分为符号序列（首次为 `{"h u g": 10, "p u g": 5, "p u n": 12, "b u n": 4, "h u g s": 5}`）
4. 在符号序列中，将两两组合成 Pair，计算出现频次最高的 Pair（首次为 `ug`，出现 20 次）
5. 将频次最高的 Pair 作为符号加入词表中（首次合并后得到 `{"b", "g", "h", "n", "p", "s", "u", "ug"}`），然后进入步骤 3

### BBPE

即 Byte-level BPE。BPE 算法是在字符级别上操作，其会导致无法编码罕见字符。BBPE 的算法思想与 BPE 一致，但将文本作为字节序列，由单字节作为基础的子词进行学习（因此基础词表大小为 256），因而可以用于编码任意 Unicode 字符，并尝试学习字节级的编码特征。

BBPE 最早由 GPT2 提出，其代表模型还包括 RoBERTa、BART 等，其简单实现可以参见 [minbpe][minbpe]。BBPE 一般适用于特殊字符较多、多语言文本的情况下。

### WordPiece

WordPiece 由 Google 提出，用于 BERT 语言模型的分词。其思路与 BPE 类似，区别在于 Pair 的合并策略。BPE 中选择频次最高的 Pair 进行合并，而 WordPiece 使用语言模型来进行考虑。具体地，对子词 $t_1$、$t_2$，WordPiece 考虑合并成 Pair $t_{12}$ 的增益以确定是否合并：

$$
g = \log P(t_{12}) - (\log P(t_1) + \log P(t_2))
$$

WordPiece 的代表模型是 BERT、DistilBERT、MobileBERT、MPNET 等。由于 WordPiece 训练完成后只存储了词表，因此在编码阶段与 BPE 不同，使用词表和**最大匹配**，进而得到编码序列。

### Unigram

Unigram 与以上自底向上的方法不同，该算法首先初始化一个非常大的子词词表 $\mathcal V$，然后逐渐从词表中移除词，直到 $|\mathcal V|$ 达到预设值。

该分词方法基于 Unigram 语言模型，认为当前词的出现不依赖于前面的词，因此子词序列 $\mathbf{x} = (x_1, \cdots, x_M)$ 的概率将表示为 $P(\mathbf{x}) = \prod_{i=1}^{M}p(x_i)$，其中 $\forall{i}, x_i \in \mathcal V, \sum_{x \in {\mathcal{V}}} p(x) = 1$。

对于输入文本 $X$，其最优分割 $\mathbf{x}^*$，即 $\mathbf{x}^* = \underset{\mathbf x\in S(X)}\argmax P(\mathbf x)$，其中 $S(X)$ 为 $X$ 的所有可能分割，最优分割可用 Viterbi 算法求解**最大概率路径**即可。

对于语料库 $\mathcal D$ 和词表 $\mathcal V$，算法通过不断重复以下步骤，以得到最终的词表：

1. 使用 EM 算法学习 Unigram 语言模型
   1. E Step：根据模型参数 $p(x)$ 计算句子分割的条件概率期望
   2. M step：最大化语言模型似然函数 $\mathcal L = \sum_{s \in \mathcal{D}} \log(P(X^{(s)}))$，更新 $p(x)$
2. 对于一个子词 $x_i$，计算 $\mathcal V$ 中移除该子词时 $\mathcal L$ 减少的值，即损失 $loss_i$
3. 根据 $loss_i$ 进行排序，只保留头部 $\eta\%$ 的子词，将其他子词移除掉（当然，单字符的子词是不会被移除的）

在编码阶段，Unigram 允许在分词时加入正则化（即概率分割），对同一输入文本，可以产生多个不同的 Token 序列。具体地：

1. 对于给定文本 $X$，根据概率得到最优的 $l$ 个分割 $P(\mathbf{x} | X)$
2. 从 $l$ 个分割中进行随机采样 $\mathbf{x}_i$：$P(\mathbf{x}_i | X) \cong \cfrac{P(\mathbf{x}_i)^\alpha}{\sum_{i=1}^{l}P(\mathbf{x}_i)^\alpha}$，其中 $\alpha \in \mathbb{R^+}$ 为平滑参数（其倒数即是温度）

细节详见 [Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates][unigram]。Unigram 的代表模型包括 T5、XLNet、Reformer 等。

## SentencePiece 分词库

### 简介

[SentencePiece][sentencepiece] 是一种基于无监督的分词库，包含训练器、编码器和解码器，主要用于基于神经网络的文本生成系统，其中词表大小预先确定。

SentencePiece 将输入文本作为 Unicode 字符序列，在训练和编码时显式地将空格转换为 `▁`（Unicode 表示为 `\u2581`，也被称为 Metaspace），以作为普通字符处理，其功能完备，集标准化、训练、编码、解码等功能于一身。

> SentencePiece 并不是新的分词算法。SentencePiece 也不支持 BBPE，而是基于 BPE，当遇到 OOV 问题时，允许利用字节回退的功能将其编码为字节序列，而不使用 `<unk>` 标记。

### 优势

- 数据驱动：可以基于原始语料进行无监督训练，不依赖于语种中词的概念
- 无损重构：分词结果可以无损还原输入
- 分词算法：支持 BPE、Unigram、word-level、char-level
- 自给自足：完全自给自足的分词工具
- 性能与易用性：使用 C++ 编写，速度快、效率高；提供命令行工具和 Python 接口，便于使用

> SentencePiece 使用 [Protobuf][sentencepiece_model.proto] 作为模型序列化格式，其中包含词表、合并规则，也包含标准化、训练过程中的参数，因此是完全的自给自足的。

### 组件

- Normalizer：对文本进行标准化（大小写、Unicode 标准化等）
- Trainer：加载语料到内存中，从中训练学习词表和合并规则
- Encoder：使用 Normalizer 对输入文本执行标准化，接着分词并产生子词序列
- Decoder：将子词序列转换为文本

### 词表

#### 词的定义

词，语言模型的最小单位，亦是分词模型的基本单位。

- piece：唯一字符串表示，一定是非空的
- id：唯一数值表示，一定是整数
- score：分值，决定子词的合并规则
- type：类型，决定模型处理词时的行为
- _surface_：字符串，用于 Piece 的显示

#### 词的类型

词的类型有 6 种：

|   词的类型   | 枚举值 |   默认符号    | 含义                                                   | 编码时                                 | 解码时                                               |
|:------------:|:------:|:-------------:|:-------------------------------------------------------|:---------------------------------------|:-----------------------------------------------------|
|    NORMAL    |   1    |   自动计算    | 普通符号，即普通的词。对于此类词，score 才是有意义的   | 合并时，使用 score 进行排序            | 替换 Metaspace 为空格                                |
|   UNKNOWN    |   2    |    `<unk>`    | 未知符号，用于表示不在词表中的词                       | 不在词表中的使用该表示                 | 不在词表中的用 ` ⁇ ` 表示                            |
|   CONTROL    |   3    | `<s>`、`</s>` | 控制符号，只为其预留 ID                                | 可能会被切分，因此必须由开发者自行添加 | 空串                                                 |
| USER_DEFINED |   4    |      无       | 用户自定义符号，在任何上下文都保证被处理为单个 Token   | 字典树匹配                             | 不变                                                 |
|     BYTE     |   5    |      无       | 字节符号，当启用 `byte_fallback` 时出现，必定是 256 个 | 对不在词表中的使用字节编码             | 对连续 BYTE 符号使用 UTF-8 解码，失败时使用 `�` 表示 |
|    UNUSED    |   6    |      无       | 不使用的符号                                           | 保证一定不合并成单个 Token             | 不变                                                 |

> 除 NORMAL 外，其他类型的 Piece 统称为 Meta Pieces。因此词表大小的最小值就是 Meta Pieces 的个数和语料库中唯一字符数之和。

### SentencePiece BPE 的训练

|             参数             | 作用                                            | 值类型                                                |  默认值  |
|:----------------------------:|:------------------------------------------------|:------------------------------------------------------|:--------:|
|       `--input_format`       | 指定输入文件的格式                              | string，可选值包括 tsv 和其他值                       |   空串   |
|        `--model_type`        | 指定训练的模型类型                              | string，可选值为 unigram、bpe、word、char             | unigram  |
|        `--vocab_size`        | 词表大小                                        | int32                                                 |   8000   |
|    `--character_coverage`    | 字符总数覆盖率，间接确定词表大小的最小值        | double                                                |  0.9995  |
|   `--split_by_whitespace`    | 是否以空格先分割                                | bool                                                  |   true   |
|     `--split_by_number`      | 是否保证数值与上下文分割开，影响“F1”这种        | bool                                                  |   true   |
|       `--split_digits`       | 是否保证数字（0-9）被单独分割                   | bool                                                  |  false   |
|     `--control_symbols`      | 指定 CONTROL 符号，以逗号分割                   | string                                                |   空串   |
|   `--user_defined_symbols`   | 指定 USER_DEFINED 符号，以逗号分割              | string                                                |   空串   |
|      `--byte_fallback`       | 是否开启字节回退                                | bool                                                  |  false   |
| `--normalization_rule_name`  | 文本标准化规则，对于生成模型，建议使用 identity | string，可选值包括 nmt_nfkc、nmt_nfkc_cf、identity 等 | nmt_nfkc |
|     `--add_dummy_prefix`     | 是否增加虚空格前缀                              | bool                                                  |   true   |
| `--remove_extra_whitespaces` | 是否移除多余的连续空格                          | bool                                                  |   true   |

其他常用参数详见 [Training Options][training_options]。尽管 SentencePiece 提供了 `spm_train`、`spm_encode`、`spm_decode` 相关命令行工具，但开发者更多地使用的是 Python 模块，使用实例见 [Sentencepiece python module][sentencepiece-python]。

### 特殊处理

**Dummy Prefix**

训练时设定 `--add_dummy_prefix` 为 `true`，决定 BOS 位置是否添加 `▁`。但一旦添加，在解码时容易出偏差。

**Math**

训练时设定 `--split_digits` 为 `true` 或定义数字为 USER_DEFINIED 符号，编码时可保证所有数字都被切分。

**Code**

设置 `--remove_extra_whitespaces` 为 `false`，禁止移除多余空格，或增加换行、空格、制表等 USER_DEFINED 的 Token。

### BPE-Dropout

- 背景：使用相同的词汇表对文本可以进行多种分割，但 BPE 将文本分割成独特的序列。这可能会阻止模型更好地学习单词的组成性以及对分割错误的鲁棒性。
- 做法：编码时，对于一次合并，以一定概率取消合并，概率值一般取 0.1。

![BPE Dropout Process](/assets//img/20240427/bpe-dropout.png)

详见 [BPE-Dropout: Simple and Effective Subword Regularization][bpe-dropout]。

## Tiktoken

[Tiktoken](https://github.com/openai/tiktoken) 是由 OpenAI 推出的 BBPE 分词器，由 GPT 系列模型使用。Tiktoken 由 Rust 实现，支持 Python 接口，其虽然是开源的，具备编码、解码功能，但其分词模型的训练部分未开源。另外，[Qwen1.5-0.5B][qwen1.5_0.5b]，[Llama3 8B][llama3-8b] 等一些模型也使用 Tiktoken 作为分词器。

### Tiktoken 与 BBPE 的区别

|                |       BBPE       | Tiktoken                                           |
|:--------------:|:----------------:|:---------------------------------------------------|
|     预分词     |      不支持      | 支持，使用正则在空格等位置进行预分词               |
|    特殊符号    |      不支持      | 支持，通过正则进行匹配                             |
| 预处理和后处理 | UTF-8 编码与解码 | UTF-8 编码与解码中间添加字节与可见字符相互转换的层 |

### Tiktoken 分词器的比较

|  分词器名称   | 对应的语言模型          | 词表                    |
|:-------------:|:------------------------|:------------------------|
|  `r50k_base`  | GPT 2                   | 50,256 及 1 个特殊符号  |
|  `p50k_base`  | Codex 系列              | 50,256 及 1 个特殊符号  |
| `cl100k_base` | GPT 3.5 Turbo、GPT 4 等 | 100,256 及 5 个特殊符号 |

> [Tiktokenizer][tiktokenizer] 是 OpenAI Tiktoken 的一个在线 Playground，能以直观的方式显示分词情况，可用于了解 Token 使用情况。

## 扩展

### 不同语言模型 SentencePiece 分词器的比较

#### 词表区别

| 模型名称                               | 词表大小 | NORMAL  | UNKNOWN | CONTROL | USER_DEFINED | BYTE | UNUSED |
|:---------------------------------------|:--------:|:-------:|:-------:|:-------:|:------------:|:----:|:------:|
| [Baichuan 7B][baichuan-7b]             |  64,000  | 633,51  |    1    |    2    |     390      | 256  |   0    |
| [Baichuan2 7B][baichuan2-7b]           | 125,696  | 124,351 |    1    |    2    |     1086     | 256  |   0    |
| [Llama2 7B][Llama2-7b]                 |  32,000  | 31,741  |    1    |    2    |      0       | 256  |   0    |
| [Chinese Llama2 7B][chinese-Llama2-7b] |  55,296  | 55,037  |    1    |    2    |      0       | 256  |   0    |
| [Mistral 7B][mistral-7b]               |  32,000  | 31,741  |    1    |    2    |      0       | 256  |   0    |
| [Gemma 2B][gemma-2b]                   | 256,000  | 255,495 |    1    |    3    |     245      | 256  |   0    |

#### 编码示例

|                   | 长度 | 内容                                                                                                                                                                                                                                                |
|-------------------|:----:|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 原始句子          |  28  | 人工智能是计算机科学、心理学、哲学等学科融合的交叉学科。                                                                                                                                                                                            |
| Baichuan 7B       |  16  | '▁', '人工智能', '是', '计算机', '科学', '、', '心理学', '、', '哲学', '等', '学科', '融合', '的', '交叉', '学科', '。'                                                                                                                             |
| Baichuan2 7B      |  15  | '人工智能', '是', '计算机', '科学', '、', '心理学', '、', '哲学', '等', '学科', '融合', '的', '交叉', '学科', '。'                                                                                                                                  |
| Llama2 7B         |  35  | '▁', '人', '工', '智', '能', '是', '计', '算', '机', '科', '学', '、', '心', '理', '学', '、', '<0xE5>', '<0x93>', '<0xB2>', '学', '等', '学', '科', '<0xE8>', '<0x9E>', '<0x8D>', '合', '的', '交', '<0xE5>', '<0x8F>', '<0x89>', '学', '科', '。' |
| Chinese Llama2 7B |  16  | '▁', '人工智能', '是', '计算机', '科学', '、', '心理学', '、', '哲学', '等', '学科', '融合', '的', '交叉', '学科', '。'                                                                                                                             |
| Mistral 7B        |  31  | '▁', '人', '工', '智', '能', '是', '计', '算', '机', '科', '学', '、', '心', '理', '学', '、', '<0xE5>', '<0x93>', '<0xB2>', '学', '等', '学', '科', '融', '合', '的', '交', '叉', '学', '科', '。'                                                 |
| Gemma 2B          |  15  | '人工智能', '是', '计算机', '科学', '、', '心理学', '、', '哲学', '等', '学科', '融合', '的', '交叉', '学科', '。'                                                                                                                                  |

由上面的编码示例，可见：

1. Llama2 7B 和 Mistral 7B 分词器对中文的支持性较差，缺少长词，且一些词会被转换为 BYTE 类词
2. Chinese Llama2 基于 Llama2 扩充中文词表，因此编码效率提升
3. Baichuan2 7B 和 Gemma 2B 在编码时，不会在开头增加额外的空格

### SentencePiece 词表扩充——Chinese Llama2

WHY：Llama2 预训练语料的语种主要是英语和少量欧洲语言，原始 Llama 的词表中只有不到 1,000 的中文字符，因此中文能力相对弱。对于未知的中文 UTF-8 字符，尽管 Llama2 分词器可以通过将其转换为字节的方式来编码，但仍会存在多种问题：

1. 中文字符的 UTF-8 序列一般是 3-4 字节，导致编码、解码效率低，且会大幅增加序列长度
2. 字节编码不能很好表达字符的语义特征
3. 在解码时，可能会出现无效的 UTF-8 序列

HOW：为解决 Llama2 分词器编码中文的问题，Chinese Llama2 对其扩充中文词表，具体做法：

1. 在中文语料上训练一个 SentencePiece 模型，新词表大小为 32,000
2. 将新词表与原词表取差集，在原词表后追加差集中的词

合并词表涉及到模型文件的修改，该合并代码详见 [merge_tokenizers.py][chinese_Llama_merge_tokenizers]。

### Huggingface Tokenizers

[Tokenizers][huggingface_tokenizers] 是 Huggingface 提供的开源分词库，其提供了预处理、后处理、分词器训练、编码、解码等功能。Tokenizers 使用 Rust 编写，兼容常用分词算法，提供了 Python 接口，与 Huggingface Transformers 库集成更好、文档更全，定制和扩展更加灵活。

在 Huggingface Transformers 中，分词器一般分为 Slow 和 Fast 两类：

1. Slow：指原始版本实现，一般是纯 Python、SentencePiece、Tiktoken 等
2. Fast：一般指 Tokenizers 版实现，需要兼容不同（并不能保证一定更快）

### Transformers 中 AutoTokenizer 的 Bug

```python
from transformers import GemmaTokenizer, AutoTokenizer

gemma_tokenizer = GemmaTokenizer.from_pretrained("google/gemma-7b")
gemma_auto_tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b")

# CONTROL Symbols
print(gemma_tokenizer.tokenize("<bos>")) # ['<bos>']
print(gemma_auto_tokenizer.tokenize("<bos>")) # ['<bos>']

# USER_DEFINED Symbols
print(gemma_tokenizer.tokenize("<s>")) # ['<s>']
print(gemma_auto_tokenizer.tokenize("<s>")) # ['<', 's', '>']
```

Gemma 模型中该 Bug 于 2024.04.17 被修复，但其他使用 SentencePiece 作为分词器的模型仍会有此类兼容问题。其本质上是 Tokenizers 对 SentencePiece 中 USER_DEFINED 符号兼容的 Bug，关于此类问题的相关信息可参考 [convert_slow_tokenizer.py][convert_slow_tokenizer.py] 和 [AutoTokenizer tokenization issue][autotokenizer_issue]。预计此类 Bug 将在不远的未来被统一解决。

[Llama2-7b]: https://huggingface.co/meta-Llama/Llama-2-7b
[mistral-7b]: https://huggingface.co/mistralai/Mistral-7B-v0.1
[gemma-2b]: https://huggingface.co/google/gemma-2b
[baichuan-7b]: https://huggingface.co/baichuan-inc/Baichuan-7B
[baichuan2-7b]: https://huggingface.co/baichuan-inc/Baichuan2-7B-Base
[chinese-Llama2-7b]: https://huggingface.co/hfl/chinese-Llama-2-7b
[chinese_Llama_merge_tokenizers]: https://github.com/ymcui/Chinese-Llama-Alpaca/blob/602d43ab1fe45113a9d41c038d4d8795182cd72b/scripts/merge_tokenizer/merge_tokenizers.py
[sentencepiece_processor.cc]: https://github.com/google/sentencepiece/blob/7dc9a76ec747f3fe996af4cec6035e75adaff539/src/sentencepiece_processor.cc
[bpe-dropout]: https://arxiv.org/abs/1910.13267
[bpe_model.cc]: https://github.com/google/sentencepiece/blob/7dc9a76ec747f3fe996af4cec6035e75adaff539/src/bpe_model.cc
[training_options]: https://github.com/google/sentencepiece/blob/master/doc/options.md
[bpe_model_trainer.cc]: https://github.com/google/sentencepiece/blob/7dc9a76ec747f3fe996af4cec6035e75adaff539/src/bpe_model_trainer.cc
[sentencepiece]: https://github.com/google/sentencepiece
[autotokenizer_issue]: https://github.com/huggingface/transformers/issues/29440
[convert_slow_tokenizer.py]: https://github.com/huggingface/transformers/blob/main/src/transformers/convert_slow_tokenizer.py
[sentencepiece_model.proto]: https://github.com/google/sentencepiece/blob/master/src/sentencepiece_model.proto
[huggingface_tokenizers]: https://huggingface.co/docs/tokenizers/index
[minbpe]: https://github.com/karpathy/minbpe
[tokenizers-components]: https://huggingface.co/docs/tokenizers/components
[sentencepiece-python]: https://github.com/google/sentencepiece/blob/master/python/sentencepiece_python_module_example.ipynb
[unigram]: https://arxiv.org/abs/1804.10959
[author-badge]: https://img.shields.io/badge/Made_with_%E2%9D%A4%EF%B8%8F_by-onlyzdd-DC3545
[github-home]: https://github.com/onlyzdd
[llama3-8b]: https://huggingface.co/meta-llama/Meta-Llama-3-8B
[qwen1.5_0.5b]: https://huggingface.co/Qwen/Qwen1.5-0.5B
[tiktokenizer]: https://tiktokenizer.vercel.app/
