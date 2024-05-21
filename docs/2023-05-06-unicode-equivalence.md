# Unicode 等价性与正规化

## 等价性与正规化

Unicode 等价性（Unicode equivalence）是指 Unicode 中包含一些特殊字符，这使得不同的字符序列可能具有相同的功能（如显示等）。Unicode 等价性包括标准等价和兼容等价。而 Unicode 正规化（Unicode normalization）是指根据 Unicode 等价性对字符序列进行转换的过程，其分为分解与合成两类。

### 标准等价和兼容等价

标准等价（canonical equivalence）是指字符序列在显示或打印时具有相同的视觉显示。例如字符 `ô` 可以用序列 `\u00f4` 表示，也可以用序列 `\u006f\u0302` 表示，这两个序列的视觉显示相同，因此是标准等价的。

兼容等价（compability equivalence）是指字符序列可能有不同的视觉显示，但可能保持相同的语义。例如字符 `㍢` 与 `10点` 的显示不同，但语义是一致的，这两个序列是兼容等价的。

兼容等价是对标准等价的扩展，标准等价的字符序列也是兼容等价的。

### 合成与分解

合成（Composition）与分解（Decomposiition）指使用标准等价或兼容等价规则对字符串进行分解和合成。

- 分解：根据等价性规则，一个 Unicode 字符可能会被分解成多个 Unicode 字符
- 合成：根据等价性规则，先执行分解，之后多个 Unicode 字符可能会被组合成一个 Unicode 字符

### 正规化的四种形式

| 正规化形式                                             | 说明                                                         |
| ------------------------------------------------------ | ------------------------------------------------------------ |
| NFD（Normalization Form Canonical Decomposition）      | 标准等价分解：将字符串以标准等价的形式进行分解                 |
| NFC（Normalization Form Canonical Composition）        | 标准等价合成：将字符串以标准等价的形式进行分解和合成           |
| NFKD（Normalization Form Compatibility Decomposition） | 兼容等价分解：将字符串以兼容等价的形式进行分解                 |
| NFKC（Normalization Form Compatibility Composition）   | 兼容等价合成：将字符以兼容等价的形式进行分解，再以标准等价的形式进行合成 |

## 使用 Python 进行 Unicode 正规化

Python 中提供的 `unicodedata` 模块中的 `normalize` 方法可以对 Unicode 字符串进行正规化。

```python
>>> import unicodedata
>>> s1 = 'é'
>>> s2 = unicodedata.normalize('NFD', s1)      # 标准等价分解：结果字符串中有两个 Unicode 字符！
>>> s2                                          # => 'é'
>>> len(s1) == len(s2)                          # => False
```

在进行文本处理时，一般要对字符串进行兼容等价合成：

```python
>>> import unicodedata
>>> unicodedata.normalize('NFKC', '㍢')				  # => 10点
```
