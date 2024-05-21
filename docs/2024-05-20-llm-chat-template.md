# 对话模型的模板

## 简介

由于语言模型只接受非结构化的文本作为输入，为了方便模型处理多轮的对话消息列表，衍生出对话模板（chat template）。对话模板的目的是用于将消息列表转换为符合模型要求的文本字符串，这种有特殊标记的字符串格式被称为 Chat Markup Language（缩写为 ChatML）。

一般而言，消息列表是字典的列表，字典中通常包含字段：`role` 字段用于描述角色类型，可选值一般为 system、user、assistant；`content` 字段用于表示消息的内容。

在转换为 ChatML 时，其中的字段可能使用特殊的字符串对消息的开始和结束、角色名等进行标记。通常将采用 `<|im_start|>{role}\n{content}<|im_end|>` 表示一条消息的，认为是标准 ChatML 格式。

在 Transformers 中，使用 [Jinja][jinja-doc] 模板引擎进行消息列表到标记字符串的转换。一般而言，对于在 `tokenizer_config.json` 文件中声明了 `chat_template` 字段的模型（通常是 Chat 或 Instruct 模型），可以显式地调用 `tokenizer.apply_chat_template` 以方便地将消息列表转换为标记字符串或模型输入。

## Llama-2-7B-Chat-HF

来自 [Llama-2-7b-chat-hf/tokenizer_config.json][llama2-hf-chat-tpl]，对应的 Python 转换代码可以参考 [llama/generation.py][llama-generation]。

```jinja
{% if messages[0]['role'] == 'system' %}
    {% set loop_messages = messages[1:] %}
    {% set system_message = messages[0]['content'] %}
{% else %}
{% set loop_messages = messages %}
    {% set system_message = false %}
{% endif %}
{% for message in loop_messages %}
    {% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}
        {{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}
    {% endif %}
    {% if loop.index0 == 0 and system_message != false %}
        {% set content = '<<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n' + message['content'] %}
    {% else %}
        {% set content = message['content'] %}
    {% endif %}
    {% if message['role'] == 'user' %}
        {{ bos_token + '[INST] ' + content.strip() + ' [/INST]' }}
    {% elif message['role'] == 'assistant' %}
        {{ ' '  + content.strip() + ' ' + eos_token }}
    {% endif %}
{% endfor %}
```

## Mistral-7B-Instruct-v0.1

来自 [Mistral-7B-Instruct-v0.1/tokenizer_config.json][mistral-instruct-v0.1-chat-tpl]。

```jinja
{{ bos_token }}
{% for message in messages %}
    {% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}
        {{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}
    {% endif %}
    {% if message['role'] == 'user' %}
        {{ '[INST] ' + message['content'] + ' [/INST]' }}
    {% elif message['role'] == 'assistant' %}
        {{ message['content'] + eos_token + ' ' }}
    {% else %}
        {{ raise_exception('Only user and assistant roles are supported!') }}
    {% endif %}
{% endfor %}
```

## Qwen-1.5-1.8B-Chat

来自 [Qwen1.5-1.8B-Chat/tokenizer_config.json][qwen-1.5-chat-tpl]，其使用的是标准的 ChatML 格式。

```jinja
{% for message in messages %}
    {% if loop.first and messages[0]['role'] != 'system' %}
        {{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}
    {% endif %}
    {{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}
{% endfor %}
{% if add_generation_prompt %}
    {{ '<|im_start|>assistant\n' }}
{% endif %}
```

> [!TIP]
> 如果消息列表中缺少 system 消息，Qwen-1.5-1.8B-Chat 的模板会自动添加一个，其内容为 `You are a helpful assistant.`。

## Gemma-1.1-2B-IT

来自 [gemma-1.1-2b-it/tokenizer_config.json][gemma-1.1-it-chat-tpl]。

```jinja
{{ bos_token }}
{% if messages[0]['role'] == 'system' %}
    {{ raise_exception('System role not supported') }}
{% endif %}
{% for message in messages %}
    {% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}
        {{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}
    {% endif %}
    {% if (message['role'] == 'assistant') %}
        {% set role = 'model' %}
    {% else %}
        {% set role = message['role'] %}
    {% endif %}
    {{ '<start_of_turn>' + role + '\n' + message['content'] | trim + '<end_of_turn>\n' }}
{% endfor %}
{% if add_generation_prompt %}
    {{'<start_of_turn>model\n'}}
{% endif %}
```

> [!NOTE]
> Gemma-1.1-2B-IT 不支持 system 消息，此外 assistant 消息在内部会被重命名为 model 消息。

## Llama3-8B-Instruct

来自 [Meta-Llama-3-8B-Instruct/tokenizer_config.json][llama3-instruct-chat-tpl]。

```jinja
{% set loop_messages = messages %}
{% for message in loop_messages %}
    {% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}
    {% if loop.index0 == 0 %}
        {% set content = bos_token + content %}
    {% endif %}
    {{ content }}
{% endfor %}
{% if add_generation_prompt %}
    {{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}
{% endif %}
```

另可参考官方实现 [llama3/tokenizer.py][llama3-tokenizer]。

## 对话模板的对比

|         模型名称         |              分词器类型              | 模板使用的特殊 Token                                                                                                                  | 模板使用的非特殊 Token                                                 | 支持的角色                                                | 是否支持推理 |
|:------------------------:|:------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------|:----------------------------------------------------------|:------------:|
|    Llama-2-7B-Chat-HF    |            SentencePiece             | &lt;s&gt;、&lt;/s&gt;                                                                                                                 | &lt;&lt;SYS&gt;&gt;、&lt;&lt;/SYS&gt;&gt;、&lt;INST&gt;、&lt;/INST&gt; | system、user、assistant                                   |      否      |
| Mistral-7B-Instruct-v0.1 |            SentencePiece             | &lt;s&gt;、&lt;/s&gt;                                                                                                                 | &lt;&lt;SYS&gt;&gt;、&lt;&lt;/SYS&gt;&gt;、&lt;INST&gt;、&lt;/INST&gt; | system（模板不支持）、user、assistant                     |      否      |
|     Gemma-1.1-2B-IT      |            SentencePiece             | &lt;bos&gt;、&lt;start_of_turn&gt;、&lt;end_of_turn&gt;                                                                               | &lt;&lt;SYS&gt;&gt;、&lt;&lt;/SYS&gt;&gt;、&lt;INST&gt;、&lt;/INST&gt; | system（模板不支持）、user、assistant（内部转换为 model） |      是      |
|    Qwen-1.5-1.8B-Chat    |               Tiktoken               | &lt;&#124;im_start&#124;&gt;、&lt;&#124;im_end&#124;&gt;                                                                              | -                                                                      | system、user、assistant                                   |      是      |
|    Llama3-8B-Instruct    | Tokenizers (converted from Tiktoken) | &lt;&#124;begin_of_text&#124;&gt;、&lt;&#124;start_header_id&#124;&gt;、&lt;&#124;end_header_id&#124;&gt;、&lt;&#124;eot_id&#124;&gt; | -                                                                      | system、user、assistant                                   |      是      |

> [!TIP]
> “是否支持推理”是指模板是否支持 `add_generation_prompt` 开关，以在推理时自动添加 assistant 后缀。

## 其他注意事项

1. 训练和推理阶段使用的转换方法必须相同，以避免任何不一致的情况。
2. 在推理时可以使用模板，但在训练阶段，输入部分的标签需要进行掩码（即包含不参与 Loss 计算的部分），因此必须熟悉模板对应的 Python 代码。
3. 为使分词器更好地工作，所有的输入字段 `message['content']`，都应该去除首尾空格（广义空格，即包括空格、换行等）。对话模板一般会通过声明 `trim` 过滤器自动去除空格，代码版直接调用 Python 中的 `strip()` 方法即可。
4. 在推理时使用模板，如果用户输入中包含了特殊 Token，那么可能会出现注入风险，使模型产生意外的行为。

[mistral-instruct-v0.1-chat-tpl]: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1/blob/main/tokenizer_config.json#L32
[gemma-1.1-it-chat-tpl]: https://huggingface.co/google/gemma-1.1-2b-it/blob/main/tokenizer_config.json#L1507
[llama3-instruct-chat-tpl]: https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct/blob/main/tokenizer_config.json#L2053
[llama3-tokenizer]: https://github.com/meta-llama/llama3/blob/14aab0428d3ec3a9596f1dea06d9c564f9c0e35f/llama/tokenizer.py#L202
[qwen-1.5-chat-tpl]: https://huggingface.co/Qwen/Qwen1.5-1.8B-Chat/blob/main/tokenizer_config.json#L31
[llama-generation]: https://github.com/meta-llama/llama/blob/be327c427cc5e89cc1d3ab3d3fec4484df771245/llama/generation.py#L284
[jinja-doc]: https://jinja.palletsprojects.com/
[llama2-hf-chat-tpl]: https://huggingface.co/meta-llama/Llama-2-7b-chat-hf/blob/main/tokenizer_config.json#L12
[author-badge]: https://img.shields.io/badge/Made_with_%E2%9D%A4%EF%B8%8F_by-onlyzdd-DC3545
