## 代码说明


### 代码结构
* models目录下包括预训练和微调代码
* third_party参考的第三方开源代码

### 算法描述
* 双流模型
* 文本title、asr、ocr拼接截断256长度 过bert模型, 使用预训练模型  hfl/chinese-macbert-base
* 视频图像桢(预训练抽8桢特征, 微调截断使用14桢保证QPS), 接swin-Transformer base模型和线性成转换
* 视频图像和文本融合, 在encoder层使用LXMERT方式融合，融合后文本和视觉特征concat后接linear层分类。

### 数据
* 仅使用大赛提供的有标注数据（10万）和无标注数据（100万）。
* 未使用任何额外数据。

### 预训练模型
* 使用了 huggingface 上提供的 hfl/chinese-macbert-base 模型。链接为： https://huggingface.co/hfl/chinese-macbert-base

* 使用了 SwinTransformer 官方提供的 swin-base 模型。链接为：https://github.com/microsoft/Swin-Transformer

### 开源代码
* 预训练模型参考了2021-QQ浏览器第一名的代码, 文件`src/third_party/masklm.py`。链接为：git@github.com:zr2021/2021_QQ_AIAC_Tack1_1st.git
* 双流模型参考了开源LXMERT模型, 文件`src/third_party/lxrt.py`。 链接为：git@github.com:airsplay/lxmert.git

### 训练流程
* 预训练： 为了保证速度，预训练两阶段,图像先抽特征。抽完特征直接接线性层与文本特征预训练， 使用MLM、ITM两个预训练任务。
* 微调：与预训练阶段不同，微调图像和文本encoder。

### 测试流程
* 5折中一折数据作为验证集。取验证集上最好的模型来做测试。
