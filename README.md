## 代码说明

### 环境配置
* Python 版本：3.8.10
* PyTorch 版本：1.10.0
* CUDA 版本：11.3
* transformers 版本: 4.19.2
* 内存： 32G
* GPU :RTX A5000 * 1 , 显存24GB
  所需环境在 `requirements.txt` 中定义。

### 代码结构
* single_stream目录内单流模型，包括预训练和微调代码
* two_stream目录内双流模型。
* third_party参考的第三方开源代码

### 算法描述
* 单流模型和双流模型融合
* 单流模型： 视觉特征和文本特征一起过bert, 文本拼接title，asr，ocr，长度256，bert输出最后4层concat后接linear层分类。
* 双流模型： 视觉特征线性层映射, 文本过bert， 视觉文本特征接一层linear，在encoder层使用LXMERT方式融合，融合后文本和视觉特征concat后接linear层分类。

### 数据
* 仅使用大赛提供的有标注数据（10万）和无标注数据（100万）和测试数据（2.5万)。
* 未使用任何额外数据。

### 预训练模型
* 使用了 huggingface 上提供的 `hfl/chinese-macbert-base` 模型。链接为： https://huggingface.co/hfl/chinese-macbert-base

### 开源代码
* 单流预训练模型参考了2021-QQ浏览器第一名的代码, 文件`src/third_party/masklm.py`。链接为：git@github.com:zr2021/2021_QQ_AIAC_Tack1_1st.git
* 双流模型参考了开源LXMERT模型, 文件`src/third_party/lxrt.py`。 链接为：git@github.com:airsplay/lxmert.git

### 训练流程
* 单流尝试预训练，双流未尝试
* 单双流10折融合。

### 算法结果
* 单流模型： A榜单流不做预训练不加tricks线上0.6531, 加EMA0.6632, 单流尝试预训练单折提升6.5k，10折效果0.6845。
* 双流模型： A榜 未做预训练，单折加EMA0.6670左右，10折0.6840 
* 单双流模型融合： 10折单双流模型融合，线上A榜0.6940。 线上B榜0.693553

### 测试流程
* 10折中一折数据作为验证集。取验证集上最好的模型来做测试。


### 记录
* 新版本EMA, 1.0 加distrill 5折第一折6752
* 2.0 不加distrill  5折第一折 6700
* https://huggingface.co/hfl/chinese-macbert-base/tree/main
* https://huggingface.co/hfl/chinese-roberta-wwm-ext/tree/main
* https://huggingface.co/google/vit-base-patch16-224