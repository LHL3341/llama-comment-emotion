模型：Llama-3b

链接：https://github.com/openlm-research/open_llama

模型占用内存较大，可能需要借助服务器。

由于算力平台无法翻墙，故代码中的模型路径为本地路径。若要从huggingface上直接下载，需要修改代码中的 model_path 为 'openlm-research/open_llama_3b'
>
    model_path = 'openlm-research/open_llama_3b'


训练数据集：imdb影评（代码中有）
>   
    dataset = load_dataset("imdb")

目前已经实现英文文本的情感分析，见 demo.ipynb


推特评论数据集

待爬取