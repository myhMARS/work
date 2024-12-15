import pyLDAvis.lda_model
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Step 1: 加载数据集
newsgroups = fetch_20newsgroups(remove=('headers', 'footers', 'quotes'))
docs_raw = newsgroups.data  # 提取原始文档数据
print(len(docs_raw))  # 打印文档数量，共计 11314 篇文档

# Step 2: 数据预处理
# 使用 CountVectorizer 将文本数据转化为词频矩阵 (document-term matrix, DTM)。
# 参数说明：
# - `strip_accents='unicode'`：去除文本中的重音符号。
# - `stop_words='english'`：移除英语停用词，例如 "the", "and"。
# - `lowercase=True`：将所有单词转化为小写。
# - `token_pattern=r'\b[a-zA-Z]{3,}\b'`：仅提取长度至少为 3 的英文字母单词。
# - `max_df=0.5`：过滤掉在超过 50% 文档中出现的高频词。
# - `min_df=10`：过滤掉在少于 10 篇文档中出现的低频词。
tf_vectorizer = CountVectorizer(strip_accents='unicode',
                                stop_words='english',
                                lowercase=True,
                                token_pattern=r'\b[a-zA-Z]{3,}\b',
                                max_df=0.5,
                                min_df=10)

# 将文本数据转化为词频矩阵
dtm_tf = tf_vectorizer.fit_transform(docs_raw)
# 结果为一个稀疏矩阵，行表示文档，列表示词汇，值是词频。

# Step 3: 主题模型 - LDA
# 使用 Latent Dirichlet Allocation (LDA) 对文本进行主题建模。
# 参数说明：
# - `n_components=20`：将文档划分为 20 个主题。
# - `random_state=0`：固定随机种子以确保结果可重复。
lda_tf = LatentDirichletAllocation(n_components=20, random_state=0)
lda_tf.fit(dtm_tf)
# 模型会学习词语的主题分布和文档的主题分布。

# Step 4: 可视化
# 使用 pyLDAvis 对 LDA 模型结果进行交互式可视化。
# `pyLDAvis.lda_model.prepare` 函数生成 LDA 可视化数据。
# 参数说明：
# - `lda_tf`：训练好的 LDA 模型。
# - `dtm_tf`：文档-词项矩阵。
# - `tf_vectorizer`：用于生成矩阵的 CountVectorizer。
pic = pyLDAvis.lda_model.prepare(lda_tf, dtm_tf, tf_vectorizer)

# 将可视化结果保存为 HTML 文件，便于浏览和分析。
pyLDAvis.save_html(pic, 'lda_model1.html')
print("LDA 可视化已保存为 lda_model1.html")
