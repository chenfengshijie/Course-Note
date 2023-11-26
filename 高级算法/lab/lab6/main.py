import nltk
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag

# 如果是首次使用NLTK，你需要下载一些资源
nltk.set_proxy("http://127.0.0.1:7890")
# nltk.download("punkt", download_dir="./nltk_data/")
# nltk.download("stopwords", download_dir="./nltk_data/")
# nltk.download("averaged_perceptron_tagger", download_dir="./nltk_data/")
# 这是要分析的文本
text = """
This is a simple test. This is a simple example to show how to extract keywords from a piece of text using Python. 
Python is a powerful programming language.
"""

# 分词
nltk.data.path.append("./nltk_data/")
tokens = word_tokenize(text, language="english")

# 将所有单词转换为小写，以避免大小写差异
tokens = [token.lower() for token in tokens]

# 移除停用词（如 "is", "a", "this" 等常见但不重要的词）
stop_words = set(stopwords.words("english"))
tokens = [token for token in tokens if token.isalpha() and token not in stop_words]

tagged_tokens = pos_tag(tokens)

filtered_tokens = [token for token, pos in tagged_tokens if pos.startswith("NN")]
# 计算频率
freq_dist = FreqDist(filtered_tokens)

# 输出前10个最常见的单词
for word, freq in freq_dist.most_common(10):
    print(word, freq)
