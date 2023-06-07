import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk import pos_tag
from collections import Counter

# nltk.download('averaged_perceptron_tagger')

text = """
This is a simple test. This is a simple example to show how to extract keywords from a piece of text using Python. 
Python is a powerful programming language.
"""

# 分词
tokens = word_tokenize(text)

# 将所有单词转换为小写，以避免大小写差异
tokens = [token.lower() for token in tokens]

# 移除停用词（如 "is", "a", "this" 等常见但不重要的词）
stop_words = set(stopwords.words('english'))
tokens = [token for token in tokens if token.isalpha() and token not in stop_words]

# 词性标注
tagged = pos_tag(tokens)

# 保留名词和动词
filtered_tokens = [word for word, pos in tagged if pos.startswith('NN') or pos.startswith('VB')]

# 计算频率
freq_dist = FreqDist(filtered_tokens)

# 输出前10个最常见的单词
for word, freq in freq_dist.most_common(10):
    print(word, freq)
