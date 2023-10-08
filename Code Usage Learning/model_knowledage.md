# Model Knowledge

## Sklearn  
### 文本类别转码 LabelEncoder/OneHotEncoder  
数据预处理之将类别数据数字化的方法 —— LabelEncoder VS OneHotEncoder  
该方法用于类别的映射，比如现在有[a,b,c,d]这几类，通过这些函数转为0/1。  
https://zhuanlan.zhihu.com/p/33569866 


### 文本向量化 CountVectorizer/TfidfVectorizer
文本向量化主要目的是将一串文本转化为一个矩阵。 

比如以下文本
> "I have a dog."   
> "You have a dog and a cat."   
> "He books a book."   
> "No cost too great."

将单词拆分转为词典  
> {'have': 7, 'dog': 5, 'you': 11, 'and': 0, 'cat': 3, 'he': 8, 'books': 2, 'book': 1, 'no': 9, 'cost': 4, 'too': 10, 'great': 6}

CountVectorizer是用0/1代表这个词是否出现过。  
TfidfVectorizer是用TF-IDF值代表这个词的重要程度。   
具体可参考[此文档](https://welts.xyz/2022/03/26/sklearn_text/)

Note: TF-IDF算法，相关原理详见后文文档相关介绍。   


## TF-IDF算法
TF-IDF(term frequency–inverse document frequency)是一种用于信息检索与数据挖掘的常用加权技术，常用于挖掘文章中的关键词，而且算法简单高效，常被工业用于最开始的文本数据清洗。TF-IDF有两层意思，一层是"词频"（Term Frequency，缩写为TF），另一层是"逆文档频率"（Inverse Document Frequency，缩写为IDF）。

> 假设我们现在有一片长文叫做《量化系统架构设计》词频高在文章中往往是停用词，“的”，“是”，“了”等，这些在文档中最常见但对结果毫无帮助、需要过滤掉的词，用TF可以统计到这些停用词并把它们过滤。当高频词过滤后就只需考虑剩下的有实际意义的词。

> 但这样又会遇到了另一个问题，我们可能发现"量化"、"系统"、"架构"这三个词的出现次数一样多。这是不是意味着，作为关键词，它们的重要性是一样的？事实上系统应该在其他文章比较常见，所以在关键词排序上，“量化”和“架构”应该排在“系统”前面，这个时候就需要IDF，IDF会给常见的词较小的权重，它的大小与一个词的常见程度成反比。

> 当有TF(词频)和IDF(逆文档频率)后，将这两个词相乘，就能得到一个词的TF-IDF的值。某个词在文章中的TF-IDF越大，那么一般而言这个词在这篇文章的重要性会越高，所以通过计算文章中各个词的TF-IDF，由大到小排序，排在最前面的几个词，就是该文章的关键词。

相关原理详见[文档相关介绍](https://zhuanlan.zhihu.com/p/31197209)。
