import numpy as np
import hashlib


class Naive:
    """
    使用naive方法实现集合的相似性连接
    """

    def __init__(self):
        pass

    @staticmethod
    def intersection(list1, list2):
        sum = 0
        for element1 in list1:
            for element2 in list2:
                if element1 == element2:
                    sum += 1
        return sum

    def run(self, corpus, c: float):
        ans = set()
        for i in range(len(corpus)-1):
            for j in range(i+1, len(corpus)):
                inter = self.intersection(corpus[i], corpus[j])  # 交集大小
                union = len(corpus[i]) + len(corpus[j]) - inter  # 并集大小
                if inter >= c * union:
                    ans.add((i, j))
        return ans


class MinHash:
    def __init__(self, b, r):
        self.b = b
        self.r = r

    @staticmethod
    def singleHash(data, n):
        seq = [i for i in range(n)]
        result = [n for _ in range(len(data))]
        seq = np.random.permutation(seq)
        for i in range(len(result)):
            minhash = n
            for element in data[i]:
                if seq[element] < minhash:
                    result[i] = element
                    minhash = seq[element]
        return result

    def sigMatrix(self, data, n_elements, n_hash_funcs):
        """
        计算sigMatrix
        """
        result = []
        for i in range(n_hash_funcs):
            single = self.singleHash(data, n_elements)
            result.append(single)
        return np.array(result)

    def minHash(self, data, n_elements):
        """
        LSH算法
        """
        hashBuckets = {}
        n = self.b * self.r

        sigMatrix = self.sigMatrix(data, n_elements, n)

        begin, end = 0, self.r
        count = 0

        while end <= sigMatrix.shape[0]:
            count += 1
            for colNum in range(sigMatrix.shape[1]):
                # generate the hash object
                hashObj = hashlib.md5()

                # calculate the hash value
                band = str(sigMatrix[begin: begin +
                           self.r, colNum]) + str(count)
                hashObj.update(band.encode())
                tag = hashObj.hexdigest()

                # update the dictionary
                if tag not in hashBuckets:
                    hashBuckets[tag] = [colNum]
                elif colNum not in hashBuckets[tag]:
                    hashBuckets[tag].append(colNum)
            begin += self.r
            end += self.r

        return hashBuckets

    def run(self, data, n):
        hashBucket = self.minHash(data, n)

        result = [set() for _ in range(len(data))]

        for key in hashBucket:
            for col1 in hashBucket[key]:
                for col2 in hashBucket[key]:
                    if col1 == col2:
                        break
                    result[col1].add(col2)

        sum = 0
        for s in result:
            sum += len(s)
        return sum
