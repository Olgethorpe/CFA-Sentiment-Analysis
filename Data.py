from Vectorizers import Vectorizers
from pandas import read_csv

# Stores the data matricies
class Data:
    # Object initalization
    # Args   : datafile: A filename that contains raw data
    # Returns: The Data object
    def __init__(self, datafile):
        self.Raw = datafile
        
    # Stores the vectorized count matrix of a unigram data model
    @property
    def UnigramMatrix(self):
        if not hasattr(self, '__unigramMatrix'):
            matrix = {}         
            matrix['X'] = Vectorizers.Unigram.value.fit_transform(self.Raw.X)
            matrix['Y'] = self.Raw.Y
            self.__countMatrix = matrix
        return self.__countMatrix
    
    # Stores the vectorized count matrix of a bigram data model
    @property
    def BigramMatrix(self):
        if not hasattr(self, '__bigramMatrix'):
            matrix = {}         
            matrix['X'] = Vectorizers.Bigram.value.fit_transform(self.Raw.X)
            matrix['Y'] = self.Raw.Y
            self.__bigramMatrix = matrix
        return self.__bigramMatrix
    
    # Stores the vectorized td-idf matrix
    @property
    def Tf_idfMatrix(self):
        if not hasattr(self, '__tfidfMatrix'):
            matrix = {}         
            matrix['X'] = Vectorizers.Tf_idf.value.fit_transform(self.Raw.X)
            matrix['Y'] = self.Raw.Y
            self.__tfidfMatrix = matrix
        return self.__tfidfMatrix
    
    # Stores the dataframe with the raw data
    @property
    def Raw(self):
        return self.__raw
    
    @Raw.setter
    def Raw(self, datafile):
        data = read_csv(datafile) 
        if data.shape[1] != 2:
            raise ValueError('Your raw datafile must have 2 columns.')
        data.columns = ['X', 'Y']
        self.__raw = data