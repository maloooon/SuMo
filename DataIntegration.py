"""Data integration methods"""



class Pooling():
    """Mean-Pooling, Max Pooling"""
    def __init__(self, data,**kwargs):
        self.data = data


    def MeanPooling(self, type,size):
        """

        :param type: pool by view ( pool different values of different views together to find a mean) or by sample
        (pool different values of the same view, but different samples together to find a mean)
        :param size: pooling size (how many values to look at in one pooling iteration)
        :return: new data tensor with mean pooled values
        """
        if type == "byview":
            pass
        else: # "bysample"
            pass



    def MaxPooling(self, type, size):
        """
        :param type: pool by view ( pool different values of different views together to find a maximum) or by sample
        (pool different values of the same view, but different samples together to find a maximum)
        :param size: pooling size (how many values to look at in one pooling iteration)
        :return: new data tensor with max pooled values
        """
        if type == "byview":
            pass
        else: # "bysample"
            pass

    def Concatenation(self):
        pass









