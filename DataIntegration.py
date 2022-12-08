from statistics import mean
import torch
import FeatureSelection as Fs

"""Data integration methods
   Pooling only works if we have the same latent space for each
   view (since we pool together the value of latent features of each
   view by sample)"""



class Pooling():
    """Mean-Pooling, Max Pooling
    data input : list of lists of tensors with latent features for each
    sample for each view"""
    def __init__(self, data,**kwargs):
        self.data = data

        assert all(
            self.data[0].size(1) == x.size(1) for x in self.data
        ), "Size mismatch between latent features of views"

        self.n_samples = self.data[0].size(0)

        print(self.n_samples)



    def MeanPooling(self,size = 1):
        """

        :param size: pooling size (how many values to look at in one pooling iteration)                                 # TODO : hier nochmal sozusagen feature reduction via höhere size ?
        :return: new data tensor with mean pooled values
        """
        # TODO : Schleife ersetzen, torch funktion (max/mean)
        # ONLY implemented for size = 1
        mean_pooled_data = []
        mean_pool_for_curr_sample = []

        sample_counter = 0
        # Mean value over each latent feature of each view for each sample
        # Results in tensor of basically one view of
        while sample_counter < self.n_samples :
            for feature in range(self.data[0].size(1)):
                temp = [] # holds feature values of each view for current sample in iteration
                for view in self.data:
                    temp.append((view[sample_counter][feature]).item())


                mean_pool_for_curr_sample.append(mean(temp))
            mean_pooled_data.append(mean_pool_for_curr_sample)
            mean_pool_for_curr_sample = []
            sample_counter += 1

        mean_pooled_data = torch.tensor(mean_pooled_data)

        return mean_pooled_data














# TODO : Mixture of experts


    def MaxPooling(self, size =  1):
        """
        :param size: pooling size (how many values to look at in one pooling iteration)
        :return: new data tensor with max pooled values
        """
        # ONLY implemented for size = 1
        max_pooled_data = []
        max_pool_for_curr_sample = []

        sample_counter = 0
        # Mean value over each latent feature of each view for each sample
        # Results in tensor of basically one view of
        while sample_counter < self.n_samples :
            for feature in range(self.data[0].size(1)):
                temp = [] # holds feature values of each view for current sample in iteration
                for view in self.data:
                    temp.append((view[sample_counter][feature]).item())


                max_pool_for_curr_sample.append(max(temp))
            max_pooled_data.append(max_pool_for_curr_sample)
            max_pool_for_curr_sample = []
            sample_counter += 1

        max_pooled_data = torch.tensor(max_pooled_data)

        return max_pooled_data






    def Concatenation(self):
        pass




if __name__ == '__main__':
    a= Pooling(Fs.AE_all_compressed_features)
    b= a.MeanPooling()
    c = a.MaxPooling()
    print(b)
    print(c)








