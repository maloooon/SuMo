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



    def MeanPooling(self,type, size = 1):
        """

        :param size: pooling size (how many values to look at in one pooling iteration)
        :return: new data tensor with mean pooled values
        :type: defines which input we get : "tensor", "list_tensor"
        """


        if type == 'tensor':
            # Input of type tensor(x,y,z) with x : views, y:samples, z:features
            # iterate through each sample

            # we pool over the views
            mean_values = torch.empty(self.data.size(1), self.data.size(2))
            for sample_idx in range(self.data.size(1)):

                for feature_idx in range(self.data.size(2)):
                    # storing current sample feature values for each view
                    temp = torch.empty(4)
                    for view_idx in range(self.data.size(0)):
                        temp[view_idx] = self.data[view_idx, sample_idx, feature_idx]
                    mean_values[sample_idx, feature_idx] = torch.mean(temp)


            return mean_values

















# TODO : Mixture of experts


    def MaxPooling(self,type, size =  1):
        """
        :param size: pooling size (how many values to look at in one pooling iteration)
        :return: new data tensor with max pooled values
        :type: defines which input we get : "tensor", "list_tensor"
        """


        if type == 'tensor':
        # Input of type tensor(x,y,z) with x : views, y:samples, z:features
        # iterate through each sample

                # we pool over the views
            mean_values = torch.empty(self.data.size(1), self.data.size(2))
            for sample_idx in range(self.data.size(1)):

                for feature_idx in range(self.data.size(2)):
                    # storing current sample feature values for each view
                    temp = torch.empty(4)
                    for view_idx in range(self.data.size(0)):
                        temp[view_idx] = self.data[view_idx, sample_idx, feature_idx]
                    mean_values[sample_idx, feature_idx] = torch.max(temp)


            return mean_values






    def Concatenation(self):
        pass




if __name__ == '__main__':
    a= Pooling(Fs.data_AE_selected_PRAD)
    b= a.MaxPooling('tensor')
  #  c = a.MaxPooling()
    print(b)
  #  print(c)








