import lifelines
import DataInputNew
import matplotlib.pyplot as plt
from lifelines.utils import median_survival_times


def KaplanMeierCurve(duration,event):
    """Plotting simple KaplanMeierCurve based on duration & event only.
       duration and event are of type pandas df
       For seeing the plot, use Jupyter-Notebook"""
    kmf = lifelines.KaplanMeierFitter()
    kmf.fit(duration,event)
    kmf.plot_survival_function() # survival plot
    kmf.plot_cumulative_density() # failure plot
    median = kmf.median_survival_time_ # median of survival times # TODO : for all events ?
    median_confidence_interval = median_survival_times(kmf.confidence_interval_) # 95% confidence interval for median

    return median, median_confidence_interval



if __name__ == '__main__':
    data = DataInputNew.data_PRAD
    feat_offsets = DataInputNew.feat_offsets_PRAD
    duration_df = data.iloc[:, feat_offsets[5]]
    event_df = data.iloc[:, feat_offsets[4]]
    m, m_ci = KaplanMeierCurve(duration=duration_df, event=event_df)

    print(m,m_ci)





