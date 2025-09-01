import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_ActivityIndex(component_list, activities, figfile='./activity_index.png', selection=None, labels=None):

    if selection is not None:
        activities = [activities[i] for i in range(len(activities)) if component_list[i] in selection]
        component_list = [component_list[i] for i in range(len(component_list)) if component_list[i] in selection]
    if labels is not None:
        component_list = labels
    # sort the data
    group, labels = zip(*sorted(zip(activities, component_list),reverse=True))
    # calculate the aggregated activity index
    activity = np.array(group)
    activity = activity/np.sum(activity)*100
    acum_activity = np.cumsum(activity)
    fig, ax = plt.subplots()

    ax.plot(labels,acum_activity, marker='o',label='Cumulative activity index')
    ax.plot(labels,activity, marker='*',label='Activity index')
    ax.set_ylabel('Activity index (%)')
    ax.set_xlabel('Components')
    ax.set_title('Sorted activity indices and cumulative index')
    ax.legend()
    ax.grid()
    plt.show()
    # save the figure
    fig.savefig(figfile, dpi=600)

if __name__ == "__main__":

    datapath_='./data/'
    figfile= datapath_+'activity_R.png'

    filename_1_ = datapath_+'report_task_Terkildsen_NaK_BG_Fig5EA_summary.csv'

    # read json files to dictionary
    df_1 = pd.read_csv(filename_1_)
    # get the species
    component_list = df_1['Component'].values.tolist()
    group1 = df_1['Activity'].values.tolist()
    # select R1 to R15
    selection=['R1','R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9', 'R10', 'R11', 'R12', 'R13', 'R14', 'R15']
    plot_ActivityIndex(component_list, group1, figfile, selection=selection, labels=None)
    # select P1 to P15
    figfile= datapath_+'activity_P.png'
    selection=['P1','P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10', 'P11', 'P12', 'P13', 'P14', 'P15']
    plot_ActivityIndex(component_list, group1, figfile, selection=selection, labels=None)
