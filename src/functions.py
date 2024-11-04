import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import PredictionErrorDisplay


def plotting_raw_data(X, y, alpha, s, color, grid, axes, plot_columns_x_label, label, label_true):

    # First subplot
    axes[0].scatter(y, X.iloc[:, 0], alpha=alpha, s=s, c=color, label=label)
    axes[0].set_ylabel(plot_columns_x_label[0])
    if grid:
        axes[0].grid(linestyle = 'dotted', linewidth = 0.3)
    if label_true:
        axes[0].legend(loc='upper right')

    # Second subplot
    axes[1].scatter(y, X.iloc[:, 1], alpha=alpha, s=s, c=color, label=label)
    axes[1].set_ylabel(plot_columns_x_label[1])
    if grid:
        axes[1].grid(linestyle = 'dotted', linewidth = 0.3)
    if label_true:
        axes[1].legend(loc='upper right')

    # Third subplot
    axes[2].scatter(y, X.iloc[:, 2], alpha=alpha, s=s, c=color, label=label)
    axes[2].set_ylabel(plot_columns_x_label[2])
    if label_true:
        axes[2].legend(loc='upper right')
    if grid:
        axes[2].grid(linestyle = 'dotted', linewidth = 0.3)

    # # Fourth subplot
    axes[3].scatter(y, X.iloc[:, 3], alpha=alpha, s=s, c=color, label=label)
    axes[3].set_xlabel(plot_columns_x_label[4])
    axes[3].set_ylabel(plot_columns_x_label[3])
    if label_true:
        axes[3].legend(loc='upper right')
    if grid:
        axes[3].grid(linestyle = 'dotted', linewidth = 0.3)

    # Adjust layout to prevent overlapping
    plt.tight_layout()




def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 2 * IQR
    upper_bound = Q3 + 2 * IQR

    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]



def error_plot(figsize, y_true, y_pred, title):
    # Define plot structure
    fig, axs = plt.subplots(ncols=1, figsize=figsize, dpi=500)

    # Create an instance of PredictionErrorDisplay
    ped = PredictionErrorDisplay.from_predictions(y_true=y_true,
                                                  y_pred=y_pred,
                                                  kind="actual_vs_predicted",
                                                  #subsample=1000,
                                                  ax=axs,
                                                  random_state=0)

    # Set the x and y labels of the PredictionErrorDisplay plot
    ped.ax_.set_xlabel("Predicted $v_s$ (m/s)")  # Set x label
    ped.ax_.set_ylabel("Actual $v_s$ (m/s)")  # Set y label
    ped.ax_.set_title(title)  # Set title

    # Add grid
    ped.ax_.grid()



def plot_cpt_data(figsize, plot_columns_x, df_raw, df_SCPTu_SCPT, id_value, plot_columns_x_label, sort_column):

    fig, axes = plt.subplots(1, len(plot_columns_x)-1, figsize=figsize, dpi=500, sharey=True)

    df_id = df_raw.loc[df_raw.loc[:,'ID'] == id_value]
    df_id = df_id.sort_values([sort_column])
    for i, column in enumerate(plot_columns_x[1:-1]):
        axes[i].plot(df_id[column].values,
                      df_id[plot_columns_x[0]].values,
                      label=f'Raw data (CPT ID {id_value})',
                      marker='o', color='k', linewidth=0.2, markersize=0.8)
        axes[i].set_ylim(ymin=0)
        axes[i].set_xlim(xmin=0)

    df_id = df_SCPTu_SCPT.loc[df_raw.loc[:,'ID'] == id_value]
    df_id = df_id.sort_values([sort_column])
    for i, column in enumerate(plot_columns_x[1:-1]):
        axes[i].plot(df_id[column+"_mean"].values,
                      df_id[plot_columns_x[0]].values,
                      label=f'Data with moving average (CPT ID {id_value})',
                      marker='o', color='r', linewidth=0.2, markersize=0.8)


        axes[0].set_ylabel(plot_columns_x_label[0])
        axes[i].set_xlabel(plot_columns_x_label[i+1])
        axes[i].grid(True, which='both', linestyle = 'dotted', linewidth = 0.3)
        axes[i].minorticks_on()



    # Use a different variable for the last subplot
    last_subplot_label = plot_columns_x_label[-1]
    axes[-1].plot(df_id[plot_columns_x[-1]].values,
                  df_id[plot_columns_x[0]].values,
                  label=f'Raw data (CPT ID {id_value})',
                  marker='o', color='k', linewidth=0.2, markersize=0.8)
    axes[-1].set_xlabel(last_subplot_label)

    axes[-1].set_xlim(xmin=0)
    axes[-1].grid(True, which='both', linestyle = 'dotted', linewidth = 0.3)
    axes[-1].minorticks_on()
    axes[-1].invert_yaxis()

    return fig, axes



def plot_cpt_data_raw(figsize, plot_columns_x, df_raw, id_value, plot_columns_x_label, sort_column):

    fig, axes = plt.subplots(1, len(plot_columns_x)-1, figsize=figsize, dpi=500, sharey=True)

    df_id = df_raw.loc[df_raw.loc[:,'ID'] == id_value]
    df_id = df_id.sort_values([sort_column])
    for i, column in enumerate(plot_columns_x[1:-1]):
        axes[i].plot(df_id[column].values,
                      df_id[plot_columns_x[0]].values,
                      label=f'Raw data (CPT ID {id_value})',
                      marker='o', color='k', linewidth=0.2, markersize=0.8)
        axes[i].set_ylim(ymin=0)
        axes[i].set_xlim(xmin=0)
        axes[0].set_ylabel(plot_columns_x_label[0])
        axes[i].set_xlabel(plot_columns_x_label[i+1])
        axes[i].grid(True, which='both', linestyle = 'dotted', linewidth = 0.3)
        axes[i].minorticks_on()

    # Use a different variable for the last subplot
    last_subplot_label = plot_columns_x_label[-1]
    axes[-1].plot(df_id[plot_columns_x[-1]].values,
                  df_id[plot_columns_x[0]].values,
                  label=f'Raw data (CPT ID {id_value})',
                  marker='o', color='k', linewidth=0.2, markersize=0.8)
    axes[-1].set_xlabel(last_subplot_label)

    axes[-1].set_xlim(xmin=0)
    axes[-1].grid(True, which='both', linestyle = 'dotted', linewidth = 0.3)
    axes[-1].minorticks_on()
    axes[-1].invert_yaxis()

    return fig, axes




def plot_cpt_data_NW_site(figsize, plot_columns_x, df_site, df_smoothed, df_proccessed, y_true, y_pred, plot_columns_x_label):

    fig, axes = plt.subplots(1, len(plot_columns_x)-1, figsize=figsize, dpi=500, sharey=True)

    for i, column in enumerate(plot_columns_x[1:-1]):
        axes[i].plot(df_site[column].values,
                      df_site[plot_columns_x[0]].values,
                      label='Raw data',
                      marker='o', color='k', linewidth = 0.2, markersize=2)

        axes[i].set_ylim(ymin=0, ymax = 25)
        axes[i].set_xlim(xmin=0)

        axes[i].set_xlabel(plot_columns_x_label[i+1])
        axes[i].grid(True, which='both', linestyle = 'dotted', linewidth = 0.3)
        axes[i].minorticks_on()


    for i, column in enumerate(plot_columns_x[1:-1]):
        axes[i].plot(df_proccessed[column].values,
                      df_proccessed[plot_columns_x[0]].values,
                      label='Input ML',
                      marker='o', color='r', linewidth = 0.5, markersize=2)

        axes[i].legend(loc='lower center')

    axes[0].set_ylabel('Depth [m]')

    axes[-1].plot(y_true,
                  df_proccessed[plot_columns_x[0]].values,
                  label='Raw data',
                  marker='o', color='k', linewidth = 0.5, markersize=2)
    axes[-1].plot(y_pred,
                df_proccessed[plot_columns_x[0]].values,
                label='Output ML',
                marker='o', color='blue', linewidth = 0.5, markersize=2)

    axes[-1].set_xlabel(plot_columns_x_label[-1])
    axes[-1].grid(True, which='both', linestyle = 'dotted', linewidth = 0.3)
    axes[-1].legend(loc='lower center', handlelength=2, labelspacing=1)
    axes[-1].minorticks_on()
    axes[-1].invert_yaxis()
    axes[-1].set_xlim(xmin=0)

    plt.tight_layout()
    return fig, axes

def plot_cpt_data_NW_site_all_splittedVS(figsize, plot_columns_x, df_site, y_true_df, y_pred, plot_columns_x_label):

    fig, axes = plt.subplots(1, len(plot_columns_x)-1, figsize=figsize, dpi=500, sharey=True)

    for i, column in enumerate(plot_columns_x[1:-1]):
        axes[i].plot(df_site[column].values,
                      df_site[plot_columns_x[0]].values,
                      label='Raw data',
                      marker='o', color='k', linewidth = 0.2, markersize=0.8)

        axes[i].set_ylim(ymin=0, ymax = 10)
        axes[i].set_xlim(xmin=0)

        axes[i].set_xlabel(plot_columns_x_label[i+1])
        axes[i].grid(True, which='both', linestyle = 'dotted', linewidth = 0.3)
        axes[i].legend(loc='lower center', handlelength=1.5, labelspacing=0.3)
        axes[i].minorticks_on()

    axes[0].set_ylabel('Depth [m]')

    axes[-1].plot(y_pred,
                df_site[plot_columns_x[0]].values,
                label='ML output',
                color='blue', linewidth = 0.6, markersize=0.0)
    
    axes[-1].plot(y_true_df.iloc[:,1].values,
                  y_true_df.iloc[:,0].values,
                  label='Raw data',
                  marker='o', color='r', linewidth = 0.6, markersize=0.8)


    axes[-1].set_xlabel(plot_columns_x_label[-1])
    axes[-1].grid(True, which='both', linestyle = 'dotted', linewidth = 0.3)
    axes[-1].legend(loc='lower center', handlelength=1.5, labelspacing=0.3, frameon=True, bbox_to_anchor=(0.5, -0.02))

    axes[-1].minorticks_on()
    axes[-1].invert_yaxis()
    axes[-1].set_xlim(xmin=0)


    plt.subplots_adjust(wspace=0.0, hspace=0)
    plt.tight_layout()

    return fig, axes



def plot_cpt_data_NW_site_all(figsize, plot_columns_x, df_site, df_proccessed, y_true, y_pred, plot_columns_x_label):
    fig, axes = plt.subplots(1, len(plot_columns_x)-1, figsize=figsize, dpi=500, sharey=True)

    for i, column in enumerate(plot_columns_x[1:-1]):
        axes[i].plot(df_site[column].values,
                      df_site[plot_columns_x[0]].values,
                      label='Raw data',
                      marker='o', color='k', linewidth = 0.2, markersize=0.8)

        axes[i].set_ylim(ymin=0, ymax = 25)
        axes[i].set_xlim(xmin=0)

        axes[i].set_xlabel(plot_columns_x_label[i+1])
        axes[i].grid(True, which='both', linestyle = 'dotted', linewidth = 0.3)
        axes[i].legend(loc='lower center', handlelength=1.5, labelspacing=0.3)
        axes[i].minorticks_on()


    axes[0].set_ylabel('Depth [m]')

    axes[-1].plot(y_pred,
                df_site[plot_columns_x[0]].values,
                label='ML output',
                color='blue', linewidth = 0.6, markersize=0.0)
    axes[-1].plot(y_true,
                  df_proccessed[plot_columns_x[0]].values,
                  label='Raw data',
                  marker='o', color='r', linewidth = 0.6, markersize=0.8)


    axes[-1].set_xlabel(plot_columns_x_label[-1])
    axes[-1].grid(True, which='both', linestyle = 'dotted', linewidth = 0.3)
    axes[-1].legend(loc='lower center', handlelength=1.5, labelspacing=0.3, frameon=True, bbox_to_anchor=(0.5, -0.02))

    axes[-1].minorticks_on()
    axes[-1].invert_yaxis()
    axes[-1].set_xlim(xmin=0)


    plt.subplots_adjust(wspace=0.0, hspace=0)
    plt.tight_layout()

    return fig, axes

def plot_cpt_data_NW_tests(figsize, plot_columns_x, df_site, y_pred, plot_columns_x_label, y_true=None):
    fig, axes = plt.subplots(1, len(plot_columns_x)-1, figsize=figsize, dpi=500, sharey=True)

    for i, column in enumerate(plot_columns_x[1:-1]):
        axes[i].plot(df_site[column].values,
                      df_site[plot_columns_x[0]].values,
                      label='Raw data',
                      marker='o', color='k', linewidth = 0.2, markersize=0.8)

        axes[i].set_ylim(ymin=0, ymax = 25)
        axes[i].set_xlim(xmin=0)

        axes[i].set_xlabel(plot_columns_x_label[i+1])
        axes[i].grid(True, which='both', linestyle = 'dotted', linewidth = 0.3)
        axes[i].legend(loc='lower center', handlelength=1.5, labelspacing=0.3)
        axes[i].minorticks_on()


    axes[0].set_ylabel('Depth [m]')

    axes[-1].plot(y_pred,
                df_site[plot_columns_x[0]].values,
                label='ML output',
                color='blue', linewidth = 0.6, markersize=0.0)
    if y_true is not None:
        axes[-1].plot(y_true,
                    df_proccessed[plot_columns_x[0]].values,
                    label='Raw data',
                    marker='o', color='r', linewidth = 0.6, markersize=0.8)


    axes[-1].set_xlabel(plot_columns_x_label[-1])
    axes[-1].grid(True, which='both', linestyle = 'dotted', linewidth = 0.3)
    axes[-1].legend(loc='lower center', handlelength=1.5, labelspacing=0.3, frameon=True, bbox_to_anchor=(0.5, -0.02))

    axes[-1].minorticks_on()
    axes[-1].invert_yaxis()
    axes[-1].set_xlim(xmin=0)


    plt.subplots_adjust(wspace=0.0, hspace=0)
    plt.tight_layout()

    return fig, axes

def plot_cpt_data_NW_site_all_notrue(figsize, plot_columns_x, df_site, df_proccessed, y_pred, plot_columns_x_label):

    fig, axes = plt.subplots(1, len(plot_columns_x)-1, figsize=figsize, dpi=500, sharey=True)

    for i, column in enumerate(plot_columns_x[1:-1]):
        axes[i].plot(df_site[column].values,
                      df_site[plot_columns_x[0]].values,
                      label='Raw data',
                      marker='o', color='k', linewidth = 0.2, markersize=0.8)

        axes[i].set_ylim(ymin=0, ymax = 10)
        axes[i].set_xlim(xmin=0)

        axes[i].set_xlabel(plot_columns_x_label[i+1])
        axes[i].grid(True, which='both', linestyle = 'dotted', linewidth = 0.3)
        axes[i].legend(loc='lower center', handlelength=1.5, labelspacing=0.3)
        axes[i].minorticks_on()


    axes[0].set_ylabel('Depth [m]')

    axes[-1].plot(y_pred,
                df_site[plot_columns_x[0]].values,
                label='ML output',
                color='blue', linewidth = 0.6, markersize=0.0)
    # axes[-1].plot(y_true,
    #               df_proccessed[plot_columns_x[0]].values,
    #               label='Raw data',
    #               marker='o', color='r', linewidth = 0.6, markersize=0.8)


    axes[-1].set_xlabel(plot_columns_x_label[-1])
    axes[-1].grid(True, which='both', linestyle = 'dotted', linewidth = 0.3)
    axes[-1].legend(loc='lower center', handlelength=1.5, labelspacing=0.3, frameon=True, bbox_to_anchor=(0.5, -0.02))

    axes[-1].minorticks_on()
    axes[-1].invert_yaxis()
    axes[-1].set_xlim(xmin=0)


    plt.subplots_adjust(wspace=0.0, hspace=0)
    plt.tight_layout()

    return fig, axes









def plot_cpt_data_ML_prediction(figsize, df_raw, df_SCPTu_SCPT, id_value, selected_columns_x, clf):
    plt.figure(figsize=figsize, dpi=500)

    # Select data for the current ID
    df_id = df_raw[df_raw['ID'] == id_value]
    # Drop rows with NaN values
    df_id = df_id.dropna(subset=['Vs (m/s)'])


    # Make predictions for the selected data
    df_id['Vs_ML_predicted'] = clf.predict(df_id[selected_columns_x])

    # Plot measured data
    plt.plot(df_id['Vs (m/s)'], df_id['Depth (m)'], label=f'Raw data (CPT ID {id_value})', color='k', marker='o')

    # Plot ML predictions
    plt.plot(df_id['Vs_ML_predicted'], df_id['Depth (m)'], label=f'Prediction (CPT ID {id_value})', color='blue', linestyle='--', marker='o')

    # Set plot labels and title
    # plt.title(f'Comparison of ML Predictions and Measurement Data ID = {id_value}')
    plt.xlabel('$v_s$ [m/s]')
    plt.ylabel('Depth [m]')
    plt.minorticks_on()
    #plt.ylim(ymin=0)
    plt.gca().invert_yaxis()
    plt.grid(True, which='both', linestyle = 'dotted', linewidth = 0.3)

    # Move the legend outside and to the top
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.17), ncol=1, handlelength = 2, labelspacing=2, frameon=False)
    plt.tight_layout()


import matplotlib
import numpy as np


def LegendVertical(Ax, Rotation=90, XPad=0, YPad=0, **LegendArgs):
    if Rotation not in (90,270):
        raise NotImplementedError('Rotation must be 90 or 270.')

    # Extra spacing between labels is needed to fit the rotated labels;
    # and since the frame will not adjust to the rotated labels, it is
    # disabled by default
    DefaultLoc = 'center left' if Rotation==90 else 'center right'
    ArgsDefaults = dict(loc=DefaultLoc, labelspacing=3, frameon=False)
    Args = {**ArgsDefaults, **LegendArgs}

    Handles, Labels = Ax.get_legend_handles_labels()
    Handles, Labels = Handles[-2:], Labels[-2:]
    if Rotation==90:
        # Reverse entries
        Handles, Labels = (reversed(_) for _ in (Handles, Labels))
    AxLeg = Ax.legend(Handles, Labels, **Args)

    LegTexts = AxLeg.get_texts()
    LegHandles = AxLeg.legend_handles

    for L,Leg in enumerate(LegHandles):
        if type(Leg) == matplotlib.patches.Rectangle:
            BBounds = np.ravel(Leg.get_bbox())
            BBounds[2:] = BBounds[2:][::-1]
            Leg.set_bounds(BBounds)

            LegPos = (
                # Ideally,
                #    `(BBounds[0]+(BBounds[2]/2)) - AxLeg.handletextpad`
                # should be at the horizontal center of the legend patch,
                # but for some reason it is not. Therefore the user will
                # need to specify some padding.
                (BBounds[0]+(BBounds[2]/2)) - AxLeg.handletextpad + XPad,

                # Similarly, `(BBounds[1]+BBounds[3])` should be at the vertical
                # top of the legend patch, but it is not.
                (BBounds[1]+BBounds[3])+YPad
            )

        elif type(Leg) == matplotlib.lines.Line2D:
            LegXY = Leg.get_xydata()[:,::-1]
            Leg.set_data(*(LegXY[:,_] for _ in (0,1)))

            LegPos = (
                LegXY[0,0] - AxLeg.handletextpad + XPad,
                max(LegXY[:,1]) + YPad
            )

        elif type(Leg) == matplotlib.collections.PathCollection:
            LegPos = (
                Leg.get_offsets()[0][0] + XPad,
                Leg.get_offsets()[0][1] + YPad,
            )
        else:
            raise NotImplementedError('Legends should contain Rectangle, Line2D or PathCollection.')

        PText = LegTexts[L]
        PText.set_verticalalignment('bottom')
        PText.set_rotation(Rotation)
        PText.set_x(LegPos[0])
        PText.set_y(LegPos[1])

        Lines = LegHandles[Leg]
        Lines.set_x(LegPos[0])
        Lines.set_y(LegPos[1])

    return(None)
