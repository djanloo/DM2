"""Utils module for plotting because jupys are too long"""
import numpy as np
import pandas as pd
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.preprocessing import MinMaxScaler



def get_normalized_df(df, features):
    """Needed for ridgeline plot"""
    norm_df = pd.DataFrame(columns = ["feature", "normalized_value", "true_value"])
    for feature in features:
        rescaled_values = MinMaxScaler().fit_transform(df[feature].to_numpy().reshape(-1,1)).reshape(-1)
        single_feature_df = pd.DataFrame(dict(feature=[feature]*len(df), normalized_value=rescaled_values, true_value=df[feature]) )
        norm_df = pd.concat([norm_df, single_feature_df], ignore_index=True)
    return norm_df


def ridgeline_plot(df, columns, 
                   startcolor = 0, 
                   collective_name="dummy", 
                   bottom=0.05, top=1, 
                   hspace=-0.5, ylim=(0, 7) 
                   ):
    
    # Needed to stuff subplots like in a turkey
    rcParams["figure.autolayout"] = False

    # Needed to overlap
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

    # sc_params = dict(hspace = -0.85, startcolor=0, name="sc", family=sc_family, top=1, bottom=0.05, ylim=(0,7.5))
    # mfcc_params =  dict(hspace = -0.8, startcolor=1, name="mfcc", family=mfcc_family, top=1, bottom=0.05, ylim=(0,7.5))
    # stft_params = dict(hspace = -0.85, startcolor=1.8, name="stft", family=stft_family, top=1, bottom=0.05, ylim=(0,7.5))
    # triv_params = dict(hspace = -0.9, startcolor=0.5, name="trivial", family=trivial_quant_features, top=1, bottom=0.05, ylim=(0,20.5))

   
    normalized_df = get_normalized_df(df, columns)
    Nplots= len(columns)
    print(f"Ridgeline has {Nplots} subplots")

    pal = sns.cubehelix_palette(Nplots,start=startcolor, rot=-.25, light=.7)
    g = sns.FacetGrid(normalized_df, row="feature", hue="feature", aspect=8, height=0.7, palette=pal)

    g.despine(bottom=True, left=True)

    # Draw the densities in a few steps
    g.map(sns.kdeplot,  "normalized_value",
        clip_on=True,
        fill=True, alpha=1, linewidth=1.5, bw_adjust=0.61)
    g.map(sns.kdeplot, "normalized_value", clip_on=True, color="k", lw=1.5, bw_adjust=0.61)

    # passing color=None to refline() uses the hue mapping
    g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)

    # Define and use a simple function to label the plot in axes coordinates
    def label(x, color, label):
        ax = plt.gca()
        ax.text(-0.35, .0, label, fontweight="bold", color=color,
                    ha="left", va="bottom", transform=ax.transAxes)

    def write_extremes(trues, color, label):
        ax=plt.gca()
        filtered = trues[np.logical_not(np.isnan(trues))]

        m, M = min(filtered), max(filtered)
        q1, q2 = np.quantile(filtered, [.0, 1])
        
        x1, x2 = (q1 - m)/(M - m), (q2 - m)/(M - m)

        if abs(q2) > 0.1:
                strmax = f"{q2:.0f}"
        else:
                strmax = f"{q2:.0e}"
        
        if abs(q1) > 0.1 or abs(q1) == 0.0:
                strmin = f"{q1:.0f}"
        else:
                strmin = f"{q1:.0e}"


        ax.text(x1, .00, strmin,  color=color,
                    ha="center", va="bottom", transform=ax.transAxes, zorder=3)
        ax.text(x2, .00,strmax ,  color=color,
                    ha="center", va="bottom", transform=ax.transAxes, zorder=3)
                    
    g.map(label,  "normalized_value")
    g.map(write_extremes, "true_value")
    # Set the subplots to overlap
    g.figure.subplots_adjust(hspace=hspace, top=top, bottom=bottom)

    # Remove axes details that don't play well with overlap
    g.set_titles("")
    g.set(ylim=ylim)
    g.set(yticks=[], ylabel="")
    g.set(xticks=[], xlabel="")
    plt.savefig(f"images/{collective_name}_ridgeline.pdf")

    # Sets it back for other figures
    rcParams["figure.autolayout"] = True

    return plt.gcf()