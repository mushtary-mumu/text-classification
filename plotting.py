import matplotlib.pyplot as plt


def plot_comment_distribution(df, color, xlabel, ylabel, title, figsize=(10,6)):
    
    """
        Plot comment distribution.

        Parameters
        ----------
        df : pandas dataframe
                Dataframe containing the comment distribution.
        color : str
                Color of the plot.
        xlabel : str    
                Label of the x-axis.
        ylabel : str
                Label of the y-axis.
        title : str
                Title of the plot.
        figsize : tuple, optional
                Size of the figure. The default is (10,6).

        Returns
        -------
        fig : matplotlib figure
                Figure of the plot.
        ax : matplotlib axis
                Axis of the plot.
    
    """

    fig, ax = plt.subplots(figsize=figsize)

    df.plot(kind="barh", 
            width=.5,
            color=color,
            ax=ax)  

    ax.set_title(title)
    ax.set_xlabel(xlabel) 
    ax.set_ylabel(ylabel)

    return fig, ax

