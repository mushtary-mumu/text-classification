import matplotlib.pyplot as plt


def plot_comment_distribution(df, color, xlabel, ylabel, title, figsize=(10,6)):

    fig, ax = plt.subplots(figsize=figsize)

    df.plot(kind="barh", 
            width=.5,
            color=color,
            ax=ax)  

    ax.set_title(title)
    ax.set_xlabel(xlabel) 
    ax.set_ylabel(ylabel)

    return fig, ax

