def csv_to_png():


    "num_samples,num_features,num_classes,train_time,predict_time"
    import pandas as pd

    df = pd.read_csv('tabpfn_speed.csv')
    # no need to prerpcoes for na or duplicates
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    # plot
    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.set_theme(style="whitegrid")
    for col in ['train_time', 'predict_time']:
        sns.scatterplot(data=df, x="num_features", y=col, hue="num_samples")
        plt.savefig(f'{col}.png')
        plt.close()


if __name__ == '__main__':
    csv_to_png()