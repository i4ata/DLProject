import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_aft_simple():

    d = pickle.load(open('metrics_simple.pkl', 'rb'))

    # Add random guessing
    d['train']['losses'].insert(0, 3.6109)
    d['val']['losses'].insert(0, 3.6109)
    d['train']['accuracies'].insert(0, 1 / 37)
    d['val']['accuracies'].insert(0, 1 / 37)

    train_df = pd.DataFrame(data=d['train']).reset_index()
    train_df['set'] = 'training'
    val_df = pd.DataFrame(data=d['val']).reset_index()
    val_df['set'] = 'validation'
    df = pd.concat((train_df, val_df))

    fig, ax = plt.subplots(ncols=2)
    sns.lineplot(df, x='index', y='losses', hue='set', ax=ax[0])
    sns.lineplot(df, x='index', y='accuracies', hue='set', ax=ax[1])

    for a in ax:
        a.set_xlabel('epoch')
        a.set_xticks(np.arange(len(train_df)))

    ax[0].set_ylabel('mean loss')
    ax[1].set_ylabel('mean accuracy %')

    plt.suptitle('AFT-simple training evaluation')
    plt.tight_layout()
    plt.savefig('plots/aft_simple.pdf', bbox_inches='tight')
    plt.show()

def plot_aft_local():

    d = pickle.load(open('metrics_local.pkl', 'rb'))

    # Add random guessing
    d['train']['losses'].insert(0, 3.6109)
    d['val']['losses'].insert(0, 3.6109)
    d['train']['accuracies'].insert(0, 1 / 37)
    d['val']['accuracies'].insert(0, 1 / 37)

    train_df = pd.DataFrame(data=d['train']).reset_index()
    train_df['set'] = 'training'
    val_df = pd.DataFrame(data=d['val']).reset_index()
    val_df['set'] = 'validation'
    df = pd.concat((train_df, val_df))

    fig, ax = plt.subplots(ncols=2)
    sns.lineplot(df, x='index', y='losses', hue='set', ax=ax[0])
    sns.lineplot(df, x='index', y='accuracies', hue='set', ax=ax[1])

    for a in ax:
        a.set_xlabel('epoch')
        a.set_xticks(np.arange(len(train_df), step=2))

    ax[0].set_ylabel('mean loss')
    ax[1].set_ylabel('mean accuracy %')

    plt.suptitle('AFT-local training evaluation')
    plt.tight_layout()
    plt.savefig('plots/aft_local.pdf', bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    plot_aft_local()