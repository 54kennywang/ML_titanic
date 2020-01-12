import numpy as np
import pandas as pd
import os
dirname = os.path.dirname(__file__)


def mark(pred):
    solution = os.path.join(dirname, './output/solution.csv')
    submission = os.path.join(dirname, './output/'+pred)
    solution = pd.read_csv(solution)
    submission = pd.read_csv(submission)

    solution.columns = ['PassengerId', 'Sol']
    submission.columns = ['PassengerId', 'Pred']

    df = pd.concat([solution[['Sol']], submission[['Pred']]], axis=1)
    num_row = df.shape[0]
    print(pred[:-4], '==', (df[(df['Sol'] == df['Pred'])]).shape[0] / num_row)

if __name__== "__main__":
    mark('achieve_99_models.csv')
    mark('achieve_99_mytree.csv')
    mark('achieve_99_dtree.csv')
    mark('achieve_99_tune_model.csv')
    mark('achieve_99_dtree_rfe.csv')
    mark('achieve_99_vote_hard.csv')
    mark('achieve_99_vote_soft.csv')
    mark('advanced_feature_with_stacking_5_fold.csv')
    mark('advanced_feature_with_stacking.csv')
    # mark('GA.csv')
