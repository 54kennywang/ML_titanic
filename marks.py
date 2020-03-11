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
    mark('stacking_model_1.csv')
    mark('stacking_model_2.csv')

    mark('achieve_99_models.csv')
    mark('achieve_99_mytree.csv')
    mark('achieve_99_dtree_no_tuning.csv')
    mark('achieve_99_dtree_with_tuning.csv')
    mark('achieve_99_feature_selection.csv')
    mark('achieve_99_vote_hard.csv')
    mark('achieve_99_vote_soft.csv')
    # mark('GA.csv')
