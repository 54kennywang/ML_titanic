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
    mark('stacking_model_1.csv') # 0.7751196172248804
    # mark('stacking_model_2.csv') # 0.7727272727272727
    # mark('stacking_model_3_layers.csv')

    # mark('achieve_99_models.csv') # 0.7751196172248804
    # mark('achieve_99_mytree.csv') # 0.7631578947368421
    # mark('achieve_99_dtree_no_tuning.csv') # 0.7631578947368421
    # mark('achieve_99_dtree_with_tuning.csv') # 0.7751196172248804
    # mark('achieve_99_feature_selection.csv') # 0.7703349282296651
    # mark('achieve_99_vote_hard.csv') # 0.7631578947368421
    # mark('achieve_99_vote_soft.csv') # 0.7727272727272727
    # mark('GA.csv')
