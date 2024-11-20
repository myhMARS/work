import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from tqdm import tqdm


def show(model_factory, label, X_train, X_test, y_train, y_test, ax, color, style):
    num_trees = np.arange(1, 101, 5)
    np.random.seed(0)

    score = []

    with tqdm(num_trees) as pbar:
        for n_tree in pbar:
            model = model_factory(n_tree)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            score.append(accuracy_score(y_pred, y_test))
            pbar.set_postfix({
                'n_tree': n_tree,
                'score': score[-1]
            })

    ax.plot(
        num_trees,
        score,
        color=color,
        linestyle=style,
        label=f'{label} score'
    )

    return plt
