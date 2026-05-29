from skopt.space import Real, Integer
from skopt import gp_minimize
from skopt.utils import use_named_args
from src.train_resnet_inception import train_model
import pickle
import pandas as pd

search_space = [Real(
                    1e-5,
                    1e-2,
                    prior="log-uniform",
                    name="lr"
                ),

                Real(
                    1e-6,
                    1e-2,
                    prior="log-uniform",
                    name="weight_decay"
                ),

                Real(
                    0.0,
                    0.5,
                    name="dropout"
                ),

                Integer(
                    4,
                    32,
                    name="se_reduction"
                ),

                Real(
                    1.0,
                    5.0,
                    name="lambda_h3k27ac"
                )]

results = []

@use_named_args(search_space)
def objective(
    lr,
    weight_decay,
    dropout,
    se_reduction,
    lambda_h3k27ac
):

    print(
        f"lr={lr:.6f} "
        f"wd={weight_decay:.6f} "
        f"dropout={dropout:.3f} "
        f"se={se_reduction} "
        f"lambda={lambda_h3k27ac:.2f}"
    )

    atac_auc, pearson = train_model(
                            lr=lr,
                            weight_decay=weight_decay,
                            dropout=dropout,
                            se_reduction=se_reduction,
                            lambda_h3k27ac=lambda_h3k27ac,
                            train_fraction=0.3,
                            epochs=5,
                            save_model=False
                        )

    normalized_auc = atac_auc / 0.9636

    normalized_pearson = pearson / 0.4339

    score = (
        normalized_auc
        +
        normalized_pearson
    ) / 2

    results.append({
        "lr": lr,
        "weight_decay": weight_decay,
        "dropout": dropout,
        "se_reduction": se_reduction,
        "lambda_h3k27ac": lambda_h3k27ac,
        "atac_auc": atac_auc,
        "pearson": pearson,
        "score": score
    })

    print(
        f"AUC={atac_auc:.4f} "
        f"Pearson={pearson:.4f} "
        f"Score={score:.4f}"
    )

    pd.DataFrame(results).to_csv("bayes_opt_results.csv", index=False)

    return -score

if __name__ == "__main__":

    result = gp_minimize(
        objective,
        dimensions=search_space,
        n_calls=30,
        n_initial_points=5,
        random_state=42
    )

    pd.DataFrame(results).sort_values(
        "score",
        ascending=False
    ).to_csv(
        "bayes_opt_results_sorted.csv",
        index=False
    )

    print("\nBest Parameters:")

    for name, value in zip(
            [d.name for d in search_space],
            result.x
    ):
        print(
            f"{name}: {value}"
        )

    print(
        f"Best Score: {-result.fun:.4f}"
    )

    with open(
        "bayes_result.pkl",
        "wb"
    ) as f:
        pickle.dump(
            result,
            f
        )