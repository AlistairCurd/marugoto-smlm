from fire import Fire

from .helpers import (
    categorical_crossval_,
    deploy_categorical_model_,
    loo_,
    train_categorical_model_,
)

if __name__ == "__main__":
    Fire(
        {
            "train": train_categorical_model_,
            "deploy": deploy_categorical_model_,
            "crossval": categorical_crossval_,
            "loo": loo_
        }
    )
