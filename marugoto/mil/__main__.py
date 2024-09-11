from fire import Fire
from .helpers import (
    train_categorical_model_,
    deploy_categorical_model_,
    categorical_crossval_,
    loo_
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
