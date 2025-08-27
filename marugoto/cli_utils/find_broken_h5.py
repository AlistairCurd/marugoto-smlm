#!/usr/bin/env python3
from pathlib import Path

import h5py
from fire import Fire
from tqdm import tqdm


def find_broken_h5s_(feature_dir: str) -> None:
    for h5 in tqdm(list(Path(feature_dir).glob("*.h5"))):
        try:
            with h5py.File(h5):
                pass
        except OSError:
            print(f"Error while opening file {h5}")


if __name__ == "__main__":
    Fire(find_broken_h5s_)
