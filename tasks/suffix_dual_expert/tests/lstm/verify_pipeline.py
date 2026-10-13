"""Bounded local LSTM pipeline verification; never formal training."""
from ..common.verify_pipeline import main

if __name__ == '__main__':
    main(default_model_type='lstm')
