import pandas as pd
import joblib
import logging

def main():
    """
    Loads data and model, generates predictions, and logs
    to scores.log
    """

    # Load data to be scored
    scoring_data = pd.read_csv('scoring_data.csv')
    
    # Load model
    dummy_clf = joblib.load('dummy_clf.joblib')
    
    # Generate scores
    scores = dummy_clf.predict(scoring_data)
    
    # Log scores
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    f_handler = logging.FileHandler('scores.log')
    f_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    f_handler.setFormatter(f_format)
    logger.addHandler(f_handler)
    logger.info(f'Scores: {scores}')


if __name__ == '__main__':
    main()
