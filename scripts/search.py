import os
import sys
import joblib
import pandas as pd

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

utils_path = os.path.join(base_path, 'utils')
sys.path.append(utils_path)

from search_utils import transform_input, check_dict_structure, match_language_from_dict, match_seniority_from_dict, language_levels
from search_utils import match_degree_from_dict, match_salary_from_dict, match_job_roles_from_dict, create_dataset, create_prediction

# load model

model_path = os.path.join(base_path, 'models', 'logistic_regression_model.pkl')
model = joblib.load(model_path)

class Search:
    """
    This is a class used to match talents to jobs using a trained logistic regression model from scikit-learn library.
    
    Attributes
    ----------
    model : sklearn.linear_model.LogisticRegression
        a logistic regression model loaded from a .pkl file

    Methods
    -------
    match(talent: dict, job: dict) -> dict
        Matches a single talent with a job and returns the prediction and score.
        
    match_bulk(talents: list, jobs: list) -> list
        Matches multiple talents with multiple jobs and returns a sorted list of predictions and scores.
    """
    def __init__(self) -> None:
        """
        Initializes the Search class with a trained logistic regression model.

        Parameters
        ----------
        model_path : str
            The file path to the trained logistic regression model saved as a .pkl file.
        """
        self.model = joblib.load(model_path)    

    def match(self, talent: dict, job: dict) -> dict:
        """
        Matches a single talent with a job and returns the prediction and score.

        Parameters
        ----------
        talent : dict
            A dictionary containing talent information.
        job : dict
            A dictionary containing job information.

        Returns
        -------
        dict
            A dictionary containing the talent, job, predicted label, and score.
        """

        # ==> Method description <==
        # This method takes a talent and job as input and uses the machine learning
        # model to predict the label. Together with a calculated score, the dictionary
        # returned has the following schema:
        #
        # {
        #   "talent": ...,
        #   "job": ...,
        #   "label": ...,
        #   "score": ...
        # }
        #

        res = {}
        data = create_dataset(job, talent)
        label = model.predict(data)[0]
        if label == 0:  # filtering out non-matching talents
            return None    
        score = model.predict_proba(data)[:, 1][0]
        res['talent'] = talent
        res['job'] = job
        res['label'] = label
        res['score'] = score
    
        return res

        pass

    def match_bulk(self, talents: list[dict], jobs: list[dict]) -> list[dict]:
        """
        Matches list of multiple talents with list of multiple jobs and returns a sorted list of predictions and scores

        Input
        ----------
        talents : list of dictionaries with talent information.
        jobs :  list of dictionaries with required job information.

        Output
        -------
        res : list of dictionaries containing the talent, job, predicted label, and score, sorted by score in descending order.
        """

        # ==> Method description <==
        # This method takes a multiple talents and jobs as input and uses the machine
        # learning model to predict the label for each combination. Together with a
        # calculated score, the list returned (sorted descending by score!) has the
        # following schema:
        #
        # [
        #   {
        #     "talent": ...,
        #     "job": ...,
        #     "label": ...,
        #     "score": ...
        #   },
        #   {
        #     "talent": ...,
        #     "job": ...,
        #     "label": ...,
        #     "score": ...
        #   },
        #   ...
        # ]
        #
        
        final_result = []

        for j in range(len(jobs)):
            for t in range(len(talents)):
                talents_from_list = talents[t]
                jobs_from_list = jobs[j]
                predictions = self.match(talents_from_list, jobs_from_list)
                if prediction:  # only add if there is a match
                    final_result.append({'job_index': j, 'talent_index': t, 'prediction': prediction})

        if not final_result:  # no result if no match is found
            return []

        dff = pd.DataFrame(final_result)
        dff['score'] = [dff['prediction'][x]['score'] for x in range(dff.shape[0])]
        dff = dff.sort_values(['job_index', 'score'], ascending=[True, False])
        
        match_bulk = [
            {
                "talent": row['prediction']['talent'],
                "job": row['prediction']['job'],
                "label": row['prediction']['label'],
                "score": row['prediction']['score']
            }
            for _, row in dff.iterrows()
        ]
    
        return match_bulk

        pass


if __name__ == "__main__":
    # Load the model
    # model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'logistic_regression_model.pkl')
    search = Search()

    # Example talents and jobs data
    
    talents = [
        {
            "languages": [{"title": "German", "rating": "A2"}, {"title": "English", "rating": "C1"}],
            "job_roles": ["frontend-developer"],
            "seniority": "midlevel",
            "salary_expectation": 50000,
            "degree": "bachelor"
        },
        # ...
    ]

    jobs = [
        {
            "languages": [{"title": "German", "rating": "B2", "must_have": True}, {"title": "English", "rating": "B2", "must_have": False}],
            "job_roles": ["frontend-developer"],
            "seniorities": ["midlevel"],
            "max_salary": 70000,
            "min_degree": "bachelor"
        },
        # ...
    ]

    # Run match method for a single talent and job
    single_result = search.match(talents[0], jobs[0])
    print("Single Match Result:")
    print(single_result)

    # Run match_bulk method for multiple talents and jobs
    bulk_result = search.match_bulk(talents, jobs)
    print("Bulk Match Result:")
    for match in bulk_result:
        print(match)

