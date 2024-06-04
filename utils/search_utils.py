import sklearn
import pandas as pd
import numpy as np
import copy
import joblib
import logging
import sys
import os

# setting up the logger to log to console

logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# mapping dictionaries

# mapping dictionaries

language_levels = {
    'none' : 0,
    'A1' : 1,
    'A2' : 2,
    'B1' : 4,
    'B2' : 8,
    'C1' : 16,
    'C2' : 32  
}

seniorities = {
    'none' : 0,
    'junior' : 1,
    'midlevel' : 2,
    'senior' : 3
}

degrees = {
    'none' : 0,
    'apprenticeship' : 1,
    'bachelor' : 2,
    'master' : 3,
    'doctorate' : 4  
}


def transform_input(input_dict, dict_type):
    """
    This is a function used to map and transform raw inputs of job and talent to mapping dictionaries provided above.
    The goal is to convert categorical values to numerical values with ordered properties, which is more suitable for modeling.
    
    Attributes
    ----------
    job:
    - languages
    - senioritires
    - min_degree
    - job_roles
    - max_salary
    
    talent:
    - languages
    - seniority
    - degree
    - job_roles
    - salary_expectation
    
    The structure would remain unchanged.
    """

    try:
    
        if dict_type == 'job':
        
            # creating empty dict
            transformed = {}
            
            # transform languages

            transformed['languages'] = [
                {
                    'title': lang['title'],
                    'rating': language_levels.get(lang['rating'], 0),
                    'must_have': lang['must_have']
                } 
                for lang in input_dict['languages'] if lang['must_have']
            ]
                                
            # Transform job_roles (keep as it is)
            transformed['job_roles'] = input_dict['job_roles']
            
            # Transform seniorities
            transformed['seniorities'] = [
                seniorities.get(sen, 0) for sen in input_dict['seniorities']
            ]
            
            # Transform max_salary (keep as it is)
            transformed['max_salary'] = input_dict['max_salary']
            
            # Transform min_degree
            transformed['min_degree'] = degrees.get(input_dict['min_degree'], 0)
            
            return transformed
    
        if dict_type == 'talent':
    
            # creating empty dict
            transformed = {}
            
            # transform languages
            transformed['languages'] = [
                {
                    'title': lang['title'],
                    'rating': language_levels.get(lang['rating'], 0)
                } 
                for lang in input_dict['languages']
            ]
            
            # Transform job_roles (keep as it is)
            transformed['job_roles'] = input_dict['job_roles']
            
            # Transform seniorities
            transformed['seniority'] = seniorities.get(input_dict['seniority'], 0)
            
            # Transform max_salary (keep as it is)
            transformed['salary_expectation'] = input_dict['salary_expectation']
            
            # Transform min_degree
            transformed['degree'] = degrees.get(input_dict['degree'], 0)
            
            return transformed

        else:

            print('wrong input')

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        logger.error("Exception occurred", exc_info=True)
        print(exc_type, fname, exc_tb.tb_lineno)
        


def check_dict_structure(dict, dict_type):
    """
    This function checks the structure of input dictionaries.
    """
    if dict_type == 'job':
    
        required_job_keys = {'languages', 'job_roles', 'seniorities', 'max_salary', 'min_degree'}
        if not all(key in dict for key in required_job_keys):
            raise ValueError("job_dict is missing required keys")
        
        if not isinstance(dict['languages'], list):
            raise TypeError("job_dict['languages'] should be a list")
        
        for lang in dict['languages']:
            if not all(key in lang for key in ['title', 'rating', 'must_have']):
                raise ValueError("Each language in job_dict must have 'title', 'rating', and 'must_have'")
    else:
        return('ok')

    
    if dict_type == 'talent':

        required_talent_keys = {'languages', 'job_roles', 'seniority', 'salary_expectation', 'degree'}
        if not all(key in dict for key in required_talent_keys):
            raise ValueError("talent_dict is missing required keys")
        
        if not isinstance(dict['languages'], list):
            raise TypeError("talent_dict['languages'] should be a list")
        
        for lang in dict['languages']:
            if not all(key in lang for key in ['title', 'rating']):
                raise ValueError("Each language in talent_dict must have 'title' and 'rating'")
    else:
        return('ok')
     


def match_language_from_dict(job_dict, talent_dict):
    """
    This function transforms and maps the language inputs to numerical ordered values.
    It then compares languages of job requirement with talent language in terms of particular language and fluency level.
    If fluency level of talent equals or is better than required language, then the match is positive.
    This only counts for languages which are mandatory (must_have=True).  
    """
    try:
        if(check_dict_structure(job_dict, 'job') == 'ok' and check_dict_structure(talent_dict, 'talent') == 'ok'):
            pass
        else:
            raise('error in dict structure')
    
        transformed_job = transform_input(job_dict, 'job')
        transformed_talent = transform_input(talent_dict, 'talent')
    
        job_languages = transformed_job['languages']
        talent_languages = {lang['title']: lang['rating'] for lang in transformed_talent['languages']}
    
        for job_lang in job_languages:
            job_title = job_lang['title']
            job_rating = job_lang['rating']
            
            if job_title not in talent_languages or talent_languages[job_title] < job_rating:
                return 0
            
        return 1

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        logger.error("Exception occurred in %s at line %d", fname, exc_tb.tb_lineno, exc_info=True)
        raise    


def match_seniority_from_dict(job_dict, talent_dict):
    """
    This function transforms and maps the seniority inputs to numerical ordered values.
    It then compares seniorities of job requirement with talent seniority.
    If seniority of talent equals or is better than required seniority, then the match is positive.
    """
    try:
        if(check_dict_structure(job_dict, 'job') == 'ok' and check_dict_structure(talent_dict, 'talent') == 'ok'):
            pass
        else:
            raise('error in dict structure')
    
        transformed_job = transform_input(job_dict, 'job')
        transformed_talent = transform_input(talent_dict, 'talent')
    
        min_job_seniority = min(transformed_job['seniorities'])
        talent_seniority = transformed_talent['seniority']
    
        if talent_seniority >= min_job_seniority:
            return 1
        
        return 0
        
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        logger.error("Exception occurred in %s at line %d", fname, exc_tb.tb_lineno, exc_info=True)
        raise    

def match_degree_from_dict(job_dict, talent_dict):
    """
    This function transforms and maps the educational degree inputs to numerical ordered values.
    It then compares educational degree of job requirement with talent educational degree.
    If degree of talent equals or is better than required degree, then the match is positive.
    """
    try:
        if(check_dict_structure(job_dict, 'job') == 'ok' and check_dict_structure(talent_dict, 'talent') == 'ok'):
            pass
        else:
            raise('error in dict structure')
    
        transformed_job = transform_input(job_dict, 'job')
        transformed_talent = transform_input(talent_dict, 'talent')
    
        min_job_degree = transformed_job['min_degree']
        talent_degree = transformed_talent['degree']
    
        if talent_degree >= min_job_degree:
            return 1
        
        return 0

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        logger.error("Exception occurred in %s at line %d", fname, exc_tb.tb_lineno, exc_info=True)
        raise 


def match_salary_from_dict(job_dict, talent_dict):
    """
    This function transforms and compares salary values.
    If expected salary of talent equals or is smaller than job requirement max_salary, then the match is positive.
    """
    try:
        if(check_dict_structure(job_dict, 'job') == 'ok' and check_dict_structure(talent_dict, 'talent') == 'ok'):
            pass
        else:
            raise('error in dict structure')
    
        transformed_job = transform_input(job_dict, 'job')
        transformed_talent = transform_input(talent_dict, 'talent')
    
        job_max_salary = transformed_job['max_salary']
        talent_expected_salary = transformed_talent['salary_expectation']
    
        if talent_expected_salary <= job_max_salary:
            return 1
        
        return 0
        
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        logger.error("Exception occurred in %s at line %d", fname, exc_tb.tb_lineno, exc_info=True)
        raise


def match_job_roles_from_dict(job_dict, talent_dict):
    """
    This function transforms and compares job roles.
    If any job roles(s) of talent match any job requirement role(s), then the match is positive.
    """
    try:
        if(check_dict_structure(job_dict, 'job') == 'ok' and check_dict_structure(talent_dict, 'talent') == 'ok'):
            pass
        else:
            raise('error in dict structure')
    
        transformed_job = transform_input(job_dict, 'job')
        transformed_talent = transform_input(talent_dict, 'talent')
    
        job_roles = transformed_job['job_roles']
        talent_roles = transformed_talent['job_roles']
    
        for role in talent_roles:
            if role in job_roles:
                return 1
        return 0
        
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        logger.error("Exception occurred in %s at line %d", fname, exc_tb.tb_lineno, exc_info=True)
        raise


def create_dataset(j_dict, t_dict):
    """
    This function creates boolean values for given features based on matching logic.
    It then creates a pandas dataframe with these features.
    Finally, it converts it to dictionary and returns it as output of a function.
    """
    try:
        seniority_match = match_seniority_from_dict(j_dict, t_dict)
        degree_match = match_degree_from_dict(j_dict, t_dict)
        salary_match = match_salary_from_dict(j_dict, t_dict)
        language_match = match_language_from_dict(j_dict, t_dict)
        job_role_match = match_job_roles_from_dict(j_dict, t_dict)
    
        df_match = pd.DataFrame(
            {
                'seniority_match' : [seniority_match], 
                'degree_match' : [degree_match], 
                'salary_match' : [salary_match], 
                'language_match' : [language_match], 
                'job_role_match' : [job_role_match]
            }
        )
    
        input_dict = df_match.to_dict(orient='records')
    
        return input_dict
        
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        logger.error("Exception occurred in %s at line %d", fname, exc_tb.tb_lineno, exc_info=True)
        raise


def create_prediction(job, talent):
    """
    This function creates dataset of fetures.
    It then runs a binary logistic regression classification model to predict whether a talent matches the job requirement.
    Finally, it converts the to dictionary as per task requirement and returns it as output of a function.
    """
    try:
        res = {}
        data = create_dataset(job, talent)
        label = model.predict(data)[0]
        score = scorer(talent, job)
        # score = model.predict_proba(data)[:, 1][0]
        res['talent'] = talent
        res['job'] = job
        res['label'] = label
        res['score'] = score
    
        return res
        
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        logger.error("Exception occurred in %s at line %d", fname, exc_tb.tb_lineno, exc_info=True)
        raise

def check_language_match(talent_languages, job_languages):
    """
    This function:
    - maps numeric levels to language fluency levels
    - filters out any language in job requirements marked as must_have=False
    (although it can be considered and weighted in some future version as a "good to have" feature)
    - matches talent languages with job requirements
    - calculates and sums fluency weights for every matched language against required fluency
    """
    try:
        # Creating a dictionary for talent language levels with weights
        talent_language_dict = {lang['title']: language_levels[lang['rating']] for lang in talent_languages}
        
        total_weight = 0 # initiated total weight of all 'must_have' job languages
        match_count = 0 # initiated total weight of matched 'must_have' job languages

        for job_language in job_languages:
            # only consider 'must_have' job languages
            if job_language.get('must_have', False):
                job_title = job_language['title']
                job_rating = language_levels[job_language['rating']]
                total_weight += job_rating

                 # check if the talent has the required level for the 'must_have' job language
                if talent_language_dict.get(job_title, 0) >= job_rating:
                    match_count += job_rating # add the weight of the 'must_have' job language to total weight

        threshold = 0.5 # Define the threshold for matching (50% of total weight)
        is_match = match_count >= threshold * total_weight # Determine if the match count meets the threshold
        
        return int(is_match), total_weight # Return the match result (1 for match, 0 for no match) and total weight

    except KeyError as e:
        logger.error("KeyError occurred", exc_info=True)
        return 0, 0
    except TypeError as e:
        logger.error("TypeError occurred", exc_info=True)
        return 0, 0
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        logger.error("Exception occurred in %s at line %d", fname, exc_tb.tb_lineno, exc_info=True)
        print(exc_type, fname, exc_tb.tb_lineno)
        return 0, 0

        
def scorer(talent_dict, job_dict):
    """
    This function weights performance of each factors plugged in the model by weighing agains job requirements:
    - language fluency (difference)
    - seniority (difference between talent seniority and minimum required seniority)
    - degree (difference between talent degree and job required minimum degree)
    - salary (defference between job max_salary and expected salar)
    - job roles (counting number of job roles in common)
    and adds a cumulative score which would serve as a sorting criterion between the candidates during the matching process
    """
    try:
        
        def weight_languages(talent_dict, job_dict):
            return check_language_match(talent_dict['languages'], job_dict['languages'])[1]/100
            
        def weight_job_roles(talent_dict, job_dict):
            job_job_roles = job_dict['job_roles']
            talent_job_roles = talent_dict['job_roles']
            return sum(role in job_job_roles for role in talent_job_roles)
        
        def weight_seniorities(talent_dict, job_dict):
            talent_dict['seniority_mapped'] = seniorities.get(talent_dict['seniority'], 0)
            job_dict['seniorities_mapped'] = [seniorities.get(s, 0) for s in job_dict['seniorities']]
            return talent_dict['seniority_mapped'] - min(job_dict['seniorities_mapped'])
            
        def weight_salaries(talent_dict, job_dict):
            job_salary = job_dict['max_salary']
            talent_salary = talent_dict['salary_expectation']
            return (job_salary - talent_salary)/10000
        
        def weight_degrees(talent_dict, job_dict):
            talent_dict['degree_mapped'] = degrees.get(talent_dict.get('degree', 'none'), 0)
            job_dict['degree_mapped'] = degrees.get(job_dict.get('min_degree', 'none'), 0)
            return (talent_dict['degree_mapped'] - job_dict['degree_mapped'])
        
        language_weight = weight_languages(talent_dict, job_dict)
        job_role_weight = weight_job_roles(talent_dict, job_dict)
        seniority_weight = weight_seniorities(talent_dict, job_dict)
        salary_weight = weight_salaries(talent_dict, job_dict)
        degree_weight = weight_degrees(talent_dict, job_dict)
        
        score = language_weight + job_role_weight + seniority_weight + salary_weight + degree_weight
    
        return score

    except KeyError as e:
        logger.error("KeyError occurred", exc_info=True)
        return 0, 0
    except TypeError as e:
        logger.error("TypeError occurred", exc_info=True)
        return 0, 0
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        logger.error("Exception occurred in %s at line %d", fname, exc_tb.tb_lineno, exc_info=True)
        print(exc_type, fname, exc_tb.tb_lineno)
        return 0, 0
        
