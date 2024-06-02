import sklearn
import pandas as pd
import numpy as np
import copy
import joblib
import os

def transform_input(input_dict, dict_type):

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

        raise('error:', e)
        
    

def check_dict_structure(dict, dict_type):

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


def match_seniority_from_dict(job_dict, talent_dict):

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
    

def match_degree_from_dict(job_dict, talent_dict):

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
    


def match_salary_from_dict(job_dict, talent_dict):

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



def match_job_roles_from_dict(job_dict, talent_dict):

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
    


def create_dataset(j_dict, t_dict):
    
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


def create_prediction(job, talent):
    res = {}
    data = create_dataset(job, talent)
    label = model.predict(data)[0]
    score = model.predict_proba(data)[:, 1][0]
    res['talent'] = talent
    res['job'] = job
    res['label'] = label
    res['score'] = score

    return res
    

# mapping dictionaries

language_levels = {
    'none' : 0,
    'A1' : 1,
    'A2' : 2,
    'B1' : 3,
    'B2' : 4,
    'C1' : 5,
    'C2' : 6  
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


