import json
from sympy import *
import pandas as pd
from scipy import integrate
import numpy as np
from pathlib import Path

def load_json(json_file):
    """
    Load the json file to a dictionary

    Parameters
    ----------
    json_file : str
        The file path of the json file

    Returns
    -------
    comp_dict : dict
        The dictionary of the json file

    """
    with open(json_file) as f:
        comp_dict = json.load(f)
    return comp_dict

def save_json(comp_dict, json_file):
    """
    Save the dictionary to a json file

    Parameters
    ----------
    comp_dict : dict
        The dictionary of the bond graph model
    json_file : str
        The file path of the json file

    Returns
    -------
    None

    side effect
    ------------
    Save the dictionary to a json file

    """
    with open(json_file, 'w') as f:
        json.dump(comp_dict, f,indent=4)

def symExp_to_funcEval(symExp, df):
    """
    Evaluate the SymPy expression with the values in the dataframe

    Parameters
    ----------
    symExp : str
        The SymPy expression
    df : pandas.DataFrame
        The dataframe that contains the value of the SymPy expression.
        The column name of the dataframe should be the variable name in the SymPy expression.
        The column name of the dataframe should be 't' for the time variable.

    Returns
    -------
    func_eval : numpy.ndarray
        The numerical value of the SymPy expression
    func_eval_integrate : float
        The numerical value of the integral of the SymPy expression
    func_eval_integrate_abs : float
        The numerical value of the integral of the absolute value of the SymPy expression
    func_eval_integrate_cumulative : numpy.ndarray
        The numerical value of the cumulative integral of the SymPy expression
    """
    list_vars=list(symExp.free_symbols)
    list_vars_str=[str(var) for var in list_vars]
    func=lambdify(list_vars,symExp,'numpy')
    func_eval=func(*[df[var] for var in list_vars_str])
    func_eval_integrate=np.trapezoid(func_eval,df['t'])
    func_eval_integrate_abs=np.trapezoid(np.abs(func_eval),df['t'])
    func_eval_integrate_cumulative=integrate.cumulative_trapezoid(func_eval,x=df['t'],initial=0)

    return func_eval,func_eval_integrate,func_eval_integrate_abs,func_eval_integrate_cumulative

def calc_energy(P_comp_expr_dict,result_csv):
    """
    Calculate the energy and activity of the bond graph model

    Parameters
    ----------
    P_comp_expr_dict : str
        The dictionary of the expressions of the power of the components in the bond graph model
        The key of the dictionary is the component name
    result_csv : str
        The file path of the csv file to save the result

    Returns
    -------
    None

    side effect
    ------------
    Save the energy and activity of the bond graph model to a csv file

    """
    df_result_csv=pd.read_csv(result_csv)
    # get the path of the csv file
    csv_path=Path(result_csv).parent
    csv_file_name=Path(result_csv).stem
    csv_file_power_comp=csv_path/(csv_file_name+'_comp_power.csv')
    csv_file_activity_comp=csv_path/(csv_file_name+'_comp_activity.csv')
    E_comp_val=np.zeros(len(P_comp_expr_dict))
    A_comp_val=np.zeros(len(P_comp_expr_dict))
    AI_comp_val=np.zeros(len(P_comp_expr_dict))
    df_power_comp=pd.DataFrame(columns=P_comp_expr_dict.keys())
    df_activity_comp=pd.DataFrame(columns=P_comp_expr_dict.keys())
    df_power_comp['t']=df_result_csv['t']
    df_activity_comp['t']=df_result_csv['t']
    i=0
    A_total=0
    for comp in P_comp_expr_dict:
        P_comp_expr=P_comp_expr_dict[comp]
        df_power_comp[comp],E_comp_val[i],A_comp_val[i],df_activity_comp[comp]=symExp_to_funcEval(P_comp_expr,df_result_csv)
        A_total+=A_comp_val[i]
        i=i+1
        
    AI_comp_val=A_comp_val/A_total*100
    df_power_comp.to_csv(csv_file_power_comp,index=False)
    df_activity_comp.to_csv(csv_file_activity_comp,index=False)
    dict_activity={'Componentl list':list(P_comp_expr_dict.keys()),'Energy': E_comp_val.tolist(), 'Activity': A_comp_val.tolist(), 'Activity Index': AI_comp_val.tolist(),
                   }
    save_json(dict_activity,csv_path/(csv_file_name+'_activity.json'))

if __name__ == "__main__": 
    # calculate the energy and activity of the bond graph model
    data_path='./data/'
    result_csv=data_path+'report_task_Terkildsen_NaK_BG_Fig5EA.csv'
    # calculate the energy and activity of the bond graph model
    # 15 reactions R1-R15
    reaction_list=['R1','R2','R3','R4','R5','R6','R7','R8','R9','R10','R11','R12','R13','R14','R15']
    # 15 storage components, P1...P15, generate a list
    storage_list=['P1','P2','P3','P4','P5','P6','P7','P8','P9','P10','P11','P12','P13','P14','P15']
    P_comp_expr_dict={}
    for reaction in reaction_list:
        P_comp_expr_dict[reaction]=sympify(f'Af_{reaction}')*sympify(f'v_{reaction}')-sympify(f'Ar_{reaction}')*sympify(f'v_{reaction}')
    
    for storage in storage_list:
        P_comp_expr_dict[storage]=sympify(f'v_{storage}')*sympify(f'mu_{storage}')
    
    calc_energy(P_comp_expr_dict,result_csv)
    
    result_csv=data_path+'report_task_New_Terkildsen_NaK_BG_Fig5_13EA.csv'
    # calculate the energy and activity of the bond graph model
    # 15 reactions R1-R15
    reaction_list=['R1','R2','R3','R5','R6','R7','R8','R9','R11','R12','R13','R14','R15']
    # 15 storage components, P1...P15, generate a list
    storage_list=['P1','P2','P3','P5','P6','P7','P8','P9','P11','P12','P13','P14','P15']
    P_comp_expr_dict={}
    for reaction in reaction_list:
        P_comp_expr_dict[reaction]=sympify(f'Af_{reaction}')*sympify(f'v_{reaction}')-sympify(f'Ar_{reaction}')*sympify(f'v_{reaction}')
    
    for storage in storage_list:
        P_comp_expr_dict[storage]=sympify(f'v_{storage}')*sympify(f'mu_{storage}')
    
    calc_energy(P_comp_expr_dict,result_csv)