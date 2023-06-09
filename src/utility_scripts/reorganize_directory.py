'''Scripts for reorganizing the directory.

   NOTE: Each function is meant to be a stand-alone operation you can enter in 
         the command line, so imports are defined locally inside the function
'''

def move_hyperparams_to_results(models_dir: str, results_dir: str, 
                                src_dir: str = '~/thesis/repo'):
    '''Utility function to save relevant hyperparameters in the results directory 
       for easier reference
    '''
    # import modules
    if 'os' not in dir():
        import os
    pwd = os.getcwd()
    print("current directory:", pwd)
    src_dir = os.path.expanduser(src_dir) if '~' in src_dir else src_dir
    os.chdir(src_dir)
    print("repo directory:", pwd)
    from eval import add_hyperparameters_to_results
    os.chdir('..')
    for exp in ['dependent', 'independent', 'zero-shot']:
        exp_models = os.path.join(models_dir, exp)
        for mdl in os.listdir(exp_models):
            m_path, r_path = os.path.join(models_dir, exp, mdl), os.path.join(results_dir, exp, mdl)
            add_hyperparameters_to_results(m_path, r_path)
    os.chdir(pwd)
    print("moving back to directory:", pwd)


if __name__ == '__main__':
    pass