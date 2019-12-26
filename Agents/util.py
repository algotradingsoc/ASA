def check_if_in_agents():
    '''
    Simple function to check if the current working directory
    is in the agents folder.
    '''
    import os
    cwd = os.getcwd()
    cwd = cwd.split('/')
    assert cwd[-1] == 'agents', "Not running python from the agents file."
