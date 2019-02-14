"""

collect_agents.py: Collect agents code from git repo

"""

__copyright__ = "Copyright 2019, MoO"
__license__ = ""
__author__ = "Mostafa Rafaie"
__maintainer__ = "Mostafa Rafaie"

from git import Repo
import os
import glob
import shutil


REPO_URL="https://bitbucket.mutualofomaha.com/scm/raf/openai-maze-challenge.git"
TMP_PATH="TEMP"
IGONRED_BRANCHS=['master', 'development', 'dev_server']
IGONRED_AGENTS=['base_agent.py', 'example.py', '__init__.py']
AGENTS_PATH="agents"
MAX_AGENTS=2

def clone_project(repo_url, path):
    return Repo.clone_from(repo_url, path)

def get_branch_list(repo, ignored_branchs=IGONRED_BRANCHS):
    all_b_list = repo.git.branch('-r').split('\n')
    b_list = []
    for b in all_b_list:
        if len(b.split()) == 1:
            b = b.split('/')[1]
            if  b not in ignored_branchs:
                b_list.append(b)
    return b_list

def pull_agents(repo, branch_list, tmp_path=TMP_PATH, agents_path=AGENTS_PATH, ignored_agents=IGONRED_AGENTS, max_agents=MAX_AGENTS):
    for b in branch_list:
        print('--------------------------------------')
        print("Start processing the branch '{}'".format(b))
        repo.git.checkout(b)
        files = glob.glob(os.path.join(tmp_path, agents_path, "*.py"))
        files.sort(key=os.path.getmtime, reverse=True)
        counter = 1
        for f in files:
            fname = os.path.basename(f)
            if  fname not in ignored_agents:
                fname2 = b + '_' + fname
                shutil.copyfile(f, os.path.join(agents_path, fname2))
                print("Copy file '{}' as a new name '{}' to '{}' folder.".format(fname, fname2, agents_path))
                counter += 1
            if counter > max_agents:
                print("Max agent is extrated!")
                break

def reset_base_repo(tmp_path=TMP_PATH, agents_path=AGENTS_PATH, ignored_agents=IGONRED_AGENTS):
    if os.path.isdir(tmp_path):
        # remove temp folder
        shutil.rmtree(tmp_path)
    
    # remove all agents
    for f in glob.iglob(os.path.join(agents_path, '*.py')):
        if os.path.basename(f) not in ignored_agents:
            os.remove(f)

    # Reset main Repo
    repo = Repo('.')
    repo.git.reset()
    print('--------------------------------------')
    print("Reset current repo and cleanup '{}' folder.".format(agents_path))

def prepare_tournament_code(rrepo_url=REPO_URL, tmp_path=TMP_PATH,
                            ignored_branchs=IGONRED_BRANCHS, agents_path=AGENTS_PATH,
                            ignored_agents=IGONRED_AGENTS ):
    reset_base_repo(tmp_path, agents_path, IGONRED_AGENTS)
    repo = clone_project(repo_url=REPO_URL, path=TMP_PATH)
    branch_list =  get_branch_list(repo, IGONRED_BRANCHS)
    pull_agents(repo, branch_list, tmp_path=TMP_PATH, agents_path=agents_path, 
                ignored_agents=IGONRED_AGENTS, max_agents=MAX_AGENTS)


if __name__ == "__main__":
    prepare_tournament_code(repo_url=REPO_URL, tmp_path=TMP_PATH,
                            ignored_branchs=IGONRED_BRANCHS, agents_path=AGENTS_PATH,
                            ignored_agents=IGONRED_AGENTS )

    reset_base_repo(TMP_PATH, AGENTS_PATH, IGONRED_AGENTS)


