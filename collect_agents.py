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
AGENT_FOLDER="agents"
MAX_AGENT=2

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

def pull_agents(repo, branch_list, tmp_path=TMP_PATH, agent_folder=AGENT_FOLDER, ignored_agents=IGONRED_AGENTS):
    for b in branch_list:
        repo.git.checkout(b)
        for f in glob.iglob(os.path.join(tmp_path, agent_folder, "*.py")):
            fname = os.path.basename(f)
            if  fname not in ignored_agents:
                fname2 = b + '_' + fname
                shutil.copyfile(f, os.path.join(agent_folder, fname2))


def prepare_tournament_code(repo_url, tmp_path, ignored_branchs, agent_folder,
                            ignored_agents):
    repo = clone_project(repo_url=REPO_URL, path=TMP_PATH)
    branch_list =  get_branch_list(repo, IGONRED_BRANCHS)
    pull_agents(repo, branch_list, tmp_path=TMP_PATH, agent_folder=AGENT_FOLDER, ignored_agents=IGONRED_AGENTS)


def reset_base_repo(repo_url, tmp_path):
    pass

if __name__ == "__main__":
    prepare_tournament_code(repo_url=REPO_URL, tmp_path=TMP_PATH,
                            ignored_branchs=IGONRED_BRANCHS, agent_folder=AGENT_FOLDER,
                            ignored_agents=IGONRED_AGENTS )


