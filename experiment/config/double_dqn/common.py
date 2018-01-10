import os

# make train dir
cwd = os.getcwd()
basename = os.path.basename(cwd)
path = os.path.dirname(cwd)
assert path[-6:] == 'config'

true_train_path = os.path.join(path[:-6], 'train_log', basename)
train_path = os.path.join(cwd, 'train_log')

if not os.path.exists(train_path):
    if not os.path.exists(true_train_path):
        os.makedirs(true_train_path)
    os.system('ln -s {} train_log'.format(true_train_path))

summary_path = os.path.join(train_path, 'summaries')
