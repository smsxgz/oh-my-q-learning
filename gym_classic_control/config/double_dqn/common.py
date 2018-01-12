import os

# make train dir
cwd = os.getcwd()
basename = os.path.basename(cwd)
path = os.path.dirname(cwd)
assert path[-6:] == 'config'

true_train_path = os.path.join(path[:-6], 'train_log', basename)
train_path = os.path.join(cwd, 'train_log')

if not os.path.exists(true_train_path):
    os.makedirs(true_train_path)

if not os.path.exists(train_path):
    os.system('ln -s {} train_log'.format(true_train_path))
elif os.path.realpath(train_path) != true_train_path:
    os.system('rm train_log')
    os.system('ln -s {} train_log'.format(true_train_path))

events_path = os.path.join(train_path, 'events')
if not os.path.exists(events_path):
    os.makedirs(events_path)

models_path = os.path.join(train_path, 'models')
if not os.path.exists(models_path):
    os.makedirs(models_path)
