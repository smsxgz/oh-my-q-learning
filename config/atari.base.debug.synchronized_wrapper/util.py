import os


def make_train_path(train_prefix=None):
    # make train dir
    cwd = os.getcwd()
    path = os.path.dirname(cwd)
    assert path[-6:] == 'config'

    basename = os.path.basename(cwd)
    if train_prefix is not None:
        base_train_path = os.path.join(train_prefix)
        if not os.path.exists(base_train_path):
            os.makedirs(base_train_path)
        make_soft_link(base_train_path, os.path.join(path[:-6], 'train_log'))

    pre_train_path = os.path.join(path[:-6], 'train_log', basename)
    train_path = os.path.join(cwd, 'train_log')

    if not os.path.exists(pre_train_path):
        os.makedirs(pre_train_path)
    make_soft_link(pre_train_path, train_path)
    return train_path


def make_soft_link(base_path, path):
    if not os.path.exists(path):
        os.system('ln -s {} {}'.format(base_path, path))
    elif os.path.realpath(path) != os.path.realpath(base_path):
        os.system('rm {}'.format(path))
        os.system('ln -s {} {}'.format(base_path, path))


train_path = make_train_path('/data1/{}/train_log/atari'.format(
    os.environ['USER']))
