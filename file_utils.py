from os import listdir
from os.path import join, isdir, isfile
from re import sub
from shutil import copy

def list_files(loc, return_dirs=False, return_files=True):
    """
    Return a list of all filenames within a directory loc.

    Inputs:
        loc - Path to directory to list files from.
    """

    files = [join(loc, x) for x in listdir(loc)]

    if return_dirs:
        # check if file is directory and add it to output if so
        is_dir = [isdir(x) for x in files]
        new_dirs = [files[x] for x in range(len(files)) if is_dir[x]]
    else:
        new_dirs = []

    if return_files:
        # check if file is not directory and add it to output
        is_file = [isfile(x) for x in files]
        new_files = [files[x] for x in range(len(files)) if is_file[x]]
    else:
        new_files = []

    return new_dirs + new_files

def get_nested_dirs(locs=['.']):
    """
    Return a list of all folders and subfolders within the directories in the list
    locs.

    Inputs:
        locs - List of paths to directories to search.

    Outputs:
        dirs - List of paths to all subfolders of locs.
    """

    dirs = []
    for loc in locs:
        # get directories with full path
        new_dirs = list_files(loc, return_files=False, return_dirs=True)

        # traverse directories to the bottom
        nested_dirs = get_nested_dirs(new_dirs)
        dirs.extend(new_dirs)
        dirs.extend(nested_dirs)

    return dirs

def copy_files(loc, dest, rename=False):
    """
    Copy all files in the directory loc to the directory dest.

    Inputs:
        loc - Path to directory to copy files from.
        dest - Path to directory to copy files to.
        rename - Flag to rename files. If set to False, copied files use the same
            basename. If True, the full path is used, with the slash characters
            replaced by hyphens.

    Outputs:
        new_files - List of paths to all copied files.
    """

    # get files with full path
    files = list_files(loc)

    # copy files in list to dest
    for i, this_file in enumerate(files):
        # change name if renaming
        if rename:
            # replace slashes with hyphens to preserve unique name
            out_file = sub(r'^./', '', this_file)
            out_file = sub(r'\\|/', '-', out_file)
            out_file = join(dest, out_file)
            copy(this_file, out_file)
            files[i] = out_file
        else:
            copy(this_file, dest)

    return files
