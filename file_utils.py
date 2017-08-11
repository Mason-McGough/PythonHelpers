from os import listdir
from os.path import join, isdir, isfile
from re import sub
from shutil import copy

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
        # get files with full path
        files = [join(loc, x) for x in listdir(loc)]

        # check if file is directory and add it to output if so
        is_dir = [isdir(x) for x in files]
        new_dirs = [files[x] for x in range(len(files)) if is_dir[x]]
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
    files = [join(loc, x) for x in listdir(loc)]

    # check if file is not directory and add it to output
    is_file = [isfile(x) for x in files]
    new_files = [files[x] for x in range(len(files)) if is_file[x]]
    
    # copy files in list to dest
    for this_file in new_files:
        # change name if renaming
        if rename:
            # replace slashes with hyphens to preserve unique name
            out_file = sub(r'\\|/', '-', this_file)
            out_file = join(dest, out_file)
            copy(this_file, out_file)
        else:
            copy(this_file, dest)
    
    return new_files