from os import listdir
from os.path import join, isdir, isfile
from re import sub
from shutil import copy

def list_files(loc, return_dirs=False, return_files=True, recursive=False, valid_exts=None):
    """
    Return a list of all filenames within a directory loc.

    Inputs:
        loc - Path to directory to list files from.
        return_dirs - If true, returns directory names in loc. (default: False)
        return_files - If true, returns filenames in loc. (default: True)
        recursive - If true, searches directories recursively. (default: False)
        valid_exts - If a list, only returns files with extensions in list. If None,
            does nothing. (default: None)

    Outputs:
        files - List of names of all files and/or directories in loc.
    """
    
    files = [join(loc, x) for x in listdir(loc)]

    if return_dirs or recursive:
        # check if file is directory and add it to output if so
        is_dir = [isdir(x) for x in files]
        found_dirs = [files[x] for x in range(len(files)) if is_dir[x]]
    else:
        found_dirs = []

    if return_files:
        # check if file is not directory and add it to output
        is_file = [isfile(x) for x in files]
        found_files = [files[x] for x in range(len(files)) if is_file[x]]
    else:
        found_files = []

    if recursive and not return_dirs:
        new_dirs = []
    else:
        new_dirs = found_dirs

    deeper_files = []
    if recursive:
        for d in found_dirs:
            deeper_files.extend(list_files(d, 
                                           return_dirs=return_dirs, 
                                           return_files=return_files,
                                           recursive=recursive))

    if isinstance(valid_exts, (list, tuple)):
        concat_files = found_files + deeper_files
        new_files = []
        for e in valid_exts:
            new_files.extend([f for f in concat_files if f.endswith(e)])
    else:
        new_files = found_files + deeper_files

    return new_dirs + new_files

def get_nested_dirs(locs=['.']):
    """
    Return a list of all folders and subfolders within the directories in the list
    locs.

    Inputs:
        locs - String of path or list of paths to directories to search.

    Outputs:
        dirs - List of paths to all subfolders of locs.
    """

    # if locs is not a list, place it in one
    if not isinstance(locs, (list, tuple)):
        locs = [locs]

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
