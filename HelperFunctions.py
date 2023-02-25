def flatten(l):
    """
    Function to flatten a list (e.g. turn a list of lists into a list)
    :param l: List input ; dtype : List
    :return: Flattened List (removal of one inner list layer)
    """
    return [item for sublist in l for item in sublist]
