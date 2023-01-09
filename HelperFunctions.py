def flatten(l):
    """
    :param l: list input
    :return: flattened list (removal of one inner list layer)
    """
    return [item for sublist in l for item in sublist]
