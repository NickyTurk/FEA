def definition_to_string( definition):
    return "-".join(map(str, list(definition)))
# def

def definition_to_file( definition, dir="results/", ext=".json"):
    base = definition_to_string( definition)
    return dir + base + ext
# def
