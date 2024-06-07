import os, re

path = "C:\\Users\\amy_l\\PycharmProjects\\FEA\\results\\factorarchive\\WFG7\\NSGA3"
pattern = "^(NSGA3_WFG7)(.*)"
replace = r"NSGA3_4partitions_WFG7\2"

comp = re.compile(pattern)
for f in os.listdir(path):
    full_path = os.path.join(path, f)
    if os.path.isfile(full_path):
        match = comp.search(f)
        if not match:
            continue

        try:
            new_name = match.expand(replace)
            new_name = os.path.join(path, new_name)
        except re.error:
            continue

        if os.path.isfile(new_name):
            print("%s -> %s skipped" % (f, new_name))
        else:
            os.rename(full_path, new_name)
