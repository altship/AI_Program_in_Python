import copy
import re
from typing import *

_VARIABLE = '[xywz]+'

def ResolutionFOL(kb: set[tuple[str]]) -> list[str]:
    clause_set = []
    cl = []
    ans = []
    while len(kb) != 0:
        clause = kb.pop()
        cl.clear()
        for func in clause:
            matchObj = re.match(r'([~A-Za-z]+)\(([a-z,]+)\)', func)
            if matchObj:
                predicate = matchObj.group(1)
                var = matchObj.group(2)
                var = var.split(',')
                l = [predicate] + var
                cl.append(l)
        clause_set.append(copy.deepcopy(cl))
    # print(clause_set)
    for i, j in enumerate(clause_set):
        s = f'{i + 1} ('
        if len(j) == 1:
            s = s + f'{j[0][0]}({",".join(j[0][1:])}),,'
        else:
            for k in j:
                s = s + f'{k[0]}({",".join(k[1:0])}),'
        s = re.sub(r',?$', '', s)
        s = s + '),'
        print(s)
        ans.append(s)
    # print(ans)



if __name__ == '__main__':
    kb = {('On(tony,mike)',), ('On(mike,john)',), ('Green(tony)',), ('~Green(john)',),
          ('~On(xx,yy)', '~Green(xx)', 'Green(yy)')}
    ResolutionFOL(kb)