import copy
import re
from typing import *

_VARIABLE = '[xywz]+'
MAX_TIMES = 5000


def converter(literal: str) -> str:         # Convert the literal to its negation
    if literal[0] == '~':
        return literal.lstrip('~')
    else:
        return '~' + literal


def find_equal(clause_set: list[list[list]], visited: list[tuple]) -> (int, int, int, int):
    # Find the equal pattern in clauses
    for i in range(len(clause_set)):
        for j in range(i + 1, len(clause_set)):
            for v in range(len(clause_set[i])):
                for u in range(len(clause_set[j])):
                    if (i, j, v, u) in visited:
                        continue
                    if converter(clause_set[i][v][0]) == clause_set[j][u][0]:
                        return i, j, v, u
    return -1, -1, -1, -1


def ResolutionFOL(kb: set[tuple[str]]) -> list[str]:
    clause_set = []
    cl = []
    ans = []
    while len(kb) != 0:  # Convert the kb to clause list, to help the resolution process.
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
        clause_set.append(copy.deepcopy(cl))  # Add the clause to clause set(deep copy to avoid change origin data)

    for i, j in enumerate(clause_set):  # print the original clause set
        s = f'{i + 1} ('
        for k in j:
            s = s + f'{k[0]}({",".join(k[1:])}),'
        if len(j) == 1:
            s = s + ','
        s = re.sub(r',?$', '', s)
        s = s + '),'
        ans.append(s)
    count = len(clause_set)

    visited = []
    times = 0  # How many times it performed
    while times < MAX_TIMES:
        times += 1

        i, j, v, u = find_equal(clause_set, visited)
        visited.append((i, j, v, u))  # Mark the clause pair as visited, avoid duplicate

        if i == -1:  # If no pattern matched, return empty list
            print("No pattern matched in clause set.")
            return []

        mgu = {}
        mgu.clear()
        for k in range(1, len(clause_set[i][v])):
            matchRe1 = re.match(_VARIABLE, clause_set[i][v][k])
            matchRe2 = re.match(_VARIABLE, clause_set[j][u][k])
            if matchRe1 and matchRe2:  # If both are variables, there is no need to change them
                continue
            elif matchRe1:
                mgu[clause_set[i][v][k]] = clause_set[j][u][k]
            elif matchRe2:
                mgu[clause_set[j][u][k]] = clause_set[i][v][k]
        if not mgu:
            flag = False  # If the relevant constant is not equal, there is no meaning to perform resolution
            for k in range(1, len(clause_set[i][v])):
                if clause_set[i][v][k] != clause_set[j][u][k]:
                    flag = True
                    break
            if flag:
                continue

        cl.clear()
        for k in range(len(clause_set[i])):  # Perform resolution, use temorary clause to store the result,
            # avoid change origin data
            if k == v:
                continue
            temp_clause = copy.deepcopy(clause_set[i][k])
            for g in range(1, len(temp_clause)):
                if temp_clause[g] in mgu:
                    temp_clause[g] = mgu[temp_clause[g]]
            cl.append(copy.deepcopy(temp_clause))
        for k in range(len(clause_set[j])):
            if k == u:
                continue
            temp_clause = copy.deepcopy(clause_set[j][k])
            for g in range(1, len(temp_clause)):
                if temp_clause[g] in mgu:
                    temp_clause[g] = mgu[temp_clause[g]]
            cl.append(copy.deepcopy(temp_clause))

        flag = True  # Check if the clause is already in the clause set, more detailed,
        # if the original clause is a subset of the new clause, it is not necessary to add it
        for r in clause_set:
            flag = True
            if len(r) > len(cl):
                flag = False
                continue
            for n in r:
                if n not in cl:
                    flag = False
                    break
            if flag:
                break
        if flag:
            continue

        clause_set.append(copy.deepcopy(cl))  # Add the new clause to clause set

        addi_char1 = addi_char2 = ''        # print the resolution process into string
        if len(clause_set[i]) > 1:
            addi_char1 = chr(ord('a') + v)
        if len(clause_set[j]) > 1:
            addi_char2 = chr(ord('a') + u)
        s = f'{count + 1}[{i + 1}{addi_char1},{j + 1}{addi_char2}]' + '{'
        for k, g in mgu.items():
            s += f'{k}={g},'
        s = s.rstrip(',') + '} = ('
        for k in clause_set[count]:
            s += f'{k[0]}({",".join(k[1:])}),'
        if len(clause_set[count]) != 1:
            s = s.rstrip(',')
        s += '),'
        ans.append(s)
        count = len(clause_set)
        if len(cl) == 0:
            return ans
    print('Over timed!')            # If the resolution process is over timed, return empty list
    return []


if __name__ == '__main__':
    kb = {('On(tony,mike)',), ('On(mike,john)',), ('Green(tony)',), ('~Green(john)',),
          ('~On(xx,yy)', '~Green(xx)', 'Green(yy)')}
    kb2 = {('GradStudent(sue)',), ('~GradStudent(x)', 'Student(x)'),
           ('~Student(x)', 'HardWorker(x)'), ('~HardWorker(sue)',)}
    ans = ResolutionFOL(kb)         # Test the resolution process, due to the random selection of pop() in set,
    # each time the result may be different(usually there are multiple correct answers)
    for i in ans:
        print(i)
