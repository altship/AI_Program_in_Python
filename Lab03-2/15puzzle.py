# -*- coding: UTF-8 -*-
import heapq
from copy import deepcopy
from time import time
import math


class status:
    def __init__(self, puzzle, spaceUnit, g, h):
        # puzzle是一个16位列表，代表第几个元素在第几号单元
        self.puzzle = puzzle
        # 空格单元所在的序号
        self.spaceUnit = spaceUnit
        self.g = g
        self.h = h
        self.f = g + h

    def __lt__(self, other):
        return self.f < other.f

    def __eq__(self, other):
        return self.puzzle == other.puzzle

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(str(self.puzzle))


class A_Star:
    def __init__(self, bPuzzle, spaceUnit, ePuzzle, dim, mode):
        self.dim = dim
        self.bPuzzle = bPuzzle
        self.ePuzzle = ePuzzle
        self.mode = mode
        # 记录每个元素在ePuzzle中的位置（用于曼哈顿距离计算）
        self.ePuzzleHash = {}
        for i in range(len(ePuzzle)):
            self.ePuzzleHash[ePuzzle[i]] = i
        # 记录成功求解后得到的路径
        self.path = []
        # -----------------------------------------
        # 待探索的节点
        self.openList = []
        # 初始状态
        originStatus = status(bPuzzle, spaceUnit, 0, self.evaluate(bPuzzle))
        self.openList.append(originStatus)
        # 构建最小堆，获取最小f值的节点
        heapq.heapify(self.openList)
        # -----------------------------------------
        # 记录已访问的节点
        self.closeList = set()
        # -----------------------------------------
        # 记录父子节点关系（即路径） puzzle:[parent_puzzle, move]
        self.lastStepQuery = {tuple(bPuzzle): 0}  # 初状态，无需记录父节点，这里用 0 标记回溯的终止
        self.solved = False

    # 启发式函数
    def evaluate(self, curPuzzle):
        res = 0
        if self.mode == 0:
            res = 0
        # 数据错位数
        if self.mode == 1:
            """
                计算数据错位数的启发式函数值,保存到res中
            """
            for i, num in enumerate(curPuzzle):
                if num != 0 and num != self.ePuzzle[i]:  # miss 0 is not a must. you can always count it in,
                    # since after all other numbers are in place, 0 will be in place.
                    res += 1
        # 曼哈顿距离
        elif self.mode == 2:
            """
                计算曼哈顿距离的启发式函数值,保存到res中
                提示: self.ePuzzleHash中存储了目标puzzle每个值所在的位置
                我们采用一维存储puzzle, 与二维坐标的对应可以用 (x = idx // dim, y = idx % dim)
            """
            for i, num in enumerate(curPuzzle):
                if num != 0:
                    x, y = i // self.dim, i % self.dim
                    _x, _y = self.ePuzzleHash[num] // self.dim, self.ePuzzleHash[num] % self.dim
                    res += abs(x - _x) + abs(y - _y)
        return res

    def run(self):
        while self.openList:
            # 获得当前状态
            curStatus = heapq.heappop(self.openList)
            # 如果当前puzzle即是终点，那么整理输出路径
            if curStatus.puzzle == self.ePuzzle:
                move = self.lastStepQuery[tuple(curStatus.puzzle)]
                while move != 0:
                    # 栈式插入
                    self.path.insert(0, move[1])
                    move = self.lastStepQuery[move[0]]
                self.solved = True
                break

            # 需要将当前状态加入closeList(hash)
            self.closeList.add(curStatus)
            # 并添加下一状态
            if curStatus.spaceUnit // self.dim != self.dim - 1:
                """将该条件补全,得到可以将0向下移动一格的条件"""
                # 向下移动一格
                newStatus = deepcopy(curStatus)
                """更新newStatus的spaceUnit坐标 (注意是1维的)"""
                newStatus.spaceUnit += self.dim
                newStatus.puzzle[newStatus.spaceUnit] = curStatus.puzzle[curStatus.spaceUnit]
                newStatus.puzzle[curStatus.spaceUnit] = curStatus.puzzle[newStatus.spaceUnit]
                if newStatus not in self.closeList:
                    # 登记lastStepQuery
                    self.lastStepQuery[tuple(newStatus.puzzle)] = [tuple(curStatus.puzzle),
                                                                   curStatus.puzzle[newStatus.spaceUnit]]
                    # 完善最新状态
                    newStatus.g += 1
                    newStatus.h = self.evaluate(newStatus.puzzle)
                    newStatus.f = newStatus.g + newStatus.h
                    # 按f值插入最小堆
                    heapq.heappush(self.openList, newStatus)

            if curStatus.spaceUnit % self.dim != self.dim - 1:
                """将该条件补全,得到可以将0向右移动一格的条件"""
                # 向右移动一格
                newStatus = deepcopy(curStatus)
                """更新newStatus的spaceUnit坐标 (注意是1维的)"""
                newStatus.spaceUnit += 1
                newStatus.puzzle[newStatus.spaceUnit] = curStatus.puzzle[curStatus.spaceUnit]
                newStatus.puzzle[curStatus.spaceUnit] = curStatus.puzzle[newStatus.spaceUnit]
                if newStatus not in self.closeList:
                    # 记录路径
                    self.lastStepQuery[tuple(newStatus.puzzle)] = [tuple(curStatus.puzzle),
                                                                   curStatus.puzzle[newStatus.spaceUnit]]
                    # 完善最新状态
                    newStatus.g += 1
                    newStatus.h = self.evaluate(newStatus.puzzle)
                    newStatus.f = newStatus.g + newStatus.h
                    # 按f值插入最小堆
                    heapq.heappush(self.openList, newStatus)

            if curStatus.spaceUnit // self.dim != 0:
                """将该条件补全,得到可以将0向上移动一格的条件"""
                # 向上移动一格
                newStatus = deepcopy(curStatus)
                """更新newStatus的spaceUnit坐标 (注意是1维的)"""
                newStatus.spaceUnit -= self.dim
                newStatus.puzzle[newStatus.spaceUnit] = curStatus.puzzle[curStatus.spaceUnit]
                newStatus.puzzle[curStatus.spaceUnit] = curStatus.puzzle[newStatus.spaceUnit]
                if newStatus not in self.closeList:
                    # 记录路径
                    self.lastStepQuery[tuple(newStatus.puzzle)] = [tuple(curStatus.puzzle),
                                                                   curStatus.puzzle[newStatus.spaceUnit]]
                    # 完善最新状态
                    newStatus.g += 1
                    newStatus.h = self.evaluate(newStatus.puzzle)
                    newStatus.f = newStatus.g + newStatus.h
                    # 按f值插入最小堆
                    heapq.heappush(self.openList, newStatus)

            if curStatus.spaceUnit % self.dim != 0:
                """将该条件补全,得到可以将0向左移动一格的条件"""
                # 向左移动一格
                newStatus = deepcopy(curStatus)
                """更新newStatus的spaceUnit坐标 (注意是1维的)"""
                newStatus.spaceUnit -= 1
                newStatus.puzzle[newStatus.spaceUnit] = curStatus.puzzle[curStatus.spaceUnit]
                newStatus.puzzle[curStatus.spaceUnit] = curStatus.puzzle[newStatus.spaceUnit]
                if newStatus not in self.closeList:
                    # 记录路径
                    self.lastStepQuery[tuple(newStatus.puzzle)] = [tuple(curStatus.puzzle),
                                                                   curStatus.puzzle[newStatus.spaceUnit]]
                    # 完善最新状态
                    newStatus.g += 1
                    newStatus.h = self.evaluate(newStatus.puzzle)
                    newStatus.f = newStatus.g + newStatus.h
                    # 按f值插入最小堆
                    heapq.heappush(self.openList, newStatus)

        if not self.solved:
            print("Have no solution!")


class IDA_Star:
    def __init__(self, bPuzzle, spaceUnit, ePuzzle, dim, mode):
        self.dim = dim
        self.bPuzzle = bPuzzle
        self.ePuzzle = ePuzzle
        self.mode = mode
        # 记录每个元素在ePuzzle中的位置（用于曼哈顿距离计算）
        self.ePuzzleHash = {}
        for i in range(len(ePuzzle)):
            self.ePuzzleHash[ePuzzle[i]] = i
        # 元素：操作
        self.path = []
        # -----------------------------------------
        # 初始状态
        self.originStatus = status(bPuzzle, spaceUnit, 0, self.evaluate(bPuzzle))
        # -----------------------------------------
        # 记录已访问的节点
        self.visitedList = set()
        # -----------------------------------------
        # 记录父子节点关系（即路径） puzzle:[parent_puzzle, move]
        self.lastStepQuery = {tuple(bPuzzle): 0}  # 初状态，无需记录父节点，这里用 0 标记回溯的终止
        self.solved = False

    # 启发式函数
    def evaluate(self, curPuzzle):
        res = 0
        if self.mode == 0:
            res = 0
        # 数据错位数
        if self.mode == 1:
            """
                计算数据错位数的启发式函数值,保存到res中
            """
            for i, num in enumerate(curPuzzle):
                if num != 0 and num != self.ePuzzle[i]:
                    res += 1
        # 曼哈顿距离
        elif self.mode == 2:
            """
                计算曼哈顿距离的启发式函数值,保存到res中
                提示: self.ePuzzleHash中存储了目标puzzle每个值所在的位置
                我们采用一维存储puzzle, 与二维坐标的对应可以用 (x = idx // dim, y = idx % dim)
            """
            for i, num in enumerate(curPuzzle):
                if num != 0:
                    x, y = i // self.dim, i % self.dim
                    _x, _y = self.ePuzzleHash[num] // self.dim, self.ePuzzleHash[num] % self.dim
                    res += abs(x - _x) + abs(y - _y)
        return res

    def run(self):
        # 迭代加深
        for max_depth in range(1, 100):
            # 初始化状态和过程存储结构
            self.visitedList = set()
            self.visitedList.add(self.originStatus)
            self.path = []
            self.lastStepQuery = {tuple(bPuzzle): 0}
            self.solved = False

            self.dls(max_depth, self.originStatus)

            if self.solved:
                print("Success at depth: {}".format(max_depth))
                break

    # Depth Limited Search
    def dls(self, max_depth, curStatus):
        # 如果当前puzzle即是终点，那么整理输出路径
        if curStatus.puzzle == self.ePuzzle:
            move = self.lastStepQuery[tuple(curStatus.puzzle)]
            while move != 0:
                # 栈式插入
                self.path.insert(0, move[1])
                move = self.lastStepQuery[move[0]]
            self.solved = True
            return
        # 并添加下一状态
        if curStatus.spaceUnit // self.dim != self.dim - 1:
            """将该条件补全,得到可以将0向下移动一格的条件"""
            # 向下移动一格
            newStatus = deepcopy(curStatus)
            """更新newStatus的spaceUnit坐标 (注意是1维的)"""
            newStatus.spaceUnit += self.dim
            newStatus.puzzle[newStatus.spaceUnit] = curStatus.puzzle[curStatus.spaceUnit]
            newStatus.puzzle[curStatus.spaceUnit] = curStatus.puzzle[newStatus.spaceUnit]
            if newStatus not in self.visitedList:
                # 完善最新状态
                newStatus.g += 1
                newStatus.h = self.evaluate(newStatus.puzzle)
                newStatus.f = newStatus.g + newStatus.h
                if newStatus.f <= max_depth:
                    self.visitedList.add(newStatus)
                    # 登记lastStepQuery
                    self.lastStepQuery[tuple(newStatus.puzzle)] = [tuple(curStatus.puzzle),
                                                                   curStatus.puzzle[newStatus.spaceUnit]]
                    # 下一步dfs
                    self.dls(max_depth, newStatus)

        if curStatus.spaceUnit % self.dim != self.dim - 1:
            """将该条件补全,得到可以将0向右移动一格的条件"""
            # 向右移动一格
            newStatus = deepcopy(curStatus)
            """更新newStatus的spaceUnit坐标 (注意是1维的)"""
            newStatus.spaceUnit += 1
            newStatus.puzzle[newStatus.spaceUnit] = curStatus.puzzle[curStatus.spaceUnit]
            newStatus.puzzle[curStatus.spaceUnit] = curStatus.puzzle[newStatus.spaceUnit]
            if newStatus not in self.visitedList:
                # 完善最新状态
                newStatus.g += 1
                newStatus.h = self.evaluate(newStatus.puzzle)
                newStatus.f = newStatus.g + newStatus.h
                if newStatus.f <= max_depth:
                    self.visitedList.add(newStatus)
                    # 登记lastStepQuery
                    self.lastStepQuery[tuple(newStatus.puzzle)] = [tuple(curStatus.puzzle),
                                                                   curStatus.puzzle[newStatus.spaceUnit]]
                    # 下一步dfs
                    self.dls(max_depth, newStatus)

        if curStatus.spaceUnit // self.dim != 0:
            """将该条件补全,得到可以将0向上移动一格的条件"""
            # 向上移动一格
            newStatus = deepcopy(curStatus)
            """更新newStatus的spaceUnit坐标 (注意是1维的)"""
            newStatus.spaceUnit -= self.dim
            newStatus.puzzle[newStatus.spaceUnit] = curStatus.puzzle[curStatus.spaceUnit]
            newStatus.puzzle[curStatus.spaceUnit] = curStatus.puzzle[newStatus.spaceUnit]
            if newStatus not in self.visitedList:
                # 完善最新状态
                newStatus.g += 1
                newStatus.h = self.evaluate(newStatus.puzzle)
                newStatus.f = newStatus.g + newStatus.h
                if newStatus.f <= max_depth:
                    self.visitedList.add(newStatus)
                    # 登记lastStepQuery
                    self.lastStepQuery[tuple(newStatus.puzzle)] = [tuple(curStatus.puzzle),
                                                                   curStatus.puzzle[newStatus.spaceUnit]]
                    # 下一步dfs
                    self.dls(max_depth, newStatus)

        if curStatus.spaceUnit % self.dim != 0:
            """将该条件补全,得到可以将0向左移动一格的条件"""
            # 向左移动一格
            newStatus = deepcopy(curStatus)
            """更新newStatus的spaceUnit坐标 (注意是1维的)"""
            newStatus.spaceUnit -= 1
            newStatus.puzzle[newStatus.spaceUnit] = curStatus.puzzle[curStatus.spaceUnit]
            newStatus.puzzle[curStatus.spaceUnit] = curStatus.puzzle[newStatus.spaceUnit]
            if newStatus not in self.visitedList:
                # 完善最新状态
                newStatus.g += 1
                newStatus.h = self.evaluate(newStatus.puzzle)
                newStatus.f = newStatus.g + newStatus.h
                if newStatus.f <= max_depth:
                    self.visitedList.add(newStatus)
                    # 登记lastStepQuery
                    self.lastStepQuery[tuple(newStatus.puzzle)] = [tuple(curStatus.puzzle),
                                                                   curStatus.puzzle[newStatus.spaceUnit]]
                    # 下一步dfs
                    self.dls(max_depth, newStatus)


if __name__ == "__main__":

    bPuzzle = [5, 1, 2, 4, 9, 6, 3, 8, 13, 15, 10, 11, 0, 14, 7,
               12]  # [1,2,3,4,5,6,7,8,9,10,11,12,0,13,14,15] # [2,8,1,0,4,3,7,6,5]#
    ePuzzle = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0]  # [1,2,3,8,0,4,7,6,5] #

    """
    mode: 启发式函数类型
        0 - 无启发式函数
        1 - 数码错位数量
        2 - 曼哈顿距离
    """
    begin_time = time()
    # solution = A_Star(bPuzzle=bPuzzle, spaceUnit=bPuzzle.index(0), ePuzzle=ePuzzle, dim=4, mode=0)
    solution = IDA_Star(bPuzzle=bPuzzle, spaceUnit=bPuzzle.index(0), ePuzzle=ePuzzle, dim=int(math.sqrt(len(bPuzzle))),
                        mode=2)
    solution.run()
    end_time = time()
    print("Search Time: {:.3f} s".format(end_time - begin_time))
    if solution.solved:
        print("Move Steps: ", solution.path)
    else:
        print("Failure")
