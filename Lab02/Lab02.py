# Lab Week 02 Assignment, From CSE SYSU

from typing import *


class StuData:
    data = []

    def __init__(self):
        f = open("student_data.txt", "r")
        while 1:
            input = f.readline()
            if input == "":
                break
            input = input.rstrip()
            self.data.append(input.split(" "))
        f.close()

    def AddData(self, name: str, stu_num: str, gender: str, age: int):
        age = str(age)
        self.data.append([name, stu_num, gender, age])

    def SortData(self, index: str):
        if index == 'name':
            self.data.sort(key=lambda x: x[0])
        elif index == 'stu_num':
            self.data.sort(key=lambda x: x[1])
        elif index == 'gender':
            self.data.sort(key=lambda x: x[2])
        elif index == 'age':
            self.data.sort(key=lambda x: x[3])
        else:
            return

    def ExportFile(self, filename: str):
        f = open(filename, "w")
        for op in self.data:
            f.write(f"{' '.join(op)}\n")
        f.close()


if __name__ == '__main__':
    s = StuData()
    s.AddData(name="Bob", stu_num="003", gender="M", age=20)
    print(s.data)
    s.SortData("stu_num")
    print(s.data)
    s.ExportFile("new_student_data.txt")
