'''
상속
'''
class A:
    @classmethod
    def printInfo(cls):
        print("A")

class B:
    @classmethod
    def printInfo(cls):
        print("B")

class AB(A, B):
    pass

ab1 = AB()
ab1.printInfo() # A가 더 상단에 있어서 A가 출력됨