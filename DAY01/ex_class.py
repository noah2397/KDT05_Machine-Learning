# 클래스 생성 (1)
# - 구성요소 : 속성 + 메서드 => 모두 없는 클래스
# - 기본 상속 : Object => __속성명__, __메서드명__()
# ---------------------------------
class A:
    pass

# 클래스 생성 (2)
# - 구성요소 : 속성 + 메서드 => 모두 없는 클래스
# - 기본 상속 : Object => __속성명__, __메서드명__()
# ---------------------------------
class B:
    # 인스턴스 객체 생성 및 속성 초기화 메서드
    def __init__(self, num, name):
        # self로 지정된 힙 메모리 주소에서 부터 속성 저장
        self.num = num
        self.name = name

    # 인스턴스 메서드 오버라이딩
    def printInfo(self):
        print(f"num : {self.num}")
        print(f"name : {self.name}")

    # 연산자 맵핑 메서드 구현
    # 연산자 +와 맵핑된 메서드
    def __add__(self, other):
        print("__add__")
        return self.num + other.num

    def __sub__ (self, other):
        print("__sub__")
        return self.num - other.num

# 클래스 생성 (3)
# - 구성요소 : 속성 + 메서드 => 모두 없는 클래스
# - 기본 상속 : Object => __속성명__, __메서드명__()
# ---------------------------------
class C:
    # 클래스 변수 => C클래스로 생성된 모든 인스턴스에서 공유
    #            => 인스턴스 생성 없이 사용 가능
    # 인스턴스 객체 생성 및 속성 초기화 메서드
    loc = "Daegu"
    def __init__(self, num, name):
        # self로 지정된 힙 메모리 주소에서 부터 속성 저장
        self.num = num
        self.name = name

    # 인스턴스 메서드 오버라이딩
    def printInfo(self):
        print(f"num : {self.num}")
        print(f"name : {self.name}")



# 클래스 생성 (4)
# - 구성요소 : 속성 + 메서드 => 모두 없는 클래스
# - 기본 상속 : Object => __속성명__, __메서드명__()
# ---------------------------------
class DCalc:
    # 클래스 변수 => 클래스로 생성된 모든 인스턴스에서 공유
    #            => 인스턴스 생성 없이 사용 가능
    # 인스턴스 객체 생성 및 속성 초기화 메서드
    name = "CASIO"

    @classmethod
    def addNum(cls, a, b):
        print(cls.name)
        print(DCalc.name) # 윗줄이랑 같은 표현임
        return a+b

    @classmethod
    def minusNum(a, b):
        return a-b



# 객체/인스턴스 생성 ----------------------
# 생성 함수: 클래스이름()
# --------------------------------------
a1 = A()
b1 = B(100, "BB")
b2 = B(30, "B2")
c1 = C(100, "CC")

# 객체/인스턴스의 연산 ----------------------------------------------
print("ABC"+"123")
print([1, 2, 3] + [10, 20, 30])
print("=====================>", b1 + b2)
print("=====================>", b1 - b2)


# 객체/인스턴스의 속성/메서드 사용-------------------------------------
# 사용 방법 : 객체/인스턴스 변수명.속성
# 기본 상속 : Object
print("A 인스턴스 a1의 속성과 메서드 => ", a1.__dict__)
print("A 클래스의 속성과 메서드 => ", A.__dict__)
# print("A 클래스의 속성과 메서드 => ", (A.__dir__()))

print("B 클래스의 속성과 메서드 => ", B.__dict__) # dict는 속성과 메서드가 다 나옴

# 인스턴스 메서드 사용
c1.printInfo()

# 인스턴스 속성 사용
print(c1.name)

# 클래스 속성 사용
print("loc => ", C.loc)

# 인스턴스 메서드는 인스턴스 생서 안하고 못씀(= 클래스명만으로는 사용 불가)
# C.printInfo() # error

# 클래스 속성 및 메서드 사용
print(f"Dcalc.name : {DCalc.name}")
print(f"Dcalc.addNum : {DCalc.addNum(10, 20)}")
print(f"Dcalc.minusNum : {DCalc.minusNum(20, 20)}")