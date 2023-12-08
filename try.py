import time
def printTest(n):
    def p(*args, **kwargs):
        print("start")
        n(*args, **kwargs)
        print("end")
        return True
    return p
@printTest
def p2(x,y):
    print("x + y = {}".format(x+y))

if __name__ == '__main__':
    p2(1,2)