import numpy as np

# training data
raw = np.array([[1,2],
                [2,4],
                [3,6],
                [4,8],
                [5,10],
                [6,12]])

# tách dữ liệu
y = raw[:,1]

# tạo X = [1 x]
X = np.c_[np.ones(len(y)), raw[:,0]]

m = len(y)

# khởi tạo theta = [0,0]
theta = np.zeros(2)

alpha = 0.1

#hàm tính độ chính xác(J0)
def cost(theta):  
    pred = X @ theta
    return (1/(2*m)) * np.sum((pred - y)**2)

# train, hàm tính gradient descent
for i in range(1000):
    pred = X @ theta
    grad = (1/m) * (X.T @ (pred - y))
    theta = theta - alpha * grad
    
#theta2 = np.round(theta) #làm tròn đến hàng đơn vị
#theta3 = np.ceil(theta) #làm tròn lên trong mọi trường hợp
#theta4 = np.round(theta, decimals = 1) #làm tròn đến (decimals) chữ số thập phân
print("theta = ", theta)   
print(f"{theta[0]:.6f}, {theta[1]:.6f}")#theta chính xác
#print(theta)#theta lm tròn
print("J =", cost(theta))#độ chính xác(J0)

def main():
    kq = int()
    while 1:
        x = int(input('x = '))
        if x != 0:
            x = [1,x]
            x =  np.array(x)
            kq = theta @ x
            print(kq)
            #print(round(theta @ x))
        else: break

main()
