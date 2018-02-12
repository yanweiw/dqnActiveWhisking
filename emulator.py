import numpy as np
import matplotlib.pyplot as plt
import math

def getPb(h,a,b):
    # Pb = np.zeros(19*2).reshape(2,19)
    Pb = np.zeros((2, 19))
    ra = h*np.tan(a)
    rb = h*np.tan(b)
    for i in range(1,7):
        Pb[0,i] = ra * np.cos(np.radians(-30 + 60*i))
        Pb[1,i] = ra * np.sin(np.radians(-30 + 60*i))
    for i in range(7,19):
        Pb[0,i] = rb * np.cos(np.radians(30 * (i-8)))
        Pb[1,i] = rb * np.sin(np.radians(30 * (i-8)))
    return Pb

def getDist(h,a,b):
    dist = np.zeros(19).reshape(1,19)
    dist[0] = h
    for i in range(1,7):
        dist[0,i] = h/np.cos(a)
    for i in range(7,19):
        dist[0,i] = h/np.cos(b)
    return dist

def emulator_tri(x,y,z,t):
    # Define laser
    a = 0.1
    b = 0.2
    # Define triangle
    v0 = [[-2],[-2]]
    v1 = [[4],[0]]
    v2 = [[2],[4]]
    # Coordinate transformation
    pb = getPb(z,a,b)
    dist = getDist(z, a, b)
    Pb = np.concatenate((pb,np.ones(19).reshape(1,19)))
    Tsb = [[np.cos(t),-np.sin(t),x],[np.sin(t),np.cos(t),y],[0,0,1]]
    Ps = np.dot(Tsb,Pb)
    ps = Ps[0:2]
    # Check position
    ab = np.dot(np.linalg.inv(np.concatenate((v1,v2), axis = 1)),(ps-v0))
    res = np.ones(19).reshape(1,19)*1000
    for i in range(0,19):
        if ab[0,i] >= 0 and ab[1,i] >= 0 and np.sum(ab[:,i], axis = 0) <= 1:
            res[0,i] = 1

    # Check plot
    plt.triplot([-2, 2, 0], [-2, -2, 2])
    for i in range(0,19):
        if res[0,i] == 1:
            plt.plot(Ps[0, i], Ps[1, i], 'k.')
        else:
            plt.plot(Ps[0, i], Ps[1, i], 'r.')
    plt.show()

    return res*dist

def emulator_hex(x,y,z,t):
    # Define laser
    a = 0.1
    b = 0.2
    # Coordinate transformation
    pb = getPb(z, a, b)
    dist = getDist(z, a, b)
    Pb = np.concatenate((pb, np.ones(19).reshape(1, 19)))
    Tsb = [[np.cos(t), -np.sin(t), x], [np.sin(t), np.cos(t), y], [0, 0, 1]]
    Ps = np.dot(Tsb, Pb)
    ps = Ps[0:2]
    # Check position
    res = np.ones(19).reshape(1, 19) * 1000
    for i in range(0,19):
        if ps[0,i] >= 0 and ps[0,i] <= np.sqrt(3) \
                and ps[1,i] <= -np.sqrt(3)/3*ps[0,i]+2 \
                and ps[1,0] >= np.sqrt(3)/2*ps[0,i]-2:
            res[0,i] = 1
        elif ps[0,i] < 0 and ps[0,i] >= -np.sqrt(3) \
                and ps[1,i] <= np.sqrt(3)/3*ps[0,i]+2 \
                and ps[1,0] >= -np.sqrt(3)/2*ps[0,i]-2:
            res[0,i] = 1

    # Check plot
    plt.plot([np.sqrt(3), 0, -np.sqrt(3), -np.sqrt(3), 0, np.sqrt(3), np.sqrt(3)], \
             [1, 2, 1, -1, -2, -1, 1], 'b-')
    for i in range(0,19):
        if res[0,i] == 1:
            plt.plot(Ps[0, i], Ps[1, i], 'k.')
        else:
            plt.plot(Ps[0, i], Ps[1, i], 'r.')
    plt.show()

    return res*dist

# Check
x = 1; y = 0; z = 2; t = math.pi/8
result = emulator_tri(x,y,z,t)
print(result)
