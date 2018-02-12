import numpy as np
import matplotlib.pyplot as plt
import math

root3 = np.sqrt(3)

def genData(num_tri):
    '''
    Generate dataset of a num_tri of triangles and a num_hax of hexagons
    The data for each is num_tri samples * 10 steps * 19 whisker distances
    '''
    tri_data = np.ones((num_tri, 10, 19))
    config = np.zeros((num_tri, 10, 7))
    for i in range(0, num_tri):
        x = np.random.randint(5, 16, dtype=np.uint8)
        y = np.random.randint(5, 16, dtype=np.uint8)
        t = np.random.uniform(0, 2 * np.pi, dtype=np.float32)
        s = np.random.uniform(6, 17, dtype=np.float32)
        for j in range(0, 5):
            X = np.random.randint(0, 21, dtype=np.uint8)
            Y = np.random.randint(0, 21, dtype=np.uint8)
            Z = np.random.randint(1, 11, dtype=np.uint8)
            tri_data[i][j] = getDist(0, x, y, t, s, X, Y, Z)
        for j in range(5, 10):
            X = x
            Y = y
            Z = np.random.randint(1, 10, dtype=np.uint8)
            tri_data[i][j] = getDist(0, x, y, t, s, X, Y, Z)
        return tri_data

def decodeData(tri_data, i, j):





def getDist(label, x, y, t, s, X, Y, Z):
    '''
    Input:

    Label = 0 is equilateral triangle, label = 1, is equilateral hexagon;
    For triangle, (x, y) is a corner such that other corners are (s, 0) and (0.5*s, 1.732*s/2);
    for non-zero t, further rotates the x-y Coordinates counter clockwise by t. s is the side length.
    For hexagon, (x, y) is the center coordinates oriented such that x-axis aligns with the (s,0)
    side of a composing equilateral triangle; t is counterclockwise rotation; s is side length.
    X, Y, Z are coordinates for the head position of the laser / whisker array. Whiskers point
    outwards.The inner circle of whiskers are tiled 30 degree away from center whisker, while the
    outer circle of whiskers are tilted 45 degree away from the center whisker.
    X ~ (0, 20),
    Y ~ (0, 20),
    Z ~ (1, 10), bottom left corner is (0, 0, 0) and top right corner is (20, 20, 0) at z = 0
    s ~ (6, 16) for triangle, (3, 8) for hexagon
    t ~ (0, 2pi)
    x ~ (5, 15)
    y ~ (5, 15)

    Output:

    Return an array of 19 L2 distances between the root of whisker 0 - 18 and the x-y plane at Z = 0.
    Whisker 0 is center whisker. Whisker 1 - 6 starts at (1, 0) counterclockwise in the inner circle.
    Whisker 7 - 18 starts at (1.732, 0) counterclockwise in the outercircle. The outer circle twice as
    dense as inner circle. If laser lands outside of the shape, returned length is 1000.
    '''
    # distance from head to contact regardless if within shape
    cont_pos, head_to_cont = getContactPos(X, Y, Z)
    # check whether the contact positions are within the labeled shape
    on_shape, bound_vec = onShape(label, x, y, t, s, cont_pos)
    # draw and return measured distances
    # drawObserv(on_shape, bound_vec, cont_pos, x, y)
    return on_shape * head_to_cont + ~on_shape * 1000


def getContactPos(X, Y, Z):
    '''
    return an array of global contact positions of 19 whiskers at surface Z = 0
    '''
    # relative distance away from the center point on contact surface
    cont_pos = np.ones((2, 19)) * Z # outer whisker circle
    cont_pos[:, 0] = 0 # center contact point
    cont_pos[:, 1:7] /= root3 # inner whisker circle
    head_to_cont = np.sqrt(cont_pos[0,:]**2 + Z**2) # calc dist from head to contact
    # now transform into positions relative to in head frame
    whis_ang = np.array([[0, 0, 60, 120, 180, 240, 300, 0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]])
    whis_ang = whis_ang / 180.0 * np.pi
    cont_pos[0,:] = cont_pos[0,:] * np.cos(whis_ang) # x relative to head
    cont_pos[1,:] = cont_pos[1,:] * np.sin(whis_ang) # y relative to head
    # now transform from head frame to global frame
    cont_pos += np.array([[X],[Y]])
    return cont_pos, head_to_cont


def onShape(label, x, y, t, s, cont_pos):
    '''
    return True for a contact point within the boundary of the label shape, else False
    '''
    s /= (label+1.0) # adjust s for different label
    s1 = s / 2.0
    s2 = s1 * root3
    # boundary vectors in frame origined at (x, y) before rotation t
    bound_vec = np.array([[s, 0], [s1, s2], [-s1, s2], [-s, 0], [-s1, -s2], [s1, -s2]]).T
    if label == 0: # triangle
        bound_vec = bound_vec[:, :2]
    bound_vec = np.dot(np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]]), bound_vec) # rotate ccw by t
    rela_cont_pos = cont_pos - np.array([[x], [y]]) # relative to shape centered frame
    # check if within shape by dot products with boundary vectors
    on_shape = np.full((1, cont_pos.shape[1]), False)
    for i in range(0, bound_vec.shape[1]):
        new_mask = within_tri(np.hstack((bound_vec[:,[i]], bound_vec[:, [(i+1) % bound_vec.shape[1]]])), rela_cont_pos)
        on_shape = np.logical_or(on_shape, new_mask)
    return on_shape, bound_vec


def within_tri(A, b):
    '''
    return True if point b is within the triangle spanned by the two column vectors of A
    '''
    on_shape = np.linalg.solve(A, b)
    mask1 = np.all((on_shape >= 0), axis=0, keepdims=True)
    mask2 = np.all((on_shape <= 1), axis=0, keepdims=True)
    mask3 = np.sum(on_shape, axis=0, keepdims=True) <= 1
    return np.all(np.vstack((mask1, mask2, mask3)), axis=0, keepdims=True)


def drawObserv(on_shape, bound_vec, cont_pos, x, y):
    plt.ion() # interactive mode
    plt.figure(figsize=(8,8))
    plt.xlim(-4,4)
    plt.ylim(-4,4)
    bound_vec = np.insert(bound_vec, 0, 0, axis=1) # append origin at front
    bound_vec = bound_vec + np.array([[x], [y]]) # transform to global frame
    plt.triplot(bound_vec[0,:], bound_vec[1,:])
    plt.plot(cont_pos[0, on_shape[0]], cont_pos[1, on_shape[0]], 'r.')
    plt.plot(cont_pos[0, ~on_shape[0]], cont_pos[1, ~on_shape[0]], 'k.')
