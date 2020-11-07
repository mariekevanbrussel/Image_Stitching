import cv2
import numpy as np
from random import sample
from matplotlib import pyplot as plt

def get_coords(keypoints):
    return np.asarray(tuple(map(lambda x: tuple(map(lambda y: round(y), x)), map(lambda x: x.pt, keypoints))))

def gen_A(coords):
    A_list = []
    for coord in coords:
        A_list.append([coord[0], coord[1], 0, 0, 1, 0])
        A_list.append([0, 0, coord[0], coord[1], 0, 1])
    return np.stack(A_list)

def gen_b(coords):
    return coords.flatten()

def map_img(img, x):
    # Get a set of image coordinates
    coords = []
    for i in range(img.shape[1]):
        for j in range(img.shape[0]):
            coords.append((i,j))
    coords = np.asarray(coords)

    # Generate A (this representation is necessary for the transformation)
    A = gen_A(coords)

    # Apply the transformation
    new_coords = np.matmul(A, x).reshape(coords.shape).round().astype('int64')

    # Get the largest indices to predefine the shape of the new image
    b, a = np.amax(new_coords, 0)

    # First create an empty image of the right size, then map the original image onto it with the new coordinates
    new_img = np.zeros((a+1, b+1, img.shape[2])).astype(img.dtype)
    for i in range(len(coords)):
        org_coord1 = coords[i,0]
        org_coord2 = coords[i,1]
        new_coord1 = new_coords[i,0]
        new_coord2 = new_coords[i,1]

        new_img[new_coord2, new_coord1] = img[org_coord2, org_coord1]

    return new_img

def ransac(img1_points, img2_points, P=4, N=10, verbose=False):
    max_count = -1

    for i in range(N):
        # Sample keypoints
        indices = sample(range(len(img1_points)), P)

        img1_sample = []
        img2_sample = []
        for index in indices:
            img1_sample.append(img1_points[index])
            img2_sample.append(img2_points[index])

        # Get the coordinates of the sampled keypoints
        coords1 = get_coords(img1_sample)
        coords2 = get_coords(img2_sample)

        # Generate matrix A and vector b
        A = gen_A(coords1)
        b = gen_b(coords2)

        # Solve Ax = b for x using the pseudo-inverse
        x = np.matmul(np.linalg.pinv(A),b)

        # Apply the transformation
        new_coords = np.matmul(A, x).reshape(coords1.shape)

        # Count the inliers and error sum
        inliers = []
        inner_count = 0
        total_dist = 0
        for j in range(coords1.shape[0]):
            dist = np.linalg.norm(coords2[j] - new_coords[j])
            total_dist += dist
            if dist <= 10:
                inner_count += 1
                inliers.append(coords1[j])

        # Update the weights and the respective inliers if this iteration performs best (e.g. has the largest amount of
        # inliers)
        if inner_count > max_count:
            if verbose:
                print("New largest amount of inliers found at iteration {}: {}".format(i, inner_count))
            max_count = inner_count
            best_inliers = inliers
            min_total_dist = total_dist

        # If the number of inliers is equal to the previous best, decide using the error sum
        elif inner_count == max_count:
            if total_dist < min_total_dist:
                if verbose:
                    print("Smaller error sum at iteration {}".format(i))
                best_inliers = inliers
                min_total_dist = total_dist

    if verbose:
        print("Number of inliers:",max_count)

    return x, best_inliers

