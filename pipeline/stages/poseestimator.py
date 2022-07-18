import numpy as np
from time import time
import cv2
from cambrian import geometry

from pipeline.core import PipelineStep, PipelineStepIndex
from pipeline.data.surface_type import SurfaceType
from pipeline.data.logging import log_segmentation_image
from pipeline.misc.utils import camera_fov_to_intrinsic_matrix, camera_fov_res_to_intrinsics, focal_to_fov, calculate_plane_xyz

class PoseEstimator:
    def __init__(self, data, image, lines, fov, floor_mask, floor_normal, floor_offset):
        super().__init__()
        self.data = data
        self.image = image
        self.fov = fov
        self.edgelets = compute_edgelets(lines)
        self.floor_mask = floor_mask
        self.floor_normal = floor_normal
        self.floor_offset = floor_offset


    def estimate(self):
        self.estimate_fov()

        img_lr = self.data["downscaled"]
        shape = (img_lr.shape[1], img_lr.shape[0])

        self.vp0 = self.vps[0] / self.vps[0][2]

        vertical_line_inliers = self.inliers[0]
        locations, directions, strengths = self.edgelets

        self.edgelets = (locations[vertical_line_inliers], directions[vertical_line_inliers], strengths[vertical_line_inliers])

        locations, directions, strengths = self.edgelets

        vp_directions = locations - self.vp0[:2]

        # arrange lines from left to right relative to vertical vp
        angles = np.arctan2(vp_directions[:, 1], vp_directions[:, 0])

        s = np.argsort(np.sign(self.vp0[1]) * angles)

        locations = locations[s]

        sx = shape[0] / self.image.shape[1]
        sy = shape[1] / self.image.shape[0]

        self.vp0[:2] *= [sx, sy]
        locations[:, 0] *= sx
        locations[:, 1] *= sy

        self.edgelets = (locations, directions, strengths)

        self.camera, _ = camera_fov_res_to_intrinsics(self.fov, np.array(shape))


    def estimate_fov(self):
        
        e_lines = edgelet_lines(self.edgelets)

        pp = [self.image.shape[1]/2, self.image.shape[0]/2]
        vps=[]
        self.inliers = []

        vertical_edgelet_indices = self.get_edgelets_close_to_dir(self.edgelets,[0,1],.03)

        if vertical_edgelet_indices is not None:
            vp_vertical, votes, inliers_vertical = ransac_vanishing_point(self.edgelets, e_lines, 2000, threshold_inlier=1, max_time=1.0, line_indices=vertical_edgelet_indices)

            if vp_vertical is not None:
                vps.append(vp_vertical)
                self.inliers.append(inliers_vertical)

        horizontal1_edgelet_indices = self.get_edgelets_close_to_dir(self.edgelets, [1, 0], .5)

        if horizontal1_edgelet_indices is not None:
            vp_horizontal1, votes, inliers_horizontal1 = ransac_vanishing_point(self.edgelets, e_lines, 2000, threshold_inlier=1, max_time=1.0, line_indices=horizontal1_edgelet_indices)

            if vp_horizontal1 is not None:
                vps.append(vp_horizontal1)
                self.inliers.append(inliers_horizontal1)

        horizontal2_edgelet_indices = self.get_edgelets_close_to_dir(self.edgelets, [1, 0], .95)
        if vp_horizontal1 is not None:
            horizontal2_edgelet_indices = np.setdiff1d(horizontal2_edgelet_indices, np.nonzero(compute_votes(self.edgelets,vp_horizontal1,10))[0])

        if vp_vertical is not None:
            horizontal2_edgelet_indices = np.setdiff1d( horizontal2_edgelet_indices, np.nonzero(compute_votes(self.edgelets,vp_vertical,10))[0])

        if horizontal2_edgelet_indices is not None:
            vp_horizontal2, votes, inliers_horizontal2 = ransac_vanishing_point(self.edgelets, e_lines, 2000, threshold_inlier=2,
                                                                              max_time=1.0,
                                                                              line_indices=horizontal2_edgelet_indices)

            if vp_horizontal2 is not None:
                vps.append(vp_horizontal2)
                self.inliers.append(inliers_horizontal2)
                horizontal2_edgelet_indices = np.setdiff1d(horizontal2_edgelet_indices,
                                                           np.nonzero(compute_votes(self.edgelets, vp_horizontal2,5))[0])

            vp_horizontal3, votes, inliers_horizontal3 = ransac_vanishing_point(self.edgelets, e_lines, 2000, threshold_inlier=1,
                                                                                max_time=1.0,
                                                                                line_indices=horizontal2_edgelet_indices)
            if vp_horizontal3 is not None:
                vps.append(vp_horizontal3)
                self.inliers.append(inliers_horizontal3)

        vps = np.float32(vps)

        axes = [None, None, None]
        axes_indices = []
        max_det = .9

        centered = (vps[:,:2]/np.dstack((vps[:,2],vps[:,2]))-pp)/pp
        forward_indices = np.logical_and((centered[0,:,0]<2.0), (centered[0,:,1]<2.0))

        index1 = [True,True,True,True]
        if np.sum(forward_indices)>0:
            index1 = forward_indices

        for i in range(len(vps)):
            for j in range(len(vps)):
                if i>=j: continue
                for k in range(len(vps)):
                    if j>=k: continue
                    s = index1[i] + index1[j] + index1[k]
                    if s==0: continue

                    vp0 = vps[i]
                    vp1 = vps[j]
                    vp2 = vps[k]
                    fov_pos = [50,70,90]

                    for f in fov_pos:
                        K = camera_fov_to_intrinsic_matrix(f, w=self.image.shape[1], h=self.image.shape[0])
                        K_inv = np.linalg.inv(K)

                        new_vps = [vp0, vp1, vp2]
                        axes_new = np.dot(K_inv, np.transpose(new_vps)).transpose()
                        lengths = np.linalg.norm(axes_new, axis=-1)
                        axes_new = np.float32(axes_new / np.dstack((lengths, lengths, lengths)))[0]
                        coors = np.argmax(abs(axes_new), 0)
                        axes_new = axes_new[coors]
                        DET = abs(np.linalg.det(axes_new))

                        if DET > max_det:

                            max_det = DET
                            axes_indices = np.int32([i,j,k])
                            axes_indices = axes_indices[coors]
                            axes = np.zeros_like(axes_new)
                            axes = axes_new

                            axes[2] = geometry.unit_vector(np.cross(axes[0],axes[1]))
                            self.fov=f


        if axes[2] is not None:
            j, k,_ = axes_indices
            vp1 = vps[j]
            vp2 = vps[k]
            v1 = vp1[:2] / vp1[2] - pp
            v2 = vp2[:2] / vp2[2] - pp
            p_f = -v1[0] * v2[0] - v1[1] * v2[1]

            if p_f > 0:
                focal_length = np.sqrt(p_f)
                fov_new = np.degrees(focal_to_fov(focal_length, 2 * pp[0]))
                if fov_new>40 and fov_new<110:
                    self.fov = fov_new
                    K = camera_fov_to_intrinsic_matrix(self.fov, w=self.image.shape[1], h=self.image.shape[0])
                    K_inv = np.linalg.inv(K)

                    new_vps = [vp1, vp2]
                    axes_new = np.dot(K_inv, np.transpose(new_vps)).transpose()
                    lengths = np.linalg.norm(axes_new, axis=-1)
                    axes_new = np.float32(axes_new / np.dstack((lengths, lengths, lengths)))[0]
                    axes = np.float32([axes_new[0], axes_new[1], geometry.unit_vector(np.cross(axes_new[0], axes_new[1]))])
                    print("axes1", axes)
            self.floor_normal = axes[2]
            self.floor_normal = -np.sign(self.floor_normal[2])*self.floor_normal

        new_cam, _ = camera_fov_res_to_intrinsics(self.fov, np.array([self.image.shape[1], self.image.shape[0]]))

        plane, depth = calculate_plane_xyz([self.floor_normal*self.floor_offset], width=self.image.shape[1], height=self.image.shape[0], camera=new_cam, max_depth=10)
        plane = plane[0]

        fm = cv2.resize(self.floor_mask, (self.image.shape[1], self.image.shape[0]))

        plane_center = np.mean(plane[fm>.5], axis=0)
        self.floor_offset = np.dot(plane_center, self.floor_normal)

        basis_forward = geometry.unit_vector(np.float32([0, self.floor_normal[2], -self.floor_normal[1]]))
        basis_right = geometry.unit_vector(np.cross(basis_forward, self.floor_normal))

        self.floor_rotation = 0
        if axes[1] is not None:
            self.floor_rotation = -geometry.angle_between(axes[1], np.sign(basis_forward[1] - axes[1,1]) * np.sign(
            basis_forward[0] - axes[1,0])*basis_forward)


        self.vps = vps

        print("fov: %.2f, floor rotation: %.2f degrees" % (self.fov, np.degrees(self.floor_rotation)))

        # M = np.eye(3)
        # from cambrian import utils
        #
        # M = utils.axisAngleToRotationMatrix(floor_normal,floor_rotation)
        #
        # basis_right = np.dot(M, basis_right)
        # basis_forward = np.dot(M, basis_forward)
        #
        # uv_center = np.int32(xyz_to_uv(plane_center, width=img.shape[1], height=img.shape[0], camera=new_cam))
        # pre_pos = np.int32(xyz_to_uv(plane_center, width=img.shape[1], height=img.shape[0], camera=new_cam))
        # color = np.int32(np.random.randint([127, 127, 127], [254, 254, 235]))
        # color = (int(color[0]), int(color[1]), int(color[2]))
        # mask = np.zeros_like(img_dir)
        # a = 50
        # scale = 4.
        #
        #
        # # plane_center = plane_center +.33 * basis_right-.68*basis_forward
        # print(plane_center)
        # for l in range(-a, a):
        #     for k in range(-a, a):
        #
        #         pos = plane_center + l / scale * basis_right + k / scale * basis_forward
        #         pos2 = plane_center + (l + 1) / scale * basis_right + (k - 1) / scale * basis_forward
        #
        #         uv_pos = np.int32(xyz_to_uv(pos, width=img.shape[1], height=img.shape[0], camera=new_cam))
        #         uv_pos2 = np.int32(xyz_to_uv(pos2,width=img.shape[1], height=img.shape[0], camera=new_cam))
        #
        #         if np.isnan(uv_pos[0]) or np.isnan(uv_pos[1]) or np.isnan(uv_pos2[1]) or np.isnan(uv_pos2[0]): continue
        #         if k < a - 1 and k >= -a + 1 and l < a - 1 and l >= -a + 1:
        #             cv2.circle(mask, (int(plane_center[0]), int(plane_center[1])), 3, color, thickness=-1)
        #             cv2.circle(mask, (uv_pos[0], uv_pos[1]), 3, color, thickness=-1)
        #             cv2.line(mask, (pre_pos[0], pre_pos[1]), (uv_pos[0], uv_pos[1]), color=(255, 0, 255), thickness=2)
        #             cv2.line(mask, (pre_pos[0], pre_pos[1]), (uv_pos2[0], uv_pos2[1]), color=(0, 255, 255),
        #                      thickness=2)
        #         pre_pos = uv_pos
        # img_dir[np.logical_and(mask[:, :, 2] > 0, fm>.5)] = mask[np.logical_and(mask[:, :, 2] > 0, fm>.5)]
        #
        # color_index = 0
        # for i in axes_indices:
        #     uv_center = pp
        #     vp =vps[i]/vps[i, 2]
        #     color = [0,0,0]
        #     color[color_index] = 255
        #     color_index += 1
        #     cv2.polylines(img_dir, pts=np.array([[[uv_center[0], uv_center[1]], [vp[0], vp[1]]]], np.int32),
        #                   isClosed=False, color=color, thickness=2)

        # save_dir = '/Users/derrickhart/cb-base/client-visualizers/cambrianar-sites/divinefloor/scenes/bedroom/2-bedroom/'
        # cv2.imwrite(save_dir + "image.jpg", img_dir)

        

    def get_edgelets_close_to_dir(self, edgelets, direction, threshold):

        dots = abs(edgelets[1] * direction)
        close_edgelets = np.nonzero(dots>1.0-threshold)[0]
        return close_edgelets


    def get_edgelets_pointing_to_point(self, edgelets, point,threshold):

        locations, directions, _ = edgelets

        desiredDir = locations - point
        dir_norm = np.linalg.norm(desiredDir, axis=1)
        dir_norm[dir_norm == 0] = 1e-5

        dots = abs(np.sum(desiredDir*directions, axis=1))/dir_norm
        close_edgelets = np.nonzero(dots > 1.0 - threshold)[0]
        return close_edgelets

##############################################################
# Vanishing pt functions

def compute_edgelets(lines):
    """Create edgelets as in the paper.
    Parameters
    ----------
    image: ndarray
        Image for which edgelets are to be computed.
    sigma: float
        Smoothing to be used for canny edge detection.
    Returns
    -------
    locations: ndarray of shape (n_edgelets, 2)
        Locations of each of the edgelets.
    directions: ndarray of shape (n_edgelets, 2)
        Direction of the edge (tangent) at each of the edgelet.
    strengths: ndarray of shape (n_edgelets,)
        Length of the line segments detected for the edgelet.
    """

    locations = []
    directions = []
    strengths = []
    # classes = []

    # check_class = False
    # if len(class_labels)>0:
    #     check_class = True

    for line in lines:
        l = line.data.reshape(4)
        p0, p1 = np.array([l[0], l[1]]), np.array([l[2], l[3]])
        c = (p0 + p1) / 2
        # if check_class:
        #     classes.append(class_labels[int(c[1]), int(c[0])])

        locations.append(c)
        directions.append(p1 - p0)
        strengths.append(np.linalg.norm(p1 - p0))

    # convert to numpy arrays and normalize
    locations = np.array(locations)
    directions = np.array(directions)
    strengths = np.array(strengths)
    # classes = np.array(classes)

    directions = np.array(directions) / \
                 np.linalg.norm(directions, axis=1)[:, np.newaxis]

    return (locations, directions, strengths)


def edgelet_lines(edgelets):
    """Compute lines in homogenous system for edglets.
    Parameters
    ----------
    edgelets: tuple of ndarrays
        (locations, directions, strengths) as computed by `compute_edgelets`.
    Returns
    -------
    lines: ndarray of shape (n_edgelets, 3)
        Lines at each of edgelet locations in homogenous system.
    """
    locations, directions = edgelets[:2]
    normals = np.zeros_like(directions)
    normals[:, 0] = directions[:, 1]
    normals[:, 1] = -directions[:, 0]
    p = -np.sum(locations * normals, axis=1)
    lines = np.concatenate((normals, p[:, np.newaxis]), axis=1)
    return lines


def compute_votes(edgelets, model, threshold_inlier=5, model_indices=[-1,-1]):
    """Compute votes for each of the edgelet against a given vanishing point.
    Votes for edgelets which lie inside threshold are same as their strengths,
    otherwise zero.
    Parameters
    ----------
    edgelets: tuple of ndarrays
        (locations, directions, strengths) as computed by `compute_edgelets`.
    model: ndarray of shape (3,)
        Vanishing point model in homogenous cordinate system.
    threshold_inlier: float
        Threshold to be used for computing inliers in degrees. Angle between
        edgelet direction and line connecting the  Vanishing point model and
        edgelet location is used to threshold.
    Returns
    -------
    votes: ndarry of shape (n_edgelets,)
        Votes towards vanishing point model for each of the edgelet.
    """
    vp = model[:2] / model[2]

    locations, directions, strengths = edgelets[:3]

    est_directions = locations - vp

    dot_prod = np.sum(est_directions * directions, axis=1)
    abs_prod = np.linalg.norm(directions, axis=1)*np.linalg.norm(est_directions, axis=1)
    abs_prod[abs_prod == 0] = 1e-5

    cosine_theta = np.abs(dot_prod / abs_prod)

    theta_thresh = np.cos(threshold_inlier * np.pi / 180)
    good = (cosine_theta > theta_thresh)

    return good

def remove_inliers(model, edgelets, threshold_inlier=10):
    """Remove all inlier edglets of a given model.
    Parameters
    ----------
    model: ndarry of shape (3,)
        Vanishing point model in homogenous coordinates which is to be
        reestimated.
    edgelets: tuple of ndarrays
        (locations, directions, strengths) as computed by `compute_edgelets`.
    threshold_inlier: float
        threshold to be used for finding inlier edgelets.
    Returns
    -------
    edgelets_new: tuple of ndarrays
        All Edgelets except those which are inliers to model.
    """
    inliers = compute_votes(edgelets, model, threshold_inlier) > 0
    locations, directions, strengths = edgelets
    locations = locations[~inliers]
    directions = directions[~inliers]
    strengths = strengths[~inliers]
    # classes = classes[~inliers]
    edgelets = (locations, directions, strengths)
    return edgelets

def ransac_vanishing_point(edgelets, lines, num_ransac_iter=2000, threshold_inlier=5, max_time=10.0, line_indices=None, roi=None
                           , vp=None,camera=None, up_normal=None):
    """Estimate vanishing point using Ransac.
    Parameters
    ----------
    edgelets: tuple of ndarrays
        (locations, directions, strengths) as computed by `compute_edgelets`.
    num_ransac_iter: int
        Number of iterations to run ransac.
    threshold_inlier: float
        threshold to be used for computing inliers in degrees.
    Returns
    -------
    best_model: ndarry of shape (3,)
        Best model for vanishing point estimated.
    Reference
    ---------
    Chaudhury, Krishnendu, Stephen DiVerdi, and Sergey Ioffe.
    "Auto-rectification of user photos." 2014 IEEE International Conference on
    Image Processing (ICIP). IEEE, 2014.
    """
    locations, directions, strengths = edgelets

    if line_indices is not None:
        edgelets = locations[line_indices], directions[line_indices], strengths[line_indices]
        locations, directions, strengths = edgelets
        lines = lines[line_indices]

    num_pts = strengths.size
    arg_sort = np.argsort(-strengths)

    first_index_space = arg_sort[:num_pts // 5]
    second_index_space = arg_sort[:num_pts // 2]

    best_models = None
    best_votes = 0
    model_inliers = None

    t = time()

    for ransac_iter in range(num_ransac_iter):
        if time() - t > max_time or  len(first_index_space)==0 or len(second_index_space)==0:
            inlier_indices = np.nonzero(model_inliers)[0]
            if inlier_indices is None:
                return None, best_votes, model_inliers
            return best_models, best_votes, line_indices[inlier_indices]
        # print(len(first_index_space), first_index_space)
        ind1 = np.random.choice(first_index_space)
        ind2 = np.random.choice(second_index_space)

        if ind1==ind2: continue

        l1 = lines[ind1]
        l2 = lines[ind2]

        current_model = np.cross(l1, l2)

        if np.sum(current_model ** 2) < 1 or current_model[2] == 0:
            # bad
            continue

        angle = 0
        axis_up = [0,0,0]
        if vp is not None:
            K=camera
            axis_up = geometry.unit_vector(np.cross(np.dot(np.linalg.inv(K), vp), np.dot(np.linalg.inv(K), current_model)))
            angle = np.dot(up_normal, -np.sign(axis_up[2])*axis_up)
            if np.degrees(np.arccos(angle))>10:
                continue


        good = compute_votes(
            edgelets, current_model, threshold_inlier)

        current_votes = (good*strengths).sum()

        if current_votes> best_votes:
            best_models = current_model
            best_votes = current_votes
            model_inliers = good
            # print(np.degrees(np.arccos(angle)), up_normal,-np.sign(-axis_up[2])*axis_up)
        continue

    inlier_indices = np.nonzero(model_inliers)[0]

    if line_indices is not None:
        inlier_indices = line_indices[inlier_indices]

    return best_models, best_votes, inlier_indices


def get_XYZ_from_depth(depth, width, height, camera, max_depth=10):
    height = depth.shape[0]
    width = depth.shape[1]

    urange = (np.arange(width, dtype=np.float32) / (width) * (camera[4]) - camera[2]) / camera[0]
    urange = urange.reshape(1, -1).repeat(height, 0)

    vrange = (np.arange(height, dtype=np.float32) / (height) * (camera[5]) - camera[3]) / camera[1]
    vrange = vrange.reshape(-1, 1).repeat(width, 1)

    X = depth * urange
    Y = depth
    Z = -depth * vrange
    XYZ = np.dstack((X, Y, Z))

    return XYZ

def xyz_to_uv(pt, width, height, camera, f=1.0):
    pt = pt / pt[1]

    u = (pt[0] * camera[0]*f + camera[2]) * width / camera[4]
    v = (-pt[2] * camera[1]*f + camera[3]) * height / camera[5]

    return np.array([u, v])


def uv_to_xyz(uv, depth, width, height, camera, f=1.0):
    p0 = (uv[0] * camera[4] / width - camera[2]) / (f*camera[0]) * depth
    p2 = - (uv[1] * camera[5] / height - camera[3]) / (f*camera[1]) * depth

    return np.array([p0, depth, p2])

def proj(points, plane):
    plane_offset = np.linalg.norm(plane)
    plane_normal = plane / plane_offset
    s = np.dot(points, plane_normal) - plane_offset
    return s

# Create fan from vertical vp
def fan_surfaces(data, img_lr, locations, vp0, sure_walls, wall_mask, normals_c):

    labels_fan = np.int32(np.zeros((img_lr.shape[0], img_lr.shape[1])))

    fan_normals = []

    k = 1
    for i in range(-1, len(locations)):
        if i == -1:
            closest = np.int32([[0, img_lr.shape[1]], [0, 0]])
            dir = closest - vp0[:2]
            closest_index = np.argmin(np.abs(dir[:, 1]))
            pt1 = np.int32(2 * closest[closest_index] - vp0[:2])
            pt2 = np.int32(2 * locations[i + 1] - vp0[:2])
        elif i == len(locations) - 1:
            closest = np.int32([[img_lr.shape[1], img_lr.shape[0]], [img_lr.shape[1], 0]])
            dir = closest - vp0[:2]
            closest_index = np.argmin(np.abs(dir[:, 1]))
            pt2 = np.int32(2 * closest[closest_index] - vp0[:2])
            pt1 = np.int32(2 * locations[i] - vp0[:2])
        else:
            pt1 = np.int32(2 * locations[i] - vp0[:2])
            pt2 = np.int32(2 * locations[i + 1] - vp0[:2])

        triangle = np.array([[[vp0[0], vp0[1]], pt1, pt2]], np.int32)

        arc_mask = cv2.fillPoly(np.zeros_like(labels_fan), pts=triangle, color=1)

        wall_arc = np.logical_and(sure_walls > 0, arc_mask > 0)
        wedge = np.logical_and(wall_arc, wall_mask > .9)
        a = np.sum(wedge)

        if a > 0:
            labels_fan[wall_arc > 0] = k

            cur_normal = np.mean(normals_c[wedge], 0)
            cur_normal /= max(np.linalg.norm(cur_normal), .00001)
            fan_normals.append(cur_normal)
            k += 1
        # else:
        #     if len(fan_normals)>0:
        #         fan_normals.append(fan_normals[-1])

    # Merge by normal angle diff

    labels_fan, fan_normals_reduced, normals_wall = merge_by_angle_sweep(labels_fan, normals_c, fan_normals,
                                                                         wall_mask > .9, angle_threshold=.85)

    log_segmentation_image(data, "fan1", labels_fan, img_lr)

    unique_labels = np.unique(labels_fan[labels_fan > 0])

    fan_normals_reduced = np.float32([np.mean(normals_c[labels_fan == j], 0) for j in unique_labels])
    lengths = np.sqrt(np.sum(fan_normals_reduced * fan_normals_reduced, -1))

    fan_normals_reduced /= np.dstack((lengths, lengths, lengths))[0]

    labels_fan, fan_normals_reduced, normals_wall = merge_by_angle_sweep(labels_fan, normals_wall,
                                                                         fan_normals_reduced, wall_mask > 0,
                                                                         angle_threshold=.8)
    log_segmentation_image(data, "fan2", labels_fan, img_lr)

    return labels_fan, fan_normals_reduced, normals_wall

def merge_by_angle_sweep(labels_fan, normals_img, fan_normals, mask, angle_threshold):

    k=1
    normals_wall = normals_img.copy()
    fan_normals_reduced = []

    for i in range(len(fan_normals) - 1):

        cur_normal = fan_normals[i]
        next_normal = fan_normals[i + 1]
        ang1 = abs(np.dot(cur_normal, next_normal))
        labels_fan[labels_fan == i + 1] = k

        if ang1 > angle_threshold:
            # print('angle threshold met', ang_threshold, ang1)

            labels_fan[labels_fan == i + 2] = k
            wedge = np.logical_and(labels_fan == k, mask)
            norm = np.mean(normals_img[wedge], 0)

            if ~np.isnan(norm[0]):
                norm /= max(np.linalg.norm(norm), .00001)

                fan_normals[i] = norm
                fan_normals[i + 1] = norm

                normals_wall[labels_fan == k] = norm

                if i==len(fan_normals) - 2:
                    fan_normals_reduced.append(norm)
                    k += 1
            else:
                normals_wall[labels_fan == k] = cur_normal

                if i == len(fan_normals) - 2:
                    fan_normals_reduced.append(norm)
                    k += 1
        else:
            # print('angle threshold not met', angle_threshold, ang1)
            wedge = np.logical_and(labels_fan == k, mask)
            norm = np.mean(normals_img[wedge], 0)

            if ~np.isnan(norm[0]):
                norm /= max(np.linalg.norm(norm), .00001)
                fan_normals_reduced.append(norm)
            else:
                fan_normals_reduced.append(norm)
            k += 1

    fan_normals_reduced = np.float32(fan_normals_reduced)

    return labels_fan, fan_normals_reduced, normals_wall


class PipelinePoseEstimator(PipelineStep):

    @property
    def index(self) -> PipelineStepIndex:
        return PipelineStepIndex.EstimatePose

    @property
    def required_keys(self) -> list:
        return ["image", "lines", "fov", "floor_normal", "floor_offset"]

    @property
    def output_keys(self) -> list:
        return ["fov", "floor_rotation", "floor_normal", "floor_offset", "edgelets", "vp0", "camera"]

    def run(self, data):

        pose_estimator = PoseEstimator(data, data["image"], data["lines"], data["fov"], data["isolated_probs"][SurfaceType.Floor], data["floor_normal"], data["floor_offset"])
        pose_estimator.estimate()

        data["fov"] = pose_estimator.fov
        data["floor_rotation"] = pose_estimator.floor_rotation
        data["floor_normal"] = pose_estimator.floor_normal
        data["floor_offset"] = pose_estimator.floor_offset
        data["edgelets"] = pose_estimator.edgelets
        data["vp0"] = pose_estimator.vp0
        data["camera"] = pose_estimator.camera

