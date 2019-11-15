from pipeline.core import PipelineStep
import cv2
from cambrian import image_processing as ip, transformations as T, geometry as geo
import skimage.io
import os
import math
import random
from skimage import feature, color, transform, io
import numpy as np
import torch
import skimage.io
import numpy.linalg as LA
import matplotlib.pyplot as plt
from skimage import feature, color, transform, io
import neurvps
from neurvps.config import C, M

debug_vanishing_points = False

# Z is UP
rotX = T.rotation_matrix(0.00, [1, 0, 0])
rotY = T.rotation_matrix(0.00, [0, 1, 0])
rotZ = T.rotation_matrix(0.00, [0, 0, 1])

r_range = g_range = b_range = 255

RIGHT_ANGLE = (math.pi/2.0)  # 90 degrees

def sample_sphere(v, alpha, num_pts):
    v1 = orth(v)
    v2 = np.cross(v, v1)
    v, v1, v2 = v[:, None], v1[:, None], v2[:, None]
    indices = np.linspace(1, num_pts, num_pts)
    phi = np.arccos(1 + (math.cos(alpha) - 1) * indices / num_pts)
    theta = np.pi * (1 + 5 ** 0.5) * indices
    r = np.sin(phi)
    return (v * np.cos(phi) + r * (v1 * np.cos(theta) + v2 * np.sin(theta))).T


def orth(v):
    x, y, z = v
    o = np.array([0.0, -z, y] if abs(x) < abs(y) else [-z, 0.0, x])
    o /= LA.norm(o)
    return o


def crop(shape, scale=(0.35, 1.0), ratio=(9 / 16, 16 / 9)):
    for attempt in range(20):
        area = shape[0] * shape[1]
        target_area = random.uniform(*scale) * area
        aspect_ratio = random.uniform(*ratio)

        w = int(round(math.sqrt(target_area * aspect_ratio)))
        h = int(round(math.sqrt(target_area / aspect_ratio)))

        if random.random() < 0.5:
            w, h = h, w

        if h <= shape[0] and w <= shape[1]:
            j = random.randint(0, shape[0] - h)
            i = random.randint(0, shape[1] - w)
            return i, j, h, w

    # Fallback
    w = min(shape[0], shape[1])
    i = (shape[1] - w) // 2
    j = (shape[0] - w) // 2
    return i, j, w, w

def compute_edgelets(image, sigma=2):
    """Create edgelets as in the paper.
    Uses canny edge detection and then finds (small) lines using probabilstic
    hough transform as edgelets.
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
    gray_img = color.rgb2gray(image)
    edges = feature.canny(gray_img, sigma)
    lines = transform.probabilistic_hough_line(edges, line_length=int(image.shape[1]/512*16),
                                               line_gap=3)

    locations = []
    directions = []
    strengths = []

    for p0, p1 in lines:
        p0, p1 = np.array(p0), np.array(p1)
        locations.append((p0 + p1) / 2)
        directions.append(p1 - p0)
        strengths.append(np.linalg.norm(p1 - p0))

    # convert to numpy arrays and normalize
    locations = np.array(locations)
    directions = np.array(directions)
    strengths = np.array(strengths)

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
    locations, directions, _ = edgelets
    normals = np.zeros_like(directions)
    normals[:, 0] = directions[:, 1]
    normals[:, 1] = -directions[:, 0]
    p = -np.sum(locations * normals, axis=1)
    lines = np.concatenate((normals, p[:, np.newaxis]), axis=1)
    return lines


def compute_votes(edgelets, model, threshold_inlier=5):
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

    locations, directions, strengths = edgelets

    est_directions = locations - vp
    dot_prod = np.sum(est_directions * directions, axis=1)
    abs_prod = np.linalg.norm(directions, axis=1) * \
        np.linalg.norm(est_directions, axis=1)
    abs_prod[abs_prod == 0] = 1e-5

    cosine_theta = dot_prod / abs_prod
    theta = np.arccos(np.abs(cosine_theta))

    theta_thresh = threshold_inlier * np.pi / 180
    return (theta < theta_thresh) * strengths

def reestimate_model(model, edgelets, threshold_reestimate=5):
    """Reestimate vanishing point using inliers and least squares.
    All the edgelets which are within a threshold are used to reestimate model
    Parameters
    ----------
    model: ndarry of shape (3,)
        Vanishing point model in homogenous coordinates which is to be
        reestimated.
    edgelets: tuple of ndarrays
        (locations, directions, strengths) as computed by `compute_edgelets`.
        All edgelets from which inliers will be computed.
    threshold_inlier: float
        threshold to be used for finding inlier edgelets.
    Returns
    -------
    restimated_model: ndarry of shape (3,)
        Reestimated model for vanishing point in homogenous coordinates.
    """
    locations, directions, strengths = edgelets

    inliers = compute_votes(edgelets, model, threshold_reestimate) > 0
    # print("inliers ",  np.count_nonzero(inliers))
    if np.count_nonzero(inliers)<10:
        return model

    locations = locations[inliers]
    directions = directions[inliers]
    strengths = strengths[inliers]

    lines = edgelet_lines((locations, directions, strengths))

    a = lines[:, :2]
    b = -lines[:, 2]
    est_model = np.linalg.lstsq(a, b, rcond=None)[0]
    # print("returning ", np.concatenate((est_model, [1.])))
    return np.concatenate((est_model, [1.]))



def scale_component(color, range=255.0):
    return 2.0 * (float(color) / float(range)) - 1.0


def get_normal_from_rgb(rgb):
    x = scale_component(rgb[0], r_range)
    y = scale_component(rgb[1], g_range)
    z = scale_component(rgb[2], b_range)

    return rotate_normal((x, y, z))


def rotate_normal(normal):
    result = np.dot(normal, rotX[:3, :3].T)
    result = np.dot(result, rotY[:3, :3].T)
    result = np.dot(result, rotZ[:3, :3].T)
    return tuple(result)


def get_matching_surface(reduced_mask, isolated_surfaces, isolated_values, ignore_indices=[], max_angle=40.0):
    isolated_mask = np.zeros(reduced_mask.shape, dtype=np.uint8)

    for value in isolated_values:
        isolated_mask[reduced_mask == value] = 255

    most_pixels = 0
    surface_index = -1
    best_intersection = 0
    best_angle = 0

    straight_up = [0, 0, 1]

    max_radians = geo.degrees_to_radians(max_angle)

    # find best matching surface
    for i, surface in enumerate(isolated_surfaces):
        if i not in ignore_indices:
            normal = surface[1]
            angle = geo.angle_between(normal, straight_up)

            intersection = cv2.bitwise_and(surface[2], isolated_mask)

            intersection_pixels = cv2.countNonZero(intersection)

            if intersection_pixels > most_pixels and angle < max_radians:
                most_pixels = intersection_pixels
                surface_index = i
                best_intersection = intersection
                best_angle = angle

    if surface_index == -1 and max_angle != 180.0:
        return get_matching_surface(reduced_mask, isolated_surfaces, isolated_values, ignore_indices, max_angle=180.0)
    # print(best_angle)
    print("Floor is %.2f degrees from UP" % geo.radians_to_degrees(best_angle))
    return surface_index, most_pixels, best_intersection



class PipelineDeterminePrimaryAngles(PipelineStep):

    @property
    def required_keys(self) -> list:
        return ["semantic_probs", "normals", "elevation", "image"]

    @property
    def output_keys(self) -> list:
        return ["kmeans_normals", "camera_rotation", "camera_elevation", "floor_rotation"]

    def run(self, data):

        C.update(C.from_yaml(filename='pytorch_models/config.yaml'))
        C.model.im2col_step = 32  # override im2col_step for evaluation
        M.update(C.model)
        # pprint.pprint(C, indent=4)

        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)

        device_name = "cpu"
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        # if torch.cuda.is_available():
        #     device_name = "cuda"
        #     torch.backends.cudnn.deterministic = True
        #     torch.cuda.manual_seed(0)
        #     print("Let's use", torch.cuda.device_count(), "GPU(s)!")
        # else:
        #     print("CUDA is not available")
        device = torch.device(device_name)

        if M.backbone == "stacked_hourglass":
            # print("stacked_hourglass")
            model = neurvps.models.hg(
                planes=64, depth=M.depth, num_stacks=M.num_stacks, num_blocks=M.num_blocks
            )
        else:
            raise NotImplementedError

        checkpoint = torch.load('pytorch_models/better-result.pth.tar')
        model = neurvps.models.VanishingNet(
            model, C.model.output_stride, C.model.upsample_scale
        )
        model = model.to(device)
        model = torch.nn.DataParallel(
            model, device_ids=[0]
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        n = 3

        image = data["image"]

        # i, j, h, w = crop(image.shape)

        i, j, h, w = 0, 0, image.shape[0], image.shape[1]

        image_rs = np.uint8(255 * skimage.transform.resize(image[j: j + h, i: i + w], (512, 512)))

        image_roll = np.rollaxis(image_rs.astype(np.float), 2).copy()

        image_tensor = torch.tensor([image_roll]).float()

        image_tensor = image_tensor.to(device)
        input_dict = {"image": image_tensor, "test": True}

        vpts = sample_sphere(np.array([0, 0, 1]), np.pi / 2, 64)
        input_dict["vpts"] = vpts

        with torch.no_grad():
            score = model(input_dict)[:, -1].cpu().numpy()
        index = np.argsort(-score)
        candidate = [index[0]]
        for i in index[1:]:
            if len(candidate) == n:
                break
            dst = np.min(np.arccos(np.abs(vpts[candidate] @ vpts[i])))
            if dst < np.pi / n:
                continue
            candidate.append(i)
        vpts_pd = vpts[candidate]

        for res in range(1, len(M.multires)):
            vpts = [sample_sphere(vpts_pd[vp], M.multires[-res], 64) for vp in range(n)]
            input_dict["vpts"] = np.vstack(vpts)
            with torch.no_grad():
                score = model(input_dict)[:, -res - 1].cpu().numpy().reshape(n, -1)
            for i, s in enumerate(score):
                vpts_pd[i] = vpts[i][np.argmax(s)]
                vpts_pd[i] = vpts_pd[i] / vpts_pd[i][2]

        # print("debug angles")
        # with np.load('/home/derrick/Downloads/0096_0_label.npz') as npz:
        #     vpts_pd = npz["vpts"]
        # for i in range(3):
        #     vpts_pd[i] = vpts_pd[i]/vpts_pd[i][2]
        # C.io.focal_length = 2.1875

        K = np.eye(3)
        K[0, 0] = C.io.focal_length*w/2.
        K[1, 1] = C.io.focal_length*h/2.
        K[0, 2] = w/2
        K[1, 2] = h/2
        K_inv = np.linalg.inv(K)

        dots = np.zeros(3)

        x_dot = np.zeros(3)
        y_dot = np.zeros(3)
        z_dot = np.zeros(3)

        for i in range(3):
            # print(vpts_pd[i])
            c = np.random.rand(3)
            vpts_pd[i][0] = (vpts_pd[i][0] * C.io.focal_length + 1)*w/2
            vpts_pd[i][1] = (1 - vpts_pd[i][1] * C.io.focal_length) * h/2
            u = np.dot(K_inv, vpts_pd[i])
            b = u / np.linalg.norm(u)
            x_dot[i] = abs(np.dot(b,(1,0,0)))
            y_dot[i] = abs(np.dot(b,(0,1,0)))
            z_dot[i] = abs(np.dot(b,(0,0,1)))

        vpts = [vpts_pd[np.argmax(x_dot)], vpts_pd[np.argmax(y_dot)], vpts_pd[np.argmax(z_dot)]]

        edgelets1 = compute_edgelets(image)
        vpts[0] = reestimate_model(vpts[0], edgelets1, 2)
        vpts[1] = reestimate_model(vpts[1], edgelets1, 2)
        vpts[2] = reestimate_model(vpts[2], edgelets1, 2)

        if debug_vanishing_points:
            plt.imshow(image)
            colors = ['red', 'blue', 'green']

            for i in range(3):
                p1 = int(vpts[i][0])
                p2 = int(vpts[i][1])
                for xy in np.linspace(0, w, 20):
                    plt.plot([p1, xy, p1, xy, p1, 0, p1, w-1], [p2, 0, p2,h-1, p2, xy, p2, xy], color=colors[i], linewidth=.2)
                vp1 = vpts[i]
                vp2 = vpts[(i+1)%3]
                plt.scatter(vp1[0], vp1[1], color=colors[i])
                plt.plot([vp1[0], vp2[0]], [vp1[1], vp2[1]], color=colors[i])

            plt.savefig("vp_far.png", dpi=500)

            plt.ylim(int(1.1 * h), -int(.1 * h))
            plt.xlim(-int(.1 * w), int(1.1 * w))

            plt.savefig("vp_near.png", dpi=500)
            plt.close()

        u = np.dot(K_inv, vpts[0])
        l1 = np.linalg.norm(u)
        r_x = np.sign(u[0])*u/l1

        u = np.dot(K_inv, vpts[1])
        l2 = np.linalg.norm(u)
        r_y = np.sign(u[1])*u/l2

        u = np.dot(K_inv, vpts[2])
        l3 = np.linalg.norm(u)
        r_z = np.sign(u[2])*u/l3
        R = np.array([r_x,r_y,r_z])

        r_z = np.cross(r_x,r_z)

        cam_pitch = -np.arcsin(r_y[2])
        cam_roll = -np.arcsin(r_y[0])

        if np.isnan(cam_pitch):
            cam_pitch = -0.4

        if np.isnan(cam_roll):
            cam_roll = 0.0

        data["camera_rotation"] = [cam_pitch, 0.0, cam_roll]

        print("camera rotation: " + str(data["camera_rotation"]))

        floor_rotation = -np.arccos(np.dot(r_x,np.cross(r_y, [1, 0, 0])))

        if np.isnan(floor_rotation):
            floor_rotation = -0.4

        data["floor_rotation"] = floor_rotation

        mask = np.uint8(255*data["semantic_probs"][:, :, 0])
        normals = np.uint8(255*data["normals"])
        elevation = data["elevation"]

        kmeans, labels, centers = ip.kmeans_image(normals, 5)

        data["kmeans_normals"] = kmeans

        reduced_normals = kmeans
        reduced_mask = mask.copy()
        reduced_mask[reduced_mask > 127] = 255
        reduced_mask[reduced_mask < 255] = 0

        # build surfaces
        isolated_surfaces = []
        for color in centers:
            normal = get_normal_from_rgb(color)
            color_mask = ip.isolate_color(reduced_normals, color)
            isolated_surfaces.append((color, normal, color_mask))

        floor_materials = [255]  # floor, rug

        # find floor
        floor_index, _, floor_intersection = get_matching_surface(
            reduced_mask, isolated_surfaces, floor_materials)

        # print("floor index: "+str(floor_index))

        if floor_index < 0:
            print("Invalid surfaces")
            return None

        floor_surface = isolated_surfaces[floor_index]

        result_prob = cv2.bitwise_and(mask, mask, mask=floor_intersection)

        floor_color = floor_surface[0]

        low_thresh = np.clip(floor_color - 4.0, 0, 255)

        high_thresh = np.clip(floor_color + 4.0, 0, 255)

        normals_mask = cv2.inRange(normals, low_thresh, high_thresh)

        floor_normal = np.mean(normals[result_prob > 127], axis=0)

        normals_mask[result_prob < 128] = 0

        floor_elevation = np.mean(elevation[normals_mask > 0], axis=0)

        floor_elevation_pixels = 127.5 - floor_elevation
        
        pixels_per_meter = 127.5 / 300.0
        
        floor_elevation = floor_elevation_pixels / pixels_per_meter

        floor_elevation = np.clip(floor_elevation, 80.0, 170.0) / 100.0  # valid range

        if np.isnan(floor_elevation):
            floor_elevation = 1.3
        
        data["camera_elevation"] = floor_elevation

        print("floor elevation: " + str(data["camera_elevation"]))
