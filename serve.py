import os
import time
import cv2
import copy
import asyncio
import math
import json
import io
import shutil

import tensorflow as tf
import numpy as np
from scipy.misc import imresize
from skimage.filters import threshold_sauvola
from scipy import ndimage
from skimage.morphology import reconstruction
import joblib
from aiohttp import web
from PIL import Image

from cambrian import utils, image_processing as ip, geometry as geo, transformations as T, diagnostics as d

# Z is UP
rotX = T.rotation_matrix(0.00, [1, 0, 0])
rotY = T.rotation_matrix(0.00, [0, 1, 0])
rotZ = T.rotation_matrix(-0.06, [0, 0, 1])

r_range = 255
g_range = 220
b_range = 255

r_range = g_range = b_range = 255

RIGHT_ANGLE = math.pi / 2.0 # 90 degrees

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

def feed_images(model, images):
    # Send in all inputs
    inputs = {}
    
    for key, feed_tensor in model.feed_tensors.items():
        img = images[key]

        shape = feed_tensor.shape

        # Resize to target size. This needs to be
        # done before potentially expanding the
        # channels dimension as it removes it again.
        img = cv2.resize(img, tuple(shape[1:3]))

        # Make sure we have the channels dimension
        # for 1-channel images.
        if len(img.shape) == 2:
            img = np.expand_dims(img, -1)

        assert img.shape == shape[1:], "Shape not equal to target shape, shape: %s target: %s" % (img.shape, shape[1:])

        input_image = img.astype(np.float32) / 255.0
        inputs[key] = [input_image]

    inference = model(inputs)

    # Process all outputs
    outputs = []
    for key in model.fetch_tensors.keys():
        output = inference[key][0] * 255.0
        
        # Remove single channel dimension if any
        if len(output.shape) == 3 and output.shape[-1] == 1:
            output = np.squeeze(output, axis=-1)

        outputs.append(output)
        
    return outputs

def process_image(args, image_s3_key, s3_client, models):
    print(image_s3_key)

    input_image = get_s3_object(s3_client, "cb-imageprocessinguseruploads", image_s3_key)

    result_path = os.path.join(args.output_path, "%s_semantic.png" % image_s3_key)

    if not os.path.isfile(result_path):
        start_processing = time.time()

        if len(input_image.shape) == 2 or input_image.shape[2] < 3:
            return None

        start_deep = time.time()
        input_image = input_image[:, :, :3]

        for model_name, model_data in models.items():
            if model_name == "semantic":
                continue
            start_model = time.time()

            # Get the input name (currently different for the networks, should
            # be cambrian.nn.get_input_name(0) later)
            input_name = list(model_data["model"].feed_tensors.keys())[0]
            model_data["results"] = feed_images(model_data["model"], {input_name: input_image})
            print("Model %s took %.2f seconds" % (model_name, time.time() - start_model))
            d.save_diagnostics_image(args, model_data["results"][0], image_s3_key, model_name + "_latents", verbose=True)

        #run semantic:
        semantic_model = models['semantic']
        unlit = models["unlit"]["results"][0]
        normals = models["normals"]["results"][0][:, :, :3] # Normals has 4 channels for some reason

        # Hardcoded DFN input names, should change to cambrian.nn.get_input_name later
        semantic_model["results"] = feed_images(semantic_model["model"], {
            "image": input_image,
            "unlit": unlit
        })

        #probability
        a_channel, b_channel = cv2.split(semantic_model["results"][0])
        c_channel = np.zeros(a_channel.shape, dtype=a_channel.dtype)
        semantic_model["results"][0] = cv2.merge((a_channel, b_channel, c_channel))

        semantic_model["result_prob"] = np.exp(semantic_model["results"][0] / 255.) / np.sum(np.exp(semantic_model["results"][0] / 255.), axis=-1, keepdims=True) * 255.

        print(semantic_model["result_prob"].shape)
        d.save_diagnostics_image(args, semantic_model["result_prob"], image_s3_key, "semantic")

        # Choose the most likely class (argmax)
        semantic_model["mask"] = (np.argmax(semantic_model["result_prob"], -1) == 0).astype(np.float32)

        if args.debug:
            mask = imresize(semantic_model["mask"], input_image.shape[:2])
            d.save_diagnostics_mask(args, mask, image_s3_key, "overlay-initial", input_image, d.HUE_GREEN)
        
        #not ready to process yet:
        print("Deep learning took %.2f seconds" % (time.time() - start_deep))

        start_determine_primary_angles = time.time()
        data = determine_primary_angles(args, models, image_s3_key, input_image)
        print("determine_primary_angles took %.2f seconds" % (time.time() - start_determine_primary_angles))

        start_refinement = time.time()
        refine_results(args, models, image_s3_key, input_image)
        
        print("Refinement took %.2f seconds" % (time.time() - start_refinement))

        save_result(args, models, image_s3_key, input_image, data)

        if not s3_client is None:
            upload_result(args, s3_client, image_s3_key)

        print("Processing took %.2f seconds" % (time.time() - start_processing))

def get_matching_surface(reduced_mask, isolated_surfaces, isolated_values, ignore_indices=[], max_angle=40.0):
    isolated_mask = np.zeros(reduced_mask.shape,dtype=np.uint8)

    for value in isolated_values:
        isolated_mask[reduced_mask == value] = 255

    most_pixels = 0
    surface_index = -1
    best_intersection = 0
    best_angle = 0

    straight_up = [0, 0, 1]

    max_radians = geo.degrees_to_radians(max_angle)

    #find best matching surface
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

    print("Floor is %.2f degrees from UP" % geo.radians_to_degrees(best_angle))
    return surface_index, most_pixels, best_intersection

def get_candidate_walls(floor_normal, isolated_surfaces, maxAngle=20):
    candidate_walls = []
    max_diff = geo.degrees_to_radians(maxAngle)

    kernel = np.ones((3,3),np.uint8)
    best_score = 0
    best_wall_index = 0
    wall_count = 0

    overall_best_index = 0
    overall_best_diff = math.pi

    for i, surface in enumerate(isolated_surfaces):
        normal = surface[1]
        angle = geo.angle_between(normal, floor_normal)
        vert_diff = abs(angle - RIGHT_ANGLE)
        
        if vert_diff < overall_best_diff:
            overall_best_diff = vert_diff
            overall_best_index = i

        if vert_diff < max_diff:
            candidate_walls.append(surface)
            angle_diff = 180.0 * vert_diff / math.pi

            opening = cv2.morphologyEx(surface[2], cv2.MORPH_OPEN, kernel)
            total_pixels = cv2.countNonZero(opening)

            score = total_pixels / math.sqrt(angle_diff + 3.0)

            if score > best_score:
                best_wall_index = wall_count
                best_score = total_pixels

            print("%d) candidate wall normal = %s, angle diff = %.2f, color = %s, score=%f" \
                % (wall_count, normal, angle_diff, surface[0], score))

            wall_count = wall_count + 1

    if len(candidate_walls) == 0:
        best_wall_index = 0
        candidate_walls.append(isolated_surfaces[best_wall_index])

    return candidate_walls, best_wall_index

def predict_fov(input_image, models):
    # Get the latents of the normals network by running it again.
    # Should be combined with the other normals run later.
    predictor = models["normals"]["model"]
    sess = predictor.session
    input_tensor = predictor.feed_tensors["input_0"]
    latent_tensors = predictor.graph.get_tensor_by_name("generator/decoder_8/conv2d_transpose/BiasAdd:0")
    latents = sess.run(latent_tensors, feed_dict={input_tensor: [cv2.resize(input_image, (512, 512)).astype(np.float32)/255]})

    # Pass latents to sklearn model and return the predicted fov.
    # Right now we are always loading the model again but it's
    # small so it's not a huge issue and it guarantees thread safety.
    latents = latents[:, :, :, :128].reshape(1, -1)
    classifier = joblib.load("sklearn_models/fov_classifier_lc128.joblib")
    fov_class = classifier.predict(latents)[0]

    return 60.0 if fov_class == 0 else 85.0

def determine_primary_angles(args, models, image_name, input_image, k=5):
    mask = models["semantic"]["mask"]
    normals = models["normals"]["results"][0][:, :, :3]
    semantic_model = models['semantic']
    elevation = models["elevation"]["results"][0]

    d.save_diagnostics_image(args, normals, image_name, "normals", verbose=True)

    kmeans, labels, centers = ip.kmeans_image(normals, k)
    d.save_diagnostics_image(args, kmeans, image_name, "normals_clustered", verbose=True)

    models["normals"]["kmeans"] = kmeans

    reduced_normals = kmeans
    reduced_mask = mask
    reduced_mask[reduced_mask > 0] = 255

    d.save_diagnostics_image(args, reduced_normals, image_name, "reduced_normals", verbose=True)
    d.save_diagnostics_image(args, reduced_mask, image_name, "reduced_mask", verbose=True)

    # build surfaces
    isolated_surfaces = []
    for color in centers:
        normal = get_normal_from_rgb(color)
        color_mask = ip.isolate_color(reduced_normals, color)
        # d.save_diagnostics_image(args, color_mask, image_name + str(color), "color_mask")
        isolated_surfaces.append((color, normal, color_mask))

    if args.debug:
        debug_image = reduced_normals.copy()

    floor_materials = [255] #floor, rug

    #wall windowpane, cabinet, wardrobe, painting, mirror, shelf, refrigerator, bookcase
    wall_materials = [0]

    # find floor
    floor_index, most_pixels, floor_intersection = get_matching_surface(reduced_mask, isolated_surfaces, floor_materials) 

    if floor_index < 0:
        print("Invalid surfaces")
        return None

    floor_surface = isolated_surfaces[floor_index]

    data = {}  

    # Calculate the camera pitch and roll from the floor normal.
    # Get the floor normal by taking the normals at the 100 pixels
    # most likely to be floor and average them.
    result_prob = semantic_model["result_prob"][:, :, 0]
    result_prob = cv2.bitwise_and(result_prob,result_prob, mask = floor_intersection)

    floor_indices = np.stack(np.unravel_index(np.argsort(-result_prob.flatten()), result_prob.shape), axis=-1)
    floor_indices = floor_indices[:100]

    strongest_floor = normals[floor_indices[:, 0], floor_indices[:, 1]]

    floor_normal = (np.mean(strongest_floor, axis=0) - 127.5) / 127.5
    floor_normal_len = max(0.00001, np.linalg.norm(floor_normal))
    floor_normal /= floor_normal_len
    cam_pitch = math.asin(floor_normal[1])
    cam_roll = math.asin(floor_normal[0])
    data["cameraRotation"] = [cam_pitch, 0.0, cam_roll]

    floor_distances = elevation[floor_indices[:, 0], floor_indices[:, 1]].flatten()
    floor_distances.sort()

    floor_elevation_pixels = 127.5 - float(floor_distances[np.floor_divide(len(floor_distances), 2)])
    pixels_per_meter = 127.5 / 300.0
    floor_elevation = floor_elevation_pixels / pixels_per_meter

    floor_elevation = np.clip(floor_elevation, 80.0, 170.0) #valid range

    data["cameraElevation"] = floor_elevation / 100.0

    if args.debug:
        debug_image = d.label_mask(debug_image, floor_intersection, "floor")

    print("Camera pitch: %.1f°, roll: %.1f°, elevation: %.2f cm" % (math.degrees(cam_pitch), math.degrees(cam_roll), floor_elevation))

    #find candidate wall surfaces
    candidate_walls, primary_wall_index = get_candidate_walls(floor_surface[1], isolated_surfaces)

    if args.debug:
        print("Wall %d is primary" % primary_wall_index)
        primary_wall_intersection = candidate_walls[primary_wall_index][2]
        if primary_wall_index >= 0:
            debug_image = d.label_mask(debug_image, primary_wall_intersection, "primary")

        d.save_diagnostics_image(args, debug_image, image_name, "surfaces")
        debug_image = d.plot_vectors(floor_surface[1], floor_surface[1], candidate_walls[primary_wall_index][1])
        d.save_diagnostics_image(args, debug_image, image_name, "vectors")
    
    #calculate floor rotation:    
    floor_rotation = 0.0

    x_unit_normal = [1,0,0]
    y_unit_normal = [0,1,0]
    z_unit_normal = [0,0,1]

    floor_angle_x = math.pi - geo.angle_between(floor_surface[1], y_unit_normal)

    if primary_wall_index >= 0:
        floor_rotation = geo.angle_between(candidate_walls[primary_wall_index][1], x_unit_normal)
        print("Primary wall normal %s" % (candidate_walls[primary_wall_index][1],))

    print("Floor normal %s, rotation=%.2f, elevation=%.2f" % (floor_surface[1], floor_rotation, floor_angle_x))

    # Predict horizontal FoV
    fov = predict_fov(input_image, models)
    print("Predicted horizontal fov %.2f°" % fov)

    #json data
    
    data['fov'] = fov
    data['floorRotation'] = floor_rotation

    models["normals"]["kmeans"] = kmeans
    
    return data

def refine_results(args, models, image_name, input_image):
    print("Prepare final images")
    shape=(1024,1024)
    
    img=input_image
    img = imresize(input_image, shape)
    mask = np.uint8(models["semantic"]["mask"])
    kmeans_normals = models["normals"]["kmeans"]
    kmeans_normals=imresize(kmeans_normals, shape)
    normals = models["normals"]["results"][0]
    kmeans = kmeans_normals.copy()
    shadows_rgb = np.uint8(models["shadows"]["results"][0])
    shadows = shadows_rgb[:, :, 1]

    d.save_diagnostics_image(args, shadows, image_name, "shadows-initial", verbose=True)
    
    shadows = ip.remove_grooves(shadows, mask)

    blurred_mask = cv2.GaussianBlur(mask, (31,31), 15)
    blurred_shadows = cv2.GaussianBlur(shadows, (21,21), 11)

    shadows = ip.alpha_blend(shadows, blurred_shadows, blurred_mask)

    d.save_diagnostics_image(args, shadows, image_name, "shadows-processed", verbose=True)

    models["shadows"]["results"][0] = shadows
    
    prob_mask_full=models["semantic"]["result_prob"]
    prob_mask_full=imresize(prob_mask_full, shape)

    prob_mask=np.uint8(prob_mask_full[:,:,0])
    mask= imresize(mask, shape)
    
    d.save_diagnostics_mask(args, mask, image_name, "pre otsu1", img, d.HUE_GREEN)

    thresh, mask=cv2.threshold(prob_mask,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    prob_mask=cv2.threshold(prob_mask,thresh,255,3)[1]
    d.save_diagnostics_mask(args, mask, image_name, "otsu", img, d.HUE_GREEN)
 

    thresh_s=threshold_sauvola(prob_mask, window_size=5, k=0.2)
    a=np.abs(thresh_s)
    a*=1/np.amax(a)
    prob_mask=a
    d.save_diagnostics_image(args,prob_mask, image_name, "sauvola", verbose=True)
    
    distance = ndimage.distance_transform_edt(1.0-prob_mask/np.max(prob_mask))

    seed = np.copy(distance)/np.amax(distance)*prob_mask
    seed[1:-1, 1:-1] = (np.copy(distance)/np.amax(distance)*prob_mask).min()
    mask = prob_mask
    
    dilated = reconstruction(seed, mask, method='dilation')
    mask = prob_mask-dilated
    d.save_diagnostics_image(args,mask, image_name, "reconstruction", verbose=True)

    #watershed some
    mask[mask > 0] = 1;
    mask = ip.refine_mask_watershed(args, cv2.bilateralFilter(img, 21,75,75), mask, image_name, distance=0.02, max_value=1)
    
    mask=(255.*mask).astype('uint8')
    mask=cv2.GaussianBlur(mask,(15,15), 0)
    t, mask=cv2.threshold(mask,0,255,cv2.THRESH_TOZERO+cv2.THRESH_OTSU)

    mask=cv2.threshold(mask,t,255,cv2.THRESH_BINARY)[1]
    d.save_diagnostics_image(args, mask, image_name, "mask otsu2", verbose=True)

    kmeans_gray = cv2.cvtColor(kmeans, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(kmeans_gray, 127, 255, 0)
    nmask = np.zeros(mask.shape, np.uint8)

    # Get biggest normals contour
    _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        areas = [cv2.contourArea(c) for c in contours]
        max_index = np.argmax(areas)
        cnt = contours[max_index]
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            c_x = int(M["m10"] / M["m00"])
            c_y = int(M["m01"] / M["m00"])
            if kmeans[c_y, c_x][2] > 127:
                nmask = ip.isolate_color(kmeans, kmeans[c_y, c_x])

    border = 50
    border_mask=cv2.copyMakeBorder(nmask, border, border,
                                   border, border, cv2.BORDER_REPLICATE)
    cv2.rectangle(border_mask, (51, 51), (border_mask.shape[1] - border - 1, border_mask.shape[0] - border - 1), 0, cv2.FILLED)

    #kernel = np.ones((5, 5), np.uint8)
    #mask = cv2.erode(mask, kernel, iterations=1)

    mask = cv2.copyMakeBorder(mask, border, border, border, border, cv2.BORDER_CONSTANT, value=0)
    mask = mask + border_mask
    img=cv2.copyMakeBorder(img, border, border, border, border, cv2.BORDER_CONSTANT, value=0)

    d.save_diagnostics_image(args,mask, image_name, "mask after contours0", verbose=True)

    # Fill border holes
    _, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        areas = [cv2.contourArea(c) for c in contours]
        max_index = np.argmax(areas)
        cnt = contours[max_index]
        # compute the center of the contour
        min_area = (shape[0] * shape[1]) / 200

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                # compute the center of the contour
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    color = 0 if mask[cY,cX] > 0 else 255
                    mask = cv2.fillPoly(mask, pts=[contour], color=color)

    d.save_diagnostics_image(args,mask, image_name, "mask after contours1", verbose=True)
    mask = ip.crop_image(mask, border)
    img = ip.crop_image(img, border)

    # Fill small holes
    _, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                # compute the center of the contour
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    color = 0 if mask[cY,cX] > 0 else 255
                    mask = cv2.fillPoly(mask, pts=[contour], color=color)

    mask = cv2.GaussianBlur(mask,(5,5),0)

    d.save_diagnostics_mask(args, mask, image_name, "overlay", img, d.HUE_GREEN)

    mask = cv2.resize(mask, (1024, 1024))
    models["semantic"]["mask"] = mask

    return {
        "lighting_url": "",
        "semantic_url": "",
    }

def upload_results(s3_client, s3_id, results):
    raise NotImplementedError()

def get_image_from_s3(s3_client, key):
    response = s3_client.Object("cb-imageprocessinguseruploads", key).get()
    image_data =  response["Body"].read()
    return Image.open(io.BytesIO(image_data))

def main():
    # Setup arguments
    parser = utils.std_args()
    parser.add_argument('-m', '--model_path', type=str, default="exported_models", help='Model path to load')
    parser.add_argument('-e', '--exp', type=str, default="*.*", help='Expression to match for input')
    parser.add_argument("--upload_to_s3", type=utils.str2bool, nargs='?', const=True, default=False, help="Upload to s3.")
    
    args = parser.parse_args()

    if args.verbose:
        args.debug = True
    
    print(args)

    s3_client = utils.get_s3_client()

    # Load models
    model_names = ["semantic", "normals", "elevation", "shadows", "unlit"]
    models = {}

    for model_name in model_names:
        model_path = os.path.join(args.model_path, model_name)
        print("Loading model " + model_name)
        model = tf.contrib.predictor.from_saved_model(model_path)
        models[model_name] = {"model": model}
        print("Loaded", model_name, "model with inputs: [%s] and outputs: [%s]" % (','.join(model.feed_tensors.keys()),  ','.join(model.fetch_tensors.keys())))

    # Setup http server
    async def handle_segment(request):
        loop = asyncio.get_event_loop()

        image_s3_key = request.match_info.get("id", None)

        if image_s3_key is None:
            raise web.HTTPBadRequest()

        def process_request():
            process_image(args, image_s3_key, s3_client, models)

        result = await loop.run_in_executor(None, process_request)

        return web.Response()

    app = web.Application()
    app.add_routes(([
        web.get("/segment/{id}", handle_segment)
    ]))

    print("Running web app")
    web.run_app(app)

if __name__ == "__main__":
    main()