import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy
from torchvision.io import read_image, write_png
import configuration as cfg


def load_labels(label_path):
    """
    Load the labels from the text file label_path.
    Each line represents 1 object in the image
    :args: label_path: str, path to the label file associated with the image
    return: list of lines after cleaned each information
    """
    lines = []
    with open(label_path, "r") as f:
        for line in f.readlines():
            line = line.replace("\n", "")
            line = [float(elem) for elem in line.split(" ")]
            lines.append(line)
    return lines


def generate_mask(mode="train"):
    """
    Function used to generate masks data from the polygons and save them.
    :args: mode: str, mode of the generation used for the path
    """
    mask_folder = os.path.join(cfg.BASE_PATH, mode, "masks")
    if not os.path.exists(mask_folder):
        os.makedirs(mask_folder)

    images_path = os.path.join(cfg.BASE_PATH, mode, "images")
    labels_path = os.path.join(cfg.BASE_PATH, mode, "labels")

    for file in os.listdir(images_path):
        base_name = ".".join(file.split(".")[:-1])

        output_mask_path = os.path.join(mask_folder, f"{base_name}.png")

        if os.path.exists(output_mask_path):
            continue

        image = read_image(os.path.join(images_path, file))
        height, width = image.shape[1:]

        mask = np.zeros((height, width), dtype=np.uint8)

        label_file = os.path.join(labels_path, f"{base_name}.txt")

        polygons = load_labels(label_file)

        for idx, poly in enumerate(polygons):
            keypoints = poly[1:]  # first is class

            pts = [(int(keypoints[i] * width), int(keypoints[i + 1] * height)) for i in range(0, len(keypoints), 2)]
            if len(pts) >= 3:
                cv2.fillPoly(mask, [np.array(pts, dtype=np.int32)], 255)  # binary mask, 0 is background, 255 is an object
            else:
                print(f"Warning. Got less then 3 points for {base_name}") # never happened
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)
        write_png(mask_tensor, output_mask_path)
    print("Fin ", mode)


def show_image_with_mask(image_path, mask_path, alpha=0.5):
    image = read_image(image_path)
    mask = read_image(mask_path)

    plt.imshow(image.permute(1, 2, 0))
    plt.imshow(mask.permute(1, 2, 0), alpha=alpha)
    plt.title("Image avec masque")
    plt.axis("off")
    plt.show()


def get_mean_pts_per_poly(labels):
    """
    Get the mean number of points within a polygone for every object in the training set.
    return: int, mean number
    """
    sum_pts = 0
    cpt_poly = 0
    for file in os.listdir(labels):
        file_path = os.path.join(labels, file)
        with open(file_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                nb_points = len(parts[1:]) // 2  # list of points is x1 y1 x2 y2 ... xN yN, thats mean that there is #parts[1:]/2 points
                sum_pts += nb_points
                cpt_poly += 1

    mean_pts_per_polygone = int(sum_pts / cpt_poly)
    return mean_pts_per_polygone


def normalize_rooftop(mode="train", num_points=7):
    """
    Function used to normalize the rooftops data from the polygons using the mean number of points and save them.
    :args: mode: str, used to automatize the process accros the dataset
    :args: num_points: int, number of points for each polygone
    """
    new_labels_path = os.path.join(cfg.BASE_PATH, mode, f"labels_resampled_{num_points}")
    if not os.path.exists(new_labels_path):
        os.makedirs(new_labels_path)

    labels_path = os.path.join(cfg.BASE_PATH, mode, "labels")
    if not os.path.exists(labels_path):
        print("Error, labels path does not exist")
        exit(1)

    for file in os.listdir(labels_path):
        file_path = os.path.join(labels_path, file)

        output_file_path = os.path.join(new_labels_path, file)
        if os.path.exists(output_file_path):
            continue

        with open(file_path, "r") as input_file:
            with open(output_file_path, "w") as output_file:
                for line in input_file:
                    parts = line.strip().split()
                    class_idx = parts[0]
                    points = list(map(float, parts[1:]))  # list of x1, y1, x2, y2, ..., xN, yN
                    points_array = np.array(points).reshape(-1, 2)  # array of [[x1 y1], [x2 y2], ..., [xN, yN]]

                    if len(points_array) < 2:
                        continue  # trop peu de points pour interpoler

                    distances = np.cumsum(np.linalg.norm(np.diff(points_array, axis=0), axis=1))
                    distances = np.insert(distances, 0, 0)

                    x_interp = scipy.interpolate.interp1d(distances, points_array[:, 0], kind='linear')
                    y_interp = scipy.interpolate.interp1d(distances, points_array[:, 1], kind='linear')
                    new_distances = np.linspace(0, distances[-1], num_points)
                    new_points = np.column_stack([x_interp(new_distances), y_interp(new_distances)])

                    # Format final
                    normalized_line = [class_idx] + list(new_points.flatten())
                    output_file.write(' '.join(map(str, normalized_line)) + '\n')
    print("Fin ", mode)


if __name__ == "__main__":
    exec = "generate_polygons"

    if exec == "generate_mask":
        modes = ["train", "valid", "test"]
        for mode in modes:
            generate_mask(mode)

    elif exec == "generate_polygons":
        mean_pts = get_mean_pts_per_poly(labels=os.path.join(cfg.BASE_PATH, "train", "labels"))
        modes = ["train", "valid", "test"]
        for mode in modes:
            normalize_rooftop(mode, num_points=mean_pts)

    else:
        train_data_path = os.path.join(cfg.BASE_PATH, "train")
        images_train = os.path.join(train_data_path, "images")
        labels_train = os.path.join(train_data_path, "labels")

        idx = np.random.randint(0, len(os.listdir(images_train)))

        sample_train = os.listdir(images_train)[idx][:-4]

        image_train_sample = os.path.join(cfg.BASE_PATH, "train/images", sample_train + ".jpg")
        mask_train_sample = os.path.join(cfg.BASE_PATH, "train/masks", sample_train + ".png")
        show_image_with_mask(image_train_sample, mask_train_sample, alpha=0.4)

