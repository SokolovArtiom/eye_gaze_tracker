import argparse
import os

import cv2
import face_alignment
import numpy as np
from src.EyeGaze import GazeModel
from tqdm import tqdm


def parse_args(args):
    parser = argparse.ArgumentParser(description="Choose config: ")
    parser.add_argument("--eye_gaze_backbone", type=str, default="mobile_net")
    parser.add_argument(
        "--eye_gaze_model_weights",
        type=str,
        default="weights/mobile_net_small100_00067.pth",
    )
    parser.add_argument("--input_path", type=str, default="test/images/")
    parser.add_argument("--output_path", type=str, default="test/results/")
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args(args)


def find_gaze(img, gaze_model, landmark, eye_points):
    x_left = int(landmark[eye_points[0]][0])
    x_right = int(landmark[eye_points[3]][0])
    y_top = int(landmark[eye_points[1]][1])
    y_bot = int(landmark[eye_points[5]][1])

    x_pad = (x_right - x_left) // 2
    y_pad = (y_top - y_bot) // 2

    x_left -= x_pad
    x_right += x_pad
    y_top += y_pad
    y_bot -= y_pad

    img = img[y_top:y_bot, x_left:x_right]

    if img.size > 0:
        pred = gaze_model.predict(img)
        return pred
    else:
        return [None, None]


def main(args=None):
    print("Loading models...")
    args = parse_args(args)
    for k in vars(args):
        print(f"{k} : {getattr(args, k)}")

    os.mkdir(args.output_path)

    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D,
        flip_input=False,
        device=args.device,
    )
    gaze_model = GazeModel(
        args.eye_gaze_backbone, args.eye_gaze_model_weights, args.device
    )

    print("Processing data...")

    for imname in tqdm(os.listdir(args.input_path)):
        img = cv2.imread(f"{args.input_path}/{imname}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (256, 256))

        landmark = fa.get_landmarks(img)[0]

        left_eye_points = [36, 37, 38, 39, 40, 41]
        right_eye_points = [42, 43, 44, 45, 46, 47]

        left_gaze = find_gaze(img, gaze_model, landmark, left_eye_points)
        right_gaze = find_gaze(img, gaze_model, landmark, right_eye_points)

        gaze = (left_gaze + right_gaze) / 2

        left_eye_center = np.mean(landmark[left_eye_points], axis=0)
        right_eye_center = np.mean(landmark[right_eye_points], axis=0)

        cv2.arrowedLine(
            img,
            left_eye_center.astype(np.int32),
            (left_eye_center + gaze * 50).astype(np.int32),
            [0, 0, 255],
            2,
        )

        cv2.arrowedLine(
            img,
            right_eye_center.astype(np.int32),
            (right_eye_center + gaze * 50).astype(np.int32),
            [0, 0, 255],
            2,
        )

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(f"{args.output_path}/{imname}", img)
    print(f"Result saved to {args.output_path}")


if __name__ == "__main__":
    main()
