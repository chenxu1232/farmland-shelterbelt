import time
import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm  # Progress bar library

from segformer import SegFormer_Segmentation

if __name__ == "__main__":
    # -------------------------------------------------------------------------#
    #   To modify the colors corresponding to different classes,
    #   change self.colors in the generate function.
    # -------------------------------------------------------------------------#
    segformer = SegFormer_Segmentation(
        # ✅ Make sure this path points to an existing model weight file
        model_path="logs/ep100-loss0.052-val_loss0.164.pth",
        num_classes=2,
        phi="b5"
    )

    # ----------------------------------------------------------------------------------------------------------#
    #   The mode parameter specifies the inference mode:
    #   'predict'       Single-image prediction
    #   'video'         Video detection
    #   'fps'           FPS performance testing
    #   'dir_predict'   Batch prediction on images in a directory, with results saved to an output directory.
    #                   By default, images are read from the img folder and saved to the img_out folder.
    #   'export_onnx'   Export the model to ONNX format
    # ----------------------------------------------------------------------------------------------------------#
    # ✅ The default mode is set to dir_predict for batch processing of .npy files
    mode = "dir_predict"

    # -------------------------------------------------------------------------#
    #   count           Specifies whether to calculate the pixel count (area)
    #                   and proportion of each predicted class.
    #   name_classes    Names of the segmentation classes, consistent with
    #                   those defined in json_to_dataset, used for displaying
    #                   class names and pixel counts.
    # -------------------------------------------------------------------------#
    count = False
    name_classes = ["background", "fanghulin"]

    # ----------------------------------------------------------------------------------------------------------#
    #   video_path          Specifies the input video path. Set video_path=0 to use the webcam.
    #   video_save_path     Specifies the output video path. An empty string disables video saving.
    #   video_fps           Specifies the FPS of the saved video.
    # ----------------------------------------------------------------------------------------------------------#
    video_path = 0
    video_save_path = ""
    video_fps = 25.0

    # ----------------------------------------------------------------------------------------------------------#
    #   test_interval       Specifies the number of inference iterations used for FPS measurement.
    #   fps_image_path      Specifies the image used for FPS testing.
    # ----------------------------------------------------------------------------------------------------------#
    test_interval = 100
    fps_image_path = "img/street.jpg"

    # -------------------------------------------------------------------------#
    #   dir_origin_path     Directory containing the input images (.npy files).
    #   dir_save_path       Directory where prediction results will be saved.
    # -------------------------------------------------------------------------#
    dir_origin_path = "img/"
    dir_save_path = "img_out/"

    # -------------------------------------------------------------------------#
    #   simplify            Whether to simplify the exported ONNX model.
    #   onnx_save_path      Path for saving the exported ONNX model.
    # -------------------------------------------------------------------------#
    simplify = True
    onnx_save_path = "model_data/models.onnx"

    if mode == "predict":
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = segformer.detect_image(image, count=count, name_classes=name_classes)
                r_image.show()

    elif mode == "video":
        capture = cv2.VideoCapture(video_path)
        if video_save_path != "":
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

        ref, frame = capture.read()
        if not ref:
            raise ValueError(
                "Failed to read the camera or video. Please check whether the camera is properly installed or the video path is correct.")

        fps = 0.0
        while (True):
            t1 = time.time()
            # Read a video frame
            ref, frame = capture.read()
            if not ref:
                break
            # Convert image format from BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert the frame to a PIL Image
            frame = Image.fromarray(np.uint8(frame))
            # Perform inference
            frame = np.array(segformer.detect_image(frame))
            # Convert RGB back to BGR for OpenCV display
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            fps = (fps + (1. / (time.time() - t1))) / 2
            print("fps= %.2f" % (fps))
            frame = cv2.putText(frame, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("video", frame)
            c = cv2.waitKey(1) & 0xff
            if video_save_path != "":
                out.write(frame)

            if c == 27:
                capture.release()
                break
        print("Video Detection Done!")
        capture.release()
        if video_save_path != "":
            print("Save processed video to the path :" + video_save_path)
            out.release()
        cv2.destroyAllWindows()

    elif mode == "fps":
        img = Image.open(fps_image_path)
        tact_time = segformer.get_FPS(img, test_interval)
        print(str(tact_time) + ' seconds, ' + str(1 / tact_time) + 'FPS, @batch_size 1')

    elif mode == "dir_predict":
        # ==============================================================
        # ✅ Core modification: Add support for .npy files
        # ==============================================================
        if not os.path.exists(dir_save_path):
            os.makedirs(dir_save_path)

        img_names = os.listdir(dir_origin_path)
        print(f"Processing images in directory: {dir_origin_path}")

        for img_name in tqdm(img_names):
            image_path = os.path.join(dir_origin_path, img_name)

            # --- Process .npy files ---
            if img_name.lower().endswith('.npy'):
                try:
                    # 1. Load NumPy data
                    npy_data = np.load(image_path)

                    # 2. Pass the data directly to the model
                    #    (The previously modified segformer.py supports NumPy inputs)
                    r_image = segformer.detect_image(npy_data)

                    # 3. Save prediction results
                    # Change the file extension to .png for convenient visualization
                    save_name = os.path.splitext(img_name)[0] + ".png"
                    r_image.save(os.path.join(dir_save_path, save_name))

                except Exception as e:
                    print(f"Failed to process {img_name}: {e}")

            # --- Process standard image files ---
            elif img_name.lower().endswith(
                    ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                try:
                    image = Image.open(image_path)
                    r_image = segformer.detect_image(image)
                    r_image.save(os.path.join(dir_save_path, img_name))
                except Exception as e:
                    print(f"Failed to process {img_name}: {e}")

        print(f"✅ Processing completed! Results have been saved to: {dir_save_path}")

    elif mode == "export_onnx":
        segformer.convert_to_onnx(simplify, onnx_save_path)

    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video', 'fps' or 'dir_predict'.")
