import cv2
import os
from tqdm import tqdm
def images_to_video(input_folder, output_video, fps):
    images = [img for img in os.listdir(input_folder) if img.endswith((".png", ".jpg", ".jpeg"))]
    images.sort()  

    first_image = cv2.imread(os.path.join(input_folder, images[0]))
    height, width, layers = first_image.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    for image in tqdm(images):
        img_path = os.path.join(input_folder, image)
        frame = cv2.imread(img_path)
        video.write(frame)

    video.release()
    print(f"视频已保存到: {output_video}")

for input_folder in os.listdir('deblur_nerf_data'):
    output_video = 'result.mp4' 
    fps = 30 
    images_to_video(os.path.join('deblur_nerf_data',input_folder,'raw_data'), os.path.join('deblur_nerf_data',input_folder,output_video), fps)
