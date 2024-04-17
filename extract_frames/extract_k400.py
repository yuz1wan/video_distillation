import os
import subprocess
from multiprocessing import Pool
from tqdm import tqdm

# path to the Kinetics-400 dataset
input_dir = "./"
output_dir = "./K400_64x64_16frame"
split_name_list = ["replacement/replacement_for_corrupted_k400", "train", "val", "test"]
num_workers = 2

# function to extract frames from video at a given fps


def extract_frames(video_path, output_path, fps=2, num_frames=None, frame_size=(64, 64)):
    # Get video duration
    try:

        cmd_duration = f"ffprobe -i {video_path} -show_entries format=duration -v quiet -of csv='p=0'"
        output = subprocess.check_output(cmd_duration, shell=True)
        duration = float(output)

        factor = num_frames // 8
        if duration<1.0:
            fps=12*factor
        elif duration<2.0:
            fps=8*factor
        elif duration<4.0:
            fps=4*factor

        # Calculate time point for middle frames
        middle_time = duration / 2
        start_time = max(0, middle_time - (num_frames / (2 * fps)))
        end_time = min(duration, middle_time + (num_frames / (2 * fps)))

        # Extract frames at middle time point
        cmd = f"ffmpeg -ss {start_time} -i {video_path} -t {end_time-start_time} -vf fps={fps},scale={frame_size[0]}:{frame_size[1]} -vframes {num_frames} {output_path} -v quiet"
        subprocess.call(cmd, shell=True)

        extract_frame_num = len(os.listdir(os.path.dirname(output_path)))
        if extract_frame_num!=num_frames:
            with open(f"{output_dir}/short_videos.txt", "a") as f:
                f.write(video_path + f",{extract_frame_num}\n")
                print("short", video_path, extract_frame_num)

    except subprocess.CalledProcessError as e:
        # If an error occurs, record the video name and skip
        with open(f"{output_dir}/broken_videos.txt", "a") as f:
            f.write(video_path + "\n")
        print("broken", video_path)



# function to extract frames from a single video and update progress bar
def extract_frames_progress(video_path):
    in_pth  = os.path.join(input_dir, video_path)
    out_dir = os.path.join(output_dir, video_path[:-4])
    os.makedirs(out_dir, exist_ok=True)
    out_pth = os.path.join(out_dir, "%02d.jpg")
    
    extract_frames(in_pth, out_pth, fps=2, num_frames=16)



for split_name in split_name_list:
    # loop through all videos in the Kinetics-400 dataset
    videos_to_extract = []
    root_path = os.path.join(input_dir, split_name)
    for video_name in os.listdir(root_path):
        if os.path.exists(os.path.join(output_dir, split_name, video_name[:-4])):
            continue
        #else:
        #    print(os.path.join(output_dir, split_name, video_name[:-4]))
        if not video_name.endswith(".mp4") or video_name[0]=='.':
            print("skip", video_name)
            continue
        video_path = os.path.join(split_name, video_name)
        videos_to_extract.append(video_path)
    print(len(videos_to_extract), "videos")
    
    # extract frames from videos in parallel using all available CPU cores and show progress bar
    with Pool(num_workers) as p:
        for _ in tqdm(
            p.imap_unordered(extract_frames_progress, videos_to_extract), total=len(videos_to_extract),
            postfix=split_name
        ):
            pass
