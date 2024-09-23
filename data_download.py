from pytubefix import YouTube
import argparse
from glob import glob 
import os
from tqdm import tqdm

iteration = 0

def DownloadVideo(link: str, output_path: str, filename: str):
    global iteration
    try:
        yt = YouTube(link)
        stream = yt.streams.get_lowest_resolution()
        print(f"Downloading Video for {link}")
        stream.download(output_path=output_path, filename=filename)
        iteration += 1
    except Exception as e:
        print(f"Error downloading {link}: {e}")

def main(file_dir: str, mode: str, output_path: str, limit: int):
    global iteration
    
    movies_dir = glob(os.path.join(file_dir, mode, "*.txt"))
    output_path = os.path.join(output_path, mode)
    os.makedirs(output_path, exist_ok=True)

    for file in tqdm(movies_dir):
        with open(file, 'r') as f:
            line = f.readline().strip()
        if not line:
            print(f"No URL found in {file}. Skipping.")
            continue
        file_name, _ = os.path.splitext(os.path.basename(file))
        DownloadVideo(link=line, output_path=output_path, filename=file_name + ".mp4")

        if iteration >= limit:
            print("Designated video limit reached. Stopping downloading...")
            break

    print("Downloading Job Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Directory to RealEstate10K folder")
    parser.add_argument("--mode", type=str, required=True, help="'train' or 'val' split")
    parser.add_argument("--output_path", type=str, required=True, help="Directory to save videos")
    parser.add_argument("--limit", type=int, default=100, help="Limit of RealEstate10k downloads")

    args = parser.parse_args()

    main(args.input_dir, args.mode, args.output_path, args.limit)
